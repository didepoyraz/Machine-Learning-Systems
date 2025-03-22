import asyncio
import multiprocessing
import time
from typing import Any, Dict, List

from openai import AsyncOpenAI

from tracestorm.logger import init_logger

logger = init_logger(__name__)


class RequestError(Exception):
    """Custom exception for request failures."""


class TracePlayer:
    """
    A TracePlayer handles a sub-trace of requests by:
      1. Initializing its own AsyncOpenAI client.
      2. Scheduling requests at given timestamps (trace).
      3. Dispatching requests to an async worker (sender_worker).
      4. Sending results back via an IPC queue.

    Args:
        name (str): Unique identifier for this TracePlayer (e.g., 'player-1').
        trace (List[int]): List of timestamps (in ms) at which requests should be sent.
        requests (List[Dict[str, Any]]): The request payloads corresponding to each timestamp.
        base_url (str): Base URL for the OpenAI API.
        api_key (str): API key for OpenAI authentication.
        ipc_queue (multiprocessing.Queue): IPC queue for returning results to the main process.
    """

    def __init__(
        self,
        name: str,
        trace: List[int],
        requests: List[Dict[str, Any]],
        base_url: str,
        api_key: str,
        ipc_queue: multiprocessing.Queue,
        max_queue_size: int = 1_000_000,
    ) -> None:
        if len(trace) != len(requests):
            raise ValueError(
                f"[{name}] Mismatch in trace ({len(trace)}) and requests ({len(requests)}) length."
            )

        self.name = name
        self.trace = trace
        self.requests = requests
        self.ipc_queue = ipc_queue
        self.shutdown_flag = asyncio.Event()

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.dispatch_queue: asyncio.Queue = asyncio.Queue(
            maxsize=max_queue_size
        )

    async def request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a single request to the API, measure time and token metrics, and return the results.

        Args:
            request_data (Dict[str, Any]): The data/payload for the chat completion request.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "result": The accumulated content string from the response.
                - "token_count": The number of chunks/tokens received.
                - "time_records": Timestamps indicating when data was received.
                - "error": None if success, otherwise the error message.

        Raises:
            RequestError: If the request fails for any reason (network, API error, etc.).
        """
        result = ""
        token_count = 0
        time_records = [time.time()]

        try:
            stream = await self.client.chat.completions.create(**request_data)
            async for completion in stream:
                # If the chunk is empty or the structure is unexpected, log and skip.
                if not completion.choices or not completion.choices[0].delta:
                    logger.warning(
                        f"[{self.name}] Empty or malformed completion data encountered."
                    )
                    continue

                tokens = completion.choices[0].delta.content
                if tokens:
                    token_count += 1
                    result += tokens
                    time_records.append(time.time())

        except Exception as e:
            logger.error(f"[{self.name}] Request failed: {e}", exc_info=True)
            raise RequestError(str(e)) from e

        return {
            "result": result,
            "token_count": token_count,
            "time_records": time_records,
            "error": None,
        }

    async def sender_worker(self) -> None:
        """
        A worker that consumes requests from the `dispatch_queue` and sends them using `request`.
        Continues until `shutdown_flag` is set.
        """
        while not (self.shutdown_flag.is_set() and self.dispatch_queue.empty()):
            try:
                # Attempt to get a queued item; if none are available quickly, re-check shutdown_flag.
                item = await asyncio.wait_for(
                    self.dispatch_queue.get(), timeout=0.1
                )
            except asyncio.TimeoutError:
                continue  # No tasks currently, keep looping.

            timestamp, request_data = item
            logger.info(
                f"[{self.name}] Sending request scheduled at {timestamp} ms"
            )

            try:
                res = await self.request(request_data)
            except RequestError as re:
                res = {
                    "result": None,
                    "token_count": 0,
                    "time_records": [],
                    "error": str(re),
                }

            # Return the result to the main process.
            self.ipc_queue.put((self.name, timestamp, res))
            self.dispatch_queue.task_done()

    async def schedule_requests(self) -> None:
        """
        Schedule requests to be sent at their specified timestamps (from `trace`).
        """
        start_time = time.time()
        for i, scheduled_time in enumerate(self.trace):
            delay = float(scheduled_time) / 1000 - (time.time() - start_time)
            if delay > 0:
                await asyncio.sleep(delay)

            request_data = self.requests[i]
            # We put both the scheduled timestamp and the request data into the queue.
            await self.dispatch_queue.put((scheduled_time, request_data))

        # Signal to the sender_worker that no more requests will be scheduled.
        self.shutdown_flag.set()

    async def run(self, concurrency: int = 10) -> None:
        """
        Main entry point for the TracePlayer's asynchronous execution:
          1. Start the specified number of sender_worker tasks.
          2. Schedule requests according to `trace`.
          3. Wait for all requests in the dispatch queue to be processed.
          4. Cancel the sender_worker tasks after everything is done.

        Args:
            concurrency (int): Number of concurrent sender_worker tasks. Defaults to 10.
        """
        if not isinstance(concurrency, int) or concurrency < 1:
            raise ValueError("concurrency must be a positive integer")

        # Create sender_worker tasks based on concurrency level.
        sender_tasks = [
            asyncio.create_task(self.sender_worker())
            for _ in range(concurrency)
        ]
        logger.info(f"[{self.name}] Started {concurrency} sender_worker tasks.")

        await self.schedule_requests()
        await (
            self.dispatch_queue.join()
        )  # Wait until all queued items are processed.
        await asyncio.sleep(0.1)  # Short grace period before shutdown.

        # Cancel all sender_worker tasks.
        for task in sender_tasks:
            task.cancel()

        # Ensure all sender_worker tasks are properly canceled.
        for task in sender_tasks:
            try:
                await task
            except asyncio.CancelledError:
                logger.info(
                    f"[{self.name}] Sender worker task canceled. All requests processed."
                )


def play(
    name: str,
    request_trace: List[int],
    request_data: List[Dict[str, Any]],
    base_url: str,
    api_key: str,
    ipc_queue: multiprocessing.Queue,
) -> None:
    """
    Entry point for running a TracePlayer in a subprocess.

    Args:
        name (str): Unique identifier for the TracePlayer.
        request_trace (List[int]): Sub-trace of timestamps.
        request_data (List[Dict[str, Any]]): Corresponding request payloads.
        base_url (str): Base URL for the API.
        api_key (str): API key for the API.
        ipc_queue (multiprocessing.Queue): Queue used to communicate results back to the main process.
    """
    try:
        # Run the player's event loop until completion.
        asyncio.run(
            TracePlayer(
                name, request_trace, request_data, base_url, api_key, ipc_queue
            ).run()
        )
    except Exception as e:
        logger.error(f"Exception in TracePlayer [{name}]: {e}", exc_info=True)
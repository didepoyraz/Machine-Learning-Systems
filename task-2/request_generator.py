import random
from typing import Any, Dict, List

from tracestorm.constants import DEFAULT_MESSAGES
from tracestorm.data_loader import Dataset
from tracestorm.logger import init_logger

logger = init_logger(__name__)


def generate_request(
    model_name: str,
    nums: int,
    messages: str = DEFAULT_MESSAGES,
    datasets: List[Dataset] = [],
    sort_strategy: str = "random",
    seed: int = None,
) -> List[Dict[str, Any]]:
    # generate default requests without datasets
    if not datasets:
        for _ in range(nums):
            return [
                {
                    "model": model_name,
                    "messages": [{"role": "user", "content": messages}],
                    "stream": True,
                }
                for _ in range(nums)
            ]
    else:  # Add and sort requests from the provided datasets
        dataset_samples = []

        # Total ratio to calculate number of requests for each dataset
        total_ratio = sum(dataset_obj.select_ratio for dataset_obj in datasets)

        for dataset_obj in datasets:
            num_requests = int(
                round(nums * dataset_obj.select_ratio / total_ratio)
            )

            # We don't have enough available prompts, repeat the dataset
            available_prompts = dataset_obj.length
            prompts = dataset_obj.prompts
            if num_requests > available_prompts:
                repeat_count = num_requests // available_prompts
                prompts.extend(prompts * repeat_count)

            assert len(prompts) >= num_requests

            # Store prompts with indexing for round-robin
            # For example, if ratio of dataset1 is 5, we will append 5 requests for each idx
            for i, sample in enumerate(prompts[:num_requests]):
                idx = i // dataset_obj.select_ratio
                dataset_samples.append((idx, sample))

            logger.info(
                f"Selected {num_requests} requests from {dataset_obj.file_name}."
            )

        # 1. Randomly sort the requests
        if sort_strategy == "random":
            if seed is not None:
                random.seed(seed)
            random.shuffle(dataset_samples)
        elif sort_strategy == "original":  # 2. original order
            dataset_samples.sort(key=lambda x: x[0])
        else:
            raise ValueError(f"Unknown sorting strategy: {sort_strategy}")

        # Extract the prompts from the list
        requests = [
            {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            }
            for _, prompt in dataset_samples
        ]

    return requests
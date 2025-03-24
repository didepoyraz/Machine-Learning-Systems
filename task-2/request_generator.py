import time
import requests
from typing import Dict, Any, List

class RequestGenerator:
    def __init__(self, base_url: str, queries: List[str]):
        """Initialize the request generator with base URL and sample queries."""
        self.base_url = base_url
        self.queries = queries
    
    def send_request(self, endpoint: str, query: str, k: int = 2) -> Dict[str, Any]:
        """
        Send a single request to a given RAG service endpoint and measure its performance.
        """
        try:
            start_time = time.time()
            response = requests.post(
                endpoint,
                json={"query": query, "k": k},
                timeout=30  # Add timeout to prevent hanging
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                result["client_time"] = end_time - start_time
                result["success"] = True
                return result
            else:
                return {
                    "success": False,
                    "client_time": end_time - start_time,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {
                "success": False,
                "client_time": time.time() - start_time if 'start_time' in locals() else -1,
                "error": str(e)
            }
    
    def request_executor(self, request_id: int, endpoint: str, arrival_time: float) -> Dict[str, Any]:
        """
        Execute individual requests with a specific arrival time and capture the response.
        """
        # Wait until the arrival time
        current_time = time.time()
        if current_time < arrival_time:
            time.sleep(arrival_time - current_time)
        
        # Select a query (round-robin from query list)
        query = self.queries[request_id % len(self.queries)]
        
        # Send the request
        result = self.send_request(endpoint, query)
        result["request_id"] = request_id
        result["arrival_time"] = arrival_time
        
        return result
    
    def generate_trace_test(self, endpoint: str, trace_config: Dict[str, Any], generator) -> List[Dict[str, Any]]:
        """
        Run a trace test, simulating a realistic workload.
        It generates request arrival times and executes requests according to the specified rate pattern.
        """
        # Generate request arrival times
        request_profile = generator.generate()
        
        # Execute requests according to the trace
        results = []
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=trace_config.get("max_concurrency", 20)) as executor:
            futures = []
            
            for i, arrival_time in enumerate(request_profile.arrival_times):
                futures.append(
                    executor.submit(
                        self.request_executor, 
                        i, 
                        endpoint, 
                        arrival_time
                    )
                )
            
            # Collect results
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                    # Print progress indicator
                    if len(results) % 10 == 0:
                        print(f"Completed {len(results)}/{len(futures)} requests")
                except Exception as e:
                    print(f"Error processing request: {e}")
        
        return results

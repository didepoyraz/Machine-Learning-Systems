import argparse
import json
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from trace_generator import TraceGenerator, RequestProfile, Metrics
from concurrent.futures import ThreadPoolExecutor
import requests
from typing import Dict, List, Any, Tuple
import numpy as np

class RAGServiceTester:
    def __init__(self, host: str = "localhost", port: int = 8000, output_dir: str = "results"):
        """Initialize the RAG service tester."""
        self.base_url = f"http://{host}:{port}"
        self.endpoint = f"{self.base_url}/rag"
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample queries for testing
        self.queries = [
            "Tell me about cats",
            "What are dogs?",
            "Which animals can hover in the air?",
            "What are some common pets?",
            "Tell me about birds",
            "How do animals communicate?",
            "What do hummingbirds eat?",
            "Are all mammals warm-blooded?",
            "What's the difference between reptiles and amphibians?",
            "How do marine mammals breathe?"
        ]
    
    def send_request(self, endpoint: str, query: str, k: int = 2) -> Dict[str, Any]:
        """
        Sending a single request to a given RAG service endpoint and measuring its performance.
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
        Executing individual requests with a specific arrival time and capturing the response.
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
    
    def run_trace_test(self, trace_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Running a trace test, which simulates a realistic workload using TracéStorm. 
        It generates a sequence of request arrival times and uses them to execute the requests according to the specified rate pattern.
        """
        # Configure TracéStorm
        generator = TraceGenerator(
            duration=trace_config["duration"],
            rate_pattern=trace_config["rate_pattern"],
            base_rate=trace_config["base_rate"],
            amplitude=trace_config.get("amplitude", 0.5),
            noise=trace_config.get("noise", 0.1)
        )
        
        # Generate request arrival times
        request_profile = generator.generate()
        
        # Execute requests according to the trace
        results = []
        with ThreadPoolExecutor(max_workers=trace_config.get("max_concurrency", 20)) as executor:
            futures = []
            
            for i, arrival_time in enumerate(request_profile.arrival_times):
                futures.append(
                    executor.submit(
                        self.request_executor, 
                        i, 
                        self.endpoint, 
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
    
    def analyze_results(self, results: List[Dict[str, Any]], test_name: str) -> Dict[str, Any]:
        """
        Analyzes the raw results from the test, calculating important metrics 
        latency, success rate, throughput, etc., based on the successful requests.
        """
        # Filter successful requests
        successful_results = [r for r in results if r.get("success", False)]
        
        # Calculate basic metrics
        client_times = [r["client_time"] for r in successful_results]
        
        if not client_times:
            return {
                "test_name": test_name,
                "total_requests": len(results),
                "successful_requests": 0,
                "success_rate": 0,
                "error": "No successful requests"
            }
        
        # Calculate time-based metrics
        metrics = {
            "test_name": test_name,
            "total_requests": len(results),
            "successful_requests": len(successful_results),
            "success_rate": len(successful_results) / len(results) if results else 0,
            "avg_latency": np.mean(client_times),
            "median_latency": np.median(client_times),
            "p95_latency": np.percentile(client_times, 95),
            "p99_latency": np.percentile(client_times, 99),
            "min_latency": np.min(client_times),
            "max_latency": np.max(client_times),
        }
        
        # Convert arrival times to relative times for throughput calculation
        if successful_results:
            arrival_times = [r.get("arrival_time", 0) for r in successful_results]
            if arrival_times:
                min_time = min(arrival_times)
                max_time = max(arrival_times)
                duration = max_time - min_time
                if duration > 0:
                    metrics["throughput"] = len(successful_results) / duration
                else:
                    metrics["throughput"] = float('inf')
            else:
                metrics["throughput"] = 0
        else:
            metrics["throughput"] = 0
        
        return metrics
    
    def save_results(self, results: List[Dict[str, Any]], metrics: Dict[str, Any], test_name: str):
        """
        Saves the raw results, metrics, and visualizations to files for further analysis and reporting.
        """
        # Save raw results
        with open(f"{self.output_dir}/{test_name}_raw_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Save metrics
        with open(f"{self.output_dir}/{test_name}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Create DataFrame from successful results for visualization
        successful_results = [r for r in results if r.get("success", False)]
        if successful_results:
            df = pd.DataFrame(successful_results)
            
            # Sort by arrival time
            if "arrival_time" in df.columns:
                df = df.sort_values("arrival_time")
                # Convert to relative time
                min_time = df["arrival_time"].min()
                df["relative_time"] = df["arrival_time"] - min_time
            
            # Save DataFrame for later analysis
            df.to_csv(f"{self.output_dir}/{test_name}_results.csv", index=False)
            
            # Create latency plot
            self.create_latency_plot(df, test_name)
    
    def create_latency_plot(self, df: pd.DataFrame, test_name: str):
        """
        Generates two types of latency visualizations: 
        One showing latency over time and the other showing the latency distribution.
        """
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot 1: Latency over time
            if "relative_time" in df.columns:
                plt.subplot(2, 1, 1)
                plt.scatter(df["relative_time"], df["client_time"], alpha=0.6)
                plt.xlabel("Relative Time (s)")
                plt.ylabel("Latency (s)")
                plt.title(f"Latency over Time - {test_name}")
                
                # Add moving average
                window_size = min(50, len(df) // 10) if len(df) > 10 else 1
                if window_size > 0:
                    df = df.sort_values("relative_time")
                    df["moving_avg"] = df["client_time"].rolling(window=window_size).mean()
                    plt.plot(df["relative_time"], df["moving_avg"], color='red', linewidth=2)
            
            # Plot 2: Latency distribution
            plt.subplot(2, 1, 2)
            sns.histplot(df["client_time"], kde=True)
            plt.xlabel("Latency (s)")
            plt.ylabel("Count")
            plt.title(f"Latency Distribution - {test_name}")
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/{test_name}_latency.png")
            plt.close()
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    def create_final_report(self, all_results: Dict[str, Any]):
        """Create a final HTML report summarizing all test results."""
        # Extract metrics for each pattern
        patterns = list(all_results.keys())
        
        # Create an HTML report
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Service Base Performance Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .figure {{ margin: 20px 0; text-align: center; }}
                .figure img {{ max-width: 100%; }}
            </style>
        </head>
        <body>
            <h1>RAG Service Base Performance Test Report</h1>
            <p>Test Date: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Summary</h2>
            <table>
                <tr>
                    <th>Workload Pattern</th>
                    <th>Total Requests</th>
                    <th>Success Rate</th>
                    <th>Avg Latency (s)</th>
                    <th>P95 Latency (s)</th>
                    <th>Throughput (req/s)</th>
                </tr>
        """
        
        # Add rows for each pattern
        for pattern in patterns:
            metrics = all_results[pattern]
            html_report += f"""
                <tr>
                    <td><b>{pattern}</b></td>
                    <td>{metrics["total_requests"]}</td>
                    <td>{metrics["success_rate"]:.2%}</td>
                    <td>{metrics["avg_latency"]:.4f}</td>
                    <td>{metrics["p95_latency"]:.4f}</td>
                    <td>{metrics["throughput"]:.4f}</td>
                </tr>
            """
        
        # Add detailed results for each pattern
        html_report += """
            </table>
            
            <h2>Detailed Results by Pattern</h2>
        """
        
        # Add individual pattern results
        for pattern in patterns:
            html_report += f"""
            <h3>{pattern} Pattern</h3>
            <div class="figure">
                <img src="{pattern}_latency.png" alt="{pattern} Latency">
                <p>Latency distribution for {pattern} workload</p>
            </div>
            """
        
        # Close HTML
        html_report += """
            <h2>Conclusion</h2>
            <p>
                This report shows the performance of the base RAG implementation across different workload patterns.
                These results can be used as a baseline for comparing optimized implementations.
            </p>
        </body>
        </html>
        """
        
        # Save the HTML report
        with open(f"{self.output_dir}/base_performance_report.html", "w") as f:
            f.write(html_report)
        
        print(f"Final report generated at {self.output_dir}/base_performance_report.html")
    
    def run_test_patterns(self):
        """
        Defines several test patterns with specific configurations. These patterns are:
        -- Steady Low: Constant request rate of 5 requests per second (RPS) for 1 minute.
        -- Steady High: Constant request rate of 20 RPS for 1 minute.
        -- Burst: A workload with spikes in requests, with a base rate of 5 RPS and amplitude of 15 (leading to spikes of up to 20 RPS).
        -- Diurnal: A sine wave pattern of requests varying between 2 and 18 RPS for 2 minutes, simulating a daily load cycle with some noise.
        """
        # Define test patterns
        test_patterns = [
            {
                "name": "steady_low",
                "config": {
                    "duration": 60,  # 1 minute
                    "rate_pattern": "constant",
                    "base_rate": 5,  # 5 requests per second
                    "max_concurrency": 10
                }
            },
            {
                "name": "steady_high",
                "config": {
                    "duration": 60,  # 1 minute
                    "rate_pattern": "constant",
                    "base_rate": 20,  # 20 requests per second
                    "max_concurrency": 30
                }
            },
            {
                "name": "burst",
                "config": {
                    "duration": 90,  # 1.5 minutes
                    "rate_pattern": "spike",
                    "base_rate": 5,
                    "amplitude": 15,  # Spike up to 20 RPS
                    "max_concurrency": 30
                }
            },
            {
                "name": "diurnal",
                "config": {
                    "duration": 120,  # 2 minutes
                    "rate_pattern": "sine",
                    "base_rate": 10,
                    "amplitude": 8,  # Vary between 2-18 RPS
                    "noise": 0.2,
                    "max_concurrency": 30
                }
            }
        ]
        
        all_results = {}
        
        for pattern in test_patterns:
            pattern_name = pattern["name"]
            config = pattern["config"]
            
            print(f"\n===== Running '{pattern_name}' pattern test =====")
            
            # Test base implementation
            results = self.run_trace_test(config)
            metrics = self.analyze_results(results, pattern_name)
            self.save_results(results, metrics, pattern_name)
            
            all_results[pattern_name] = metrics
            
            print(f"Completed '{pattern_name}' pattern test")
        
        # Save overall results summary
        with open(f"{self.output_dir}/base_results_summary.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        # Create final report
        self.create_final_report(all_results)

def main():
    parser = argparse.ArgumentParser(description="Test RAG service base performance using TracéStorm")
    parser.add_argument("--host", type=str, default="localhost", help="Host of the RAG service")
    parser.add_argument("--port", type=int, default=8000, help="Port of the RAG service")
    parser.add_argument("--output", type=str, default="base_results", help="Output directory for results")
    args = parser.parse_args()
    
    # Instantiate the tester class
    tester = RAGServiceTester(
        host=args.host,
        port=args.port,
        output_dir=args.output
    )
    
    tester.run_test_patterns()  # Run the tests

if __name__ == "__main__":
    main()
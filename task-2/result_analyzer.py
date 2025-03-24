import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import List, Dict, Any

class ResultAnalyzer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def analyze_results(self, results: List[Dict[str, Any]], test_name: str) -> Dict[str, Any]:
        """
        Analyzes the raw results from the test, calculating important metrics 
        like latency, success rate, throughput, etc., based on the successful requests.
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
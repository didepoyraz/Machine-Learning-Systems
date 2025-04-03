# testing trying to add datasets

import argparse
import json
import os

# Import our custom modules
from trace_generator import TraceGenerator
from request_generator import RequestGenerator
from result_analyzer import ResultAnalyzer

# import dataset:
from data_loader import load_datasets

class RAGServiceTester:
    def __init__(self, host: str = "localhost", port: int = 8002, output_dir: str = "results"):
        """Initialize the RAG service tester."""
        self.base_url = f"http://{host}:{port}"
        self.endpoint = f"{self.base_url}/rag"
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Load datasets dynamically
        datasets, sort_strategy = load_datasets("datasets_config_default.json")

        # Extract prompts from all datasets and flatten into a single list
        self.queries = [prompt for dataset in datasets for prompt in dataset.prompts]

        # Limit number of queries if needed
        self.queries = self.queries[:1000]  # Adjust based on needs      
        
        # Sample queries for testing
        # self.queries = [
        #    "Tell me about cats",
        #    "What are dogs?",
        #    "Which animals can hover in the air?",
        #    "What are some common pets?",
        #    "Tell me about birds",
        #    "How do animals communicate?",
        #    "What do hummingbirds eat?",
        #    "Are all mammals warm-blooded?",
        #    "What's the difference between reptiles and amphibians?",
        #    "How do marine mammals breathe?"
        # ]
        
        # Initialize our request generator and result analyzer
        self.request_generator = RequestGenerator(self.base_url, self.queries)
        self.result_analyzer = ResultAnalyzer(output_dir)
    
    def run_trace_test(self, trace_config):
        """Run a trace test using our request generator."""
        # Configure TracéStorm
        generator = TraceGenerator(
            duration=trace_config["duration"],
            rate_pattern=trace_config["rate_pattern"],
            base_rate=trace_config["base_rate"],
            amplitude=trace_config.get("amplitude", 0.5),
            noise=trace_config.get("noise", 0.1)
        )
        
        # Use our request generator to run the trace test
        return self.request_generator.generate_trace_test(
            self.endpoint, 
            trace_config, 
            generator
        )
    
    def run_test_patterns(self):
        """
        Define several test patterns with specific configurations:
        - Steady Low: Constant rate of 5 RPS for 1 minute
        - Steady High: Constant rate of 20 RPS for 1 minute
        - Burst: Workload with spikes from 5 RPS to 20 RPS
        - Diurnal: Sine wave pattern varying between 2-18 RPS for 2 minutes
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
            metrics = self.result_analyzer.analyze_results(results, pattern_name)
            self.result_analyzer.save_results(results, metrics, pattern_name)
            
            all_results[pattern_name] = metrics
            
            print(f"Completed '{pattern_name}' pattern test")
        
        # Save overall results summary
        with open(f"{self.output_dir}/base_results_summary.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        # Create final report
        self.result_analyzer.create_final_report(all_results)

def main():
    parser = argparse.ArgumentParser(description="Test RAG service base performance using TracéStorm")
    parser.add_argument("--host", type=str, default="localhost", help="Host of the RAG service")
    parser.add_argument("--port", type=int, default=8002, help="Port of the RAG service")
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
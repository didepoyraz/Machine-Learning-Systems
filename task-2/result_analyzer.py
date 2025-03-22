import json
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tracestorm.logger import init_logger
from tracestorm.utils import get_unique_file_path

logger = init_logger(__name__)


class ResultAnalyzer:
    def __init__(self):
        """
        Initializes the ResultAnalyzer with an empty raw_results list.
        """
        self.raw_results: List[Dict[str, Any]] = []
        self.ttft: List[float] = []  # time to first token
        self.tpot: List[List[float]] = []  # time per output token

    def add_result(
        self, name: str, timestamp: int, resp: Dict[str, Any]
    ) -> None:
        self.raw_results.append(
            {"name": name, "timestamp": timestamp, "response": resp}
        )
        ttft, tpot = self.process_time_records(resp.get("time_records", []))
        if not ttft or not tpot:
            logger.error(
                f"Error processing time_records for {name} at timestamp {timestamp}"
            )
            return
        self.ttft.append(ttft)
        self.tpot.append(tpot)

    def store_raw_results(
        self, aggregated_raw_results: List[Tuple[str, int, Dict[str, Any]]]
    ) -> None:
        for name, timestamp, resp in aggregated_raw_results:
            self.add_result(name, timestamp, resp)

    def process_time_records(
        self, time_records: List[int]
    ) -> Tuple[float, List[float]]:
        """
        Returns the time to first token and time per output token.
        """
        if not time_records or len(time_records) == 1:
            logger.error("No time_records to process.")
            return None, None

        # ttot = time_records[1] - time_records[0]
        # tpot = time_records[i+1] - time_records[i], i > 1
        # use numpy for faster computation
        ts1 = np.array(time_records[:-1])
        ts2 = np.array(time_records[1:])
        latency = ts2 - ts1
        ttft = latency[0]
        tpot = latency[1:].tolist()
        return ttft, tpot

    def summarize(
        self, percentiles: List[float] = [0.25, 0.5, 0.75, 0.99]
    ) -> Dict[str, Any]:
        """
        Returns a comprehensive summary of the results, including various statistics.

        Returns:
            Dict[str, Any]: A dictionary containing statistical summaries for ttft and tpot.
        """
        if not self.ttft or not self.tpot:
            logger.error("No results to summarize.")
            return {}

        # Convert ttft and tpot to Pandas Series
        ttft_series = pd.Series(self.ttft)

        # Flatten tpot list of lists to a single Series
        tpot_flat = [item for sublist in self.tpot for item in sublist]
        tpot_series = pd.Series(tpot_flat)

        # Compute statistics for ttft
        ttft_stats = ttft_series.describe(percentiles=percentiles).to_dict()
        logger.debug(f"ttft_stats: {ttft_stats}")
        # Compute statistics for tpot
        tpot_stats = tpot_series.describe(percentiles=percentiles).to_dict()
        logger.debug(f"tpot_stats: {tpot_stats}")

        # Organize the summary
        summary = {"ttft": ttft_stats, "tpot": tpot_stats}

        return summary

    def __str__(self) -> str:
        """
        Returns a string representation of the summary in an elegant format.

        Returns:
            str: Formatted summary of ttft and tpot statistics.
        """
        summary = self.summarize()
        if not summary:
            return "No results to summarize."

        ttft_stats = summary.get("ttft", {})
        tpot_stats = summary.get("tpot", {})

        # Helper function to format statistics
        def format_stats(title: str, stats: Dict[str, float]) -> str:
            lines = [f"{title.upper()} Statistics:"]
            for key, value in stats.items():
                lines.append(f"  {key}: {value:.4f}")
            return "\n".join(lines)

        ttft_str = format_stats("ttft", ttft_stats)
        tpot_str = format_stats("tpot", tpot_stats)

        return f"{ttft_str}\n\n{tpot_str}"

    def export_json(self, file_path: str, include_raw: bool = False) -> None:
        """
        Exports the summary statistics and optionally the raw results to a JSON file.

        Args:
            file_path (str): The path to the JSON file to be created.
            include_raw (bool, optional): Whether to include raw results in the export. Defaults to False.

        Raises:
            ValueError: If there are no results to export.
            IOError: If there is an error writing to the file.
        """
        summary = self.summarize()
        if not summary:
            logger.error("No results to export.")
            raise ValueError("No results to export.")

        data_to_export = {"summary": summary}

        if include_raw:
            data_to_export["raw_results"] = self.raw_results

        file_path = get_unique_file_path(file_path)
        try:
            with open(file_path, "w") as f:
                json.dump(data_to_export, f, indent=4)
            logger.info(f"Exported data to {file_path} successfully.")
        except IOError as e:
            logger.error(f"Failed to export data to {file_path}: {e}")
            raise

    def plot_cdf(
        self, ttft_file: str = "ttft_cdf.png", tpot_file: str = "tpot_cdf.png"
    ) -> None:
        """
        Plots the Cumulative Distribution Function (CDF) of ttft and tpot and saves the figures to files.

        Args:
            ttft_file (str, optional): File path to save the ttft CDF plot. Defaults to "ttft_cdf.png".
            tpot_file (str, optional): File path to save the tpot CDF plot. Defaults to "tpot_cdf.png".

        Raises:
            ValueError: If there is no data to plot.
            IOError: If there is an error saving the plot files.
        """
        if not self.ttft and not self.tpot:
            logger.error("No data available to plot.")
            raise ValueError("No data available to plot.")

        # Set Seaborn style for enhanced aesthetics
        sns.set(style="whitegrid")

        # Plot CDF for TTFT
        if self.ttft:
            try:
                plt.figure(figsize=(8, 6))
                sns.ecdfplot(self.ttft, color="blue")
                plt.title("CDF of Time to First Token (TTFT)")
                plt.xlabel("TTFT")
                plt.ylabel("Cumulative Probability")
                plt.tight_layout()
                ttft_file = get_unique_file_path(ttft_file)
                plt.savefig(ttft_file)
                logger.info(f"TTFT CDF plot saved to {ttft_file}.")
                plt.close()
            except Exception as e:
                logger.error(f"Failed to plot TTFT CDF: {e}")
                raise

        # Flatten tpot list of lists to a single list for plotting
        tpot_flat = [item for sublist in self.tpot for item in sublist]
        if tpot_flat:
            try:
                plt.figure(figsize=(8, 6))
                sns.ecdfplot(tpot_flat, color="green")
                plt.title("CDF of Time per Output Token (TPOT)")
                plt.xlabel("TPOT")
                plt.ylabel("Cumulative Probability")
                plt.tight_layout()
                tpot_file = get_unique_file_path(tpot_file)
                plt.savefig(tpot_file)
                logger.info(f"TPOT CDF plot saved to {tpot_file}.")
                plt.close()
            except Exception as e:
                logger.error(f"Failed to plot TPOT CDF: {e}")
                raise
"""
Visualization Tools

This module provides visualization tools for RAG evaluation results.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Union, Optional, Tuple


class RAGVisualizer:
    """
    A class for visualizing RAG evaluation results.
    """
    
    def __init__(
        self,
        figure_size: Tuple[int, int] = (10, 6),
        style: str = "whitegrid",
        context: str = "paper",
        palette: str = "viridis",
        font_scale: float = 1.2,
        save_format: str = "png",
        dpi: int = 300,
    ):
        """
        Initialize the RAG visualizer.
        
        Args:
            figure_size: Default figure size (width, height)
            style: Seaborn style
            context: Seaborn context
            palette: Color palette to use
            font_scale: Font scale factor
            save_format: Format for saving figures
            dpi: DPI for saved figures
        """
        self.figure_size = figure_size
        self.style = style
        self.context = context
        self.palette = palette
        self.font_scale = font_scale
        self.save_format = save_format
        self.dpi = dpi
        
        # Set default style
        sns.set_style(style)
        sns.set_context(context, font_scale=font_scale)
        sns.set_palette(palette)
    
    def plot_retrieval_metrics(
        self,
        retrieval_results: Dict[str, Any],
        output_dir: Optional[str] = None,
        filename_prefix: str = "retrieval",
        show_plots: bool = True,
    ) -> Dict[str, str]:
        """
        Plot retrieval evaluation metrics.
        
        Args:
            retrieval_results: Results from retrieval evaluation
            output_dir: Directory to save the plots (None = don't save)
            filename_prefix: Prefix for filenames
            show_plots: Whether to display the plots
            
        Returns:
            Dictionary mapping plot names to file paths (if saved)
        """
        plot_files = {}
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Precision, Recall, F1 at different k values
        if all(metric in retrieval_results for metric in ["precision", "recall", "f1"]):
            plt.figure(figsize=self.figure_size)
            
            # Extract k values and scores
            k_values = [int(k.replace("@", "")) for k in retrieval_results["precision"].keys()]
            precision_values = [retrieval_results["precision"][f"@{k}"] for k in k_values]
            recall_values = [retrieval_results["recall"][f"@{k}"] for k in k_values]
            f1_values = [retrieval_results["f1"][f"@{k}"] for k in k_values]
            
            # Create the plot
            plt.plot(k_values, precision_values, 'o-', label='Precision')
            plt.plot(k_values, recall_values, 's-', label='Recall')
            plt.plot(k_values, f1_values, '^-', label='F1')
            
            plt.title('Retrieval Performance at Different k Values')
            plt.xlabel('k (Number of Retrieved Documents)')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.ylim(0, 1.05)
            
            # Save the plot if output directory is provided
            if output_dir:
                filepath = os.path.join(output_dir, f"{filename_prefix}_precision_recall_f1.{self.save_format}")
                plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                plot_files["precision_recall_f1"] = filepath
            
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        # Context relevance scores
        if "context_relevance" in retrieval_results and retrieval_results["context_relevance"]:
            plt.figure(figsize=self.figure_size)
            
            # Extract data
            query_indices = []
            avg_scores = []
            all_scores = []
            query_labels = []
            
            for i, item in enumerate(retrieval_results["context_relevance"]):
                query_indices.append(i)
                avg_scores.append(item["average"])
                all_scores.extend(item["scores"])
                query_labels.append(f"Q{i+1}")
            
            # Bar chart of average relevance scores by query
            plt.bar(query_indices, avg_scores, alpha=0.7)
            plt.axhline(y=sum(avg_scores) / len(avg_scores), color='r', linestyle='--', label='Average')
            
            plt.title('Average Context Relevance by Query')
            plt.xlabel('Query')
            plt.ylabel('Average Relevance Score')
            plt.xticks(query_indices, query_labels)
            plt.ylim(0, 1.05)
            plt.legend()
            
            # Save the plot if output directory is provided
            if output_dir:
                filepath = os.path.join(output_dir, f"{filename_prefix}_context_relevance.{self.save_format}")
                plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                plot_files["context_relevance"] = filepath
            
            if show_plots:
                plt.show()
            else:
                plt.close()
            
            # Distribution of all relevance scores
            plt.figure(figsize=self.figure_size)
            sns.histplot(all_scores, kde=True, bins=20)
            plt.title('Distribution of Context Relevance Scores')
            plt.xlabel('Relevance Score')
            plt.ylabel('Frequency')
            
            # Save the plot if output directory is provided
            if output_dir:
                filepath = os.path.join(output_dir, f"{filename_prefix}_relevance_distribution.{self.save_format}")
                plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                plot_files["relevance_distribution"] = filepath
            
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        # MAP score
        if "map" in retrieval_results:
            plt.figure(figsize=self.figure_size)
            
            plt.bar(["MAP"], [retrieval_results["map"]], color='blue', alpha=0.7)
            
            plt.title('Mean Average Precision (MAP)')
            plt.ylabel('Score')
            plt.ylim(0, 1.05)
            plt.grid(axis='y', alpha=0.3)
            
            # Save the plot if output directory is provided
            if output_dir:
                filepath = os.path.join(output_dir, f"{filename_prefix}_map.{self.save_format}")
                plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                plot_files["map"] = filepath
            
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        return plot_files
    
    def plot_generation_metrics(
        self,
        generation_results: Dict[str, Any],
        output_dir: Optional[str] = None,
        filename_prefix: str = "generation",
        show_plots: bool = True,
    ) -> Dict[str, str]:
        """
        Plot generation evaluation metrics.
        
        Args:
            generation_results: Results from generation evaluation
            output_dir: Directory to save the plots (None = don't save)
            filename_prefix: Prefix for filenames
            show_plots: Whether to display the plots
            
        Returns:
            Dictionary mapping plot names to file paths (if saved)
        """
        plot_files = {}
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Plot overall scores
        if "overall_scores" in generation_results:
            plt.figure(figsize=self.figure_size)
            
            metrics = []
            scores = []
            
            for metric, score in generation_results["overall_scores"].items():
                if metric != "overall":  # We'll use overall score separately
                    metrics.append(metric)
                    scores.append(score)
            
            # Sort by score for better visualization
            metrics_sorted = [x for _, x in sorted(zip(scores, metrics), reverse=True)]
            scores_sorted = sorted(scores, reverse=True)
            
            # Create the bar chart
            bars = plt.barh(metrics_sorted, scores_sorted, alpha=0.7)
            
            # Add overall score as a vertical line if available
            if "overall" in generation_results["overall_scores"]:
                plt.axvline(x=generation_results["overall_scores"]["overall"], 
                            color='red', linestyle='--', linewidth=2,
                            label=f'Overall: {generation_results["overall_scores"]["overall"]:.2f}')
                plt.legend()
            
            plt.title('Generation Metrics - Overall Scores')
            plt.xlabel('Score')
            plt.ylabel('Metric')
            plt.xlim(0, 1.05)
            
            # Add score values at the end of bars
            for i, bar in enumerate(bars):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{scores_sorted[i]:.2f}', va='center')
            
            # Save the plot if output directory is provided
            if output_dir:
                filepath = os.path.join(output_dir, f"{filename_prefix}_overall_scores.{self.save_format}")
                plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                plot_files["overall_scores"] = filepath
            
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        # Plot faithfulness scores per query
        if "faithfulness" in generation_results:
            plt.figure(figsize=self.figure_size)
            
            queries = []
            scores = []
            
            for item in generation_results["faithfulness"]:
                queries.append(f"Q{len(queries)+1}")
                scores.append(item["score"])
            
            plt.bar(queries, scores, alpha=0.7)
            plt.axhline(y=sum(scores) / len(scores), color='r', linestyle='--', label='Average')
            
            plt.title('Faithfulness Scores by Query')
            plt.xlabel('Query')
            plt.ylabel('Faithfulness Score')
            plt.ylim(0, 1.05)
            plt.legend()
            
            # Save the plot if output directory is provided
            if output_dir:
                filepath = os.path.join(output_dir, f"{filename_prefix}_faithfulness.{self.save_format}")
                plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                plot_files["faithfulness"] = filepath
            
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        # Plot hallucination analysis
        if "hallucination" in generation_results:
            plt.figure(figsize=self.figure_size)
            
            queries = []
            hallucination_scores = []
            
            for item in generation_results["hallucination"]:
                queries.append(f"Q{len(queries)+1}")
                hallucination_scores.append(item["hallucination_score"])
            
            plt.bar(queries, hallucination_scores, alpha=0.7, color='red')
            plt.axhline(y=sum(hallucination_scores) / len(hallucination_scores), 
                        color='black', linestyle='--', label='Average')
            
            plt.title('Hallucination Scores by Query')
            plt.xlabel('Query')
            plt.ylabel('Hallucination Score')
            plt.ylim(0, 1.05)
            plt.legend()
            
            # Save the plot if output directory is provided
            if output_dir:
                filepath = os.path.join(output_dir, f"{filename_prefix}_hallucination.{self.save_format}")
                plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                plot_files["hallucination"] = filepath
            
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        # Multi-metric comparison across queries
        metrics_to_plot = ["faithfulness", "answer_relevance", "factuality", "coherence", "conciseness"]
        metrics_data = {}
        
        for metric_name in metrics_to_plot:
            if metric_name in generation_results:
                metrics_data[metric_name] = [item["score"] for item in generation_results[metric_name]]
        
        if metrics_data:
            plt.figure(figsize=(max(self.figure_size[0], len(metrics_data) * 3), self.figure_size[1]))
            
            # Number of metrics and queries
            num_metrics = len(metrics_data)
            num_queries = len(list(metrics_data.values())[0])
            
            # Set up the bar positions
            bar_width = 0.7 / num_metrics
            index = np.arange(num_queries)
            
            # Create bars for each metric
            for i, (metric_name, scores) in enumerate(metrics_data.items()):
                position = index + (i - num_metrics/2 + 0.5) * bar_width
                plt.bar(position, scores, bar_width, alpha=0.7, label=metric_name.capitalize())
            
            plt.title('Generation Metrics by Query')
            plt.xlabel('Query')
            plt.ylabel('Score')
            plt.xticks(index, [f"Q{i+1}" for i in range(num_queries)])
            plt.ylim(0, 1.05)
            plt.legend()
            
            # Save the plot if output directory is provided
            if output_dir:
                filepath = os.path.join(output_dir, f"{filename_prefix}_metrics_by_query.{self.save_format}")
                plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                plot_files["metrics_by_query"] = filepath
            
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        return plot_files
    
    def plot_end_to_end_metrics(
        self,
        end_to_end_results: Dict[str, Any],
        output_dir: Optional[str] = None,
        filename_prefix: str = "end_to_end",
        show_plots: bool = True,
    ) -> Dict[str, str]:
        """
        Plot end-to-end evaluation metrics.
        
        Args:
            end_to_end_results: Results from end-to-end evaluation
            output_dir: Directory to save the plots (None = don't save)
            filename_prefix: Prefix for filenames
            show_plots: Whether to display the plots
            
        Returns:
            Dictionary mapping plot names to file paths (if saved)
        """
        plot_files = {}
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Task completion rate
        if "task_completion" in end_to_end_results:
            plt.figure(figsize=self.figure_size)
            
            plt.bar(["Task Completion Rate"], [end_to_end_results["task_completion"]], alpha=0.7)
            
            plt.title('Task Completion Rate')
            plt.ylabel('Rate')
            plt.ylim(0, 1.05)
            plt.grid(axis='y', alpha=0.3)
            
            # Add completion rate text
            plt.text(0, end_to_end_results["task_completion"] / 2, 
                     f"{end_to_end_results['task_completion']:.2f}", 
                     ha='center', va='center', fontsize=14)
            
            # Save the plot if output directory is provided
            if output_dir:
                filepath = os.path.join(output_dir, f"{filename_prefix}_task_completion.{self.save_format}")
                plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                plot_files["task_completion"] = filepath
            
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        # User satisfaction
        if "user_satisfaction" in end_to_end_results:
            user_sat = end_to_end_results["user_satisfaction"]
            
            if "distribution" in user_sat:
                plt.figure(figsize=self.figure_size)
                
                # Extract rating distribution
                ratings = []
                counts = []
                
                for rating, percentage in user_sat["distribution"].items():
                    ratings.append(int(rating))
                    counts.append(percentage)
                
                # Sort by rating
                ratings, counts = zip(*sorted(zip(ratings, counts)))
                
                # Plot the distribution
                plt.bar(ratings, counts, alpha=0.7)
                plt.axvline(x=user_sat["average"], color='r', linestyle='--', 
                            label=f'Average: {user_sat["average"]:.2f}')
                
                plt.title('User Satisfaction Distribution')
                plt.xlabel('Rating')
                plt.ylabel('Percentage')
                plt.xticks(ratings)
                plt.legend()
                
                # Save the plot if output directory is provided
                if output_dir:
                    filepath = os.path.join(output_dir, f"{filename_prefix}_user_satisfaction.{self.save_format}")
                    plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                    plot_files["user_satisfaction"] = filepath
                
                if show_plots:
                    plt.show()
                else:
                    plt.close()
            
            # Simple user satisfaction metrics
            plt.figure(figsize=self.figure_size)
            
            metrics = ["average", "median", "normalized_average"]
            values = [user_sat[m] for m in metrics if m in user_sat]
            metrics = [m for m in metrics if m in user_sat]
            
            plt.bar(metrics, values, alpha=0.7)
            
            plt.title('User Satisfaction Metrics')
            plt.ylabel('Score')
            plt.ylim(0, max(values) * 1.2)
            
            # Add values on top of bars
            for i, v in enumerate(values):
                plt.text(i, v + 0.1, f"{v:.2f}", ha='center')
            
            # Save the plot if output directory is provided
            if output_dir:
                filepath = os.path.join(output_dir, f"{filename_prefix}_user_sat_metrics.{self.save_format}")
                plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                plot_files["user_sat_metrics"] = filepath
            
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        # Efficiency metrics
        if "efficiency" in end_to_end_results:
            efficiency = end_to_end_results["efficiency"]
            
            # Latency metrics
            if all(k in efficiency for k in ["latency_p50", "latency_p90", "latency_p99"]):
                plt.figure(figsize=self.figure_size)
                
                percentiles = ["p50", "p90", "p99"]
                latencies = [efficiency[f"latency_{p}"] for p in percentiles]
                
                plt.bar(percentiles, latencies, alpha=0.7)
                
                plt.title('Latency Percentiles')
                plt.xlabel('Percentile')
                plt.ylabel('Latency (seconds)')
                
                # Save the plot if output directory is provided
                if output_dir:
                    filepath = os.path.join(output_dir, f"{filename_prefix}_latency.{self.save_format}")
                    plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                    plot_files["latency"] = filepath
                
                if show_plots:
                    plt.show()
                else:
                    plt.close()
            
            # Memory usage if available
            if "memory_usage" in efficiency:
                plt.figure(figsize=self.figure_size)
                
                memory = efficiency["memory_usage"]
                
                # Check which memory metrics are available
                metrics = []
                values = []
                
                for metric in ["average", "peak", "final"]:
                    if metric in memory:
                        metrics.append(metric.capitalize())
                        values.append(memory[metric])
                
                plt.bar(metrics, values, alpha=0.7)
                
                plt.title('Memory Usage')
                plt.ylabel('Memory (MB)')
                
                # Save the plot if output directory is provided
                if output_dir:
                    filepath = os.path.join(output_dir, f"{filename_prefix}_memory_usage.{self.save_format}")
                    plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                    plot_files["memory_usage"] = filepath
                
                if show_plots:
                    plt.show()
                else:
                    plt.close()
        
        # Robustness metrics
        if "robustness" in end_to_end_results:
            robustness = end_to_end_results["robustness"]
            
            plt.figure(figsize=self.figure_size)
            
            metrics = ["standard_score", "adversarial_score", "robustness_ratio"]
            values = [robustness[m] for m in metrics if m in robustness]
            metrics = [m.replace("_", " ").title() for m in metrics if m in robustness]
            
            plt.bar(metrics, values, alpha=0.7)
            
            plt.title('Robustness Metrics')
            plt.ylabel('Score')
            
            # Save the plot if output directory is provided
            if output_dir:
                filepath = os.path.join(output_dir, f"{filename_prefix}_robustness.{self.save_format}")
                plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                plot_files["robustness"] = filepath
            
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        # Overall score if available
        if "overall_score" in end_to_end_results:
            plt.figure(figsize=self.figure_size)
            
            plt.bar(["Overall Score"], [end_to_end_results["overall_score"]], alpha=0.7)
            
            plt.title('Overall End-to-End Score')
            plt.ylabel('Score')
            plt.ylim(0, 1.05)
            
            # Add score text
            plt.text(0, end_to_end_results["overall_score"] / 2, 
                     f"{end_to_end_results['overall_score']:.2f}", 
                     ha='center', va='center', fontsize=14)
            
            # Save the plot if output directory is provided
            if output_dir:
                filepath = os.path.join(output_dir, f"{filename_prefix}_overall_score.{self.save_format}")
                plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                plot_files["overall_score"] = filepath
            
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        return plot_files
    
    def plot_human_evaluation_metrics(
        self,
        human_eval_results: Dict[str, Any],
        output_dir: Optional[str] = None,
        filename_prefix: str = "human_eval",
        show_plots: bool = True,
    ) -> Dict[str, str]:
        """
        Plot human evaluation metrics.
        
        Args:
            human_eval_results: Results from human evaluation processing
            output_dir: Directory to save the plots (None = don't save)
            filename_prefix: Prefix for filenames
            show_plots: Whether to display the plots
            
        Returns:
            Dictionary mapping plot names to file paths (if saved)
        """
        plot_files = {}
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Plot overall scores if available
        if "overall_scores" in human_eval_results:
            plt.figure(figsize=self.figure_size)
            
            criteria = []
            scores = []
            
            for criterion, score in human_eval_results["overall_scores"].items():
                criteria.append(criterion.capitalize())
                scores.append(score)
            
            # Sort by score for better visualization
            criteria_sorted = [x for _, x in sorted(zip(scores, criteria), reverse=True)]
            scores_sorted = sorted(scores, reverse=True)
            
            # Create the bar chart
            plt.barh(criteria_sorted, scores_sorted, alpha=0.7)
            
            plt.title('Human Evaluation - Overall Scores by Criterion')
            plt.xlabel('Score')
            plt.xlim(0, 5.5)  # Assuming 5-point scale
            
            # Save the plot if output directory is provided
            if output_dir:
                filepath = os.path.join(output_dir, f"{filename_prefix}_overall_scores.{self.save_format}")
                plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                plot_files["overall_scores"] = filepath
            
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        # Plot system preferences if available
        if "overall_system_preference" in human_eval_results:
            plt.figure(figsize=self.figure_size)
            
            prefs = human_eval_results["overall_system_preference"]
            
            labels = []
            percentages = []
            
            # Define preference labels
            pref_labels = {
                "a": "Strongly Prefer A",
                "slight-a": "Slightly Prefer A",
                "equal": "Equal Quality",
                "slight-b": "Slightly Prefer B",
                "b": "Strongly Prefer B"
            }
            
            # Get data by preference order
            order = ["a", "slight-a", "equal", "slight-b", "b"]
            for pref in order:
                if pref in prefs["percentages"]:
                    labels.append(pref_labels.get(pref, pref))
                    percentages.append(prefs["percentages"][pref])
            
            plt.bar(labels, percentages, alpha=0.7)
            
            plt.title('System Preference Distribution')
            plt.ylabel('Percentage')
            plt.xticks(rotation=45, ha='right')
            
            # Add winner info
            winner = prefs.get("winner")
            if winner and winner in pref_labels:
                plt.text(len(labels)/2, max(percentages) * 0.9, 
                         f"Winner: {pref_labels[winner]}", 
                         ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
            
            # Save the plot if output directory is provided
            if output_dir:
                filepath = os.path.join(output_dir, f"{filename_prefix}_system_preference.{self.save_format}")
                plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                plot_files["system_preference"] = filepath
            
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        # Plot per-query criteria scores
        if "criteria_scores" in human_eval_results:
            for criterion, query_scores in human_eval_results["criteria_scores"].items():
                plt.figure(figsize=self.figure_size)
                
                queries = []
                scores = []
                
                for query_num, data in sorted(query_scores.items()):
                    queries.append(f"Q{query_num}")
                    scores.append(data["average"])
                
                plt.bar(queries, scores, alpha=0.7)
                
                plt.title(f'{criterion.capitalize()} Scores by Query')
                plt.xlabel('Query')
                plt.ylabel('Score')
                plt.ylim(0, 5.5)  # Assuming 5-point scale
                
                # Save the plot if output directory is provided
                if output_dir:
                    filepath = os.path.join(output_dir, f"{filename_prefix}_{criterion}_by_query.{self.save_format}")
                    plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                    plot_files[f"{criterion}_by_query"] = filepath
                
                if show_plots:
                    plt.show()
                else:
                    plt.close()
        
        return plot_files
    
    def create_dashboard(
        self,
        results: Dict[str, Dict[str, Any]],
        output_file: str = "rag_evaluation_dashboard.html",
        title: str = "RAG Evaluation Dashboard",
    ) -> str:
        """
        Create an HTML dashboard with evaluation results.
        
        Args:
            results: Dictionary of evaluation results by category
            output_file: Path to save the dashboard HTML
            title: Dashboard title
            
        Returns:
            Path to the saved dashboard HTML
        """
        # Create temporary directory for plots
        plots_dir = os.path.join(os.path.dirname(output_file), "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate all plots
        plot_files = {}
        
        if "retrieval" in results:
            plot_files["retrieval"] = self.plot_retrieval_metrics(
                results["retrieval"], plots_dir, "retrieval", False
            )
        
        if "generation" in results:
            plot_files["generation"] = self.plot_generation_metrics(
                results["generation"], plots_dir, "generation", False
            )
        
        if "end_to_end" in results:
            plot_files["end_to_end"] = self.plot_end_to_end_metrics(
                results["end_to_end"], plots_dir, "end_to_end", False
            )
        
        if "human_eval" in results:
            plot_files["human_eval"] = self.plot_human_evaluation_metrics(
                results["human_eval"], plots_dir, "human_eval", False
            )
        
        # Generate HTML content
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ margin-bottom: 20px; }}
        .section {{ margin-bottom: 30px; }}
        .plots-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
        .plot-item {{ margin-bottom: 20px; }}
        .metrics-table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .metrics-table th {{ background-color: #f2f2f2; }}
        .metrics-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .nav {{ background-color: #333; overflow: hidden; margin-bottom: 20px; }}
        .nav a {{ float: left; color: white; text-align: center; padding: 14px 16px; text-decoration: none; }}
        .nav a:hover {{ background-color: #ddd; color: black; }}
        .nav a.active {{ background-color: #4CAF50; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="nav">
        <a href="#overview" class="active">Overview</a>
"""
        
        # Add navigation links
        if "retrieval" in results:
            html += '        <a href="#retrieval">Retrieval</a>\n'
        if "generation" in results:
            html += '        <a href="#generation">Generation</a>\n'
        if "end_to_end" in results:
            html += '        <a href="#end_to_end">End-to-End</a>\n'
        if "human_eval" in results:
            html += '        <a href="#human_eval">Human Evaluation</a>\n'
        
        html += """    </div>
    
    <div id="overview" class="section">
        <h2>Overview</h2>
        <table class="metrics-table">
            <tr>
                <th>Category</th>
                <th>Overall Score</th>
                <th>Key Metrics</th>
            </tr>
"""
        
        # Add overview table rows
        if "retrieval" in results:
            map_score = results["retrieval"].get("map", 0)
            html += f"""            <tr>
                <td>Retrieval</td>
                <td>{map_score:.3f}</td>
                <td>MAP, Precision@k, Recall@k, F1@k</td>
            </tr>
"""
        
        if "generation" in results and "overall_scores" in results["generation"]:
            overall = results["generation"]["overall_scores"].get("overall", 0)
            html += f"""            <tr>
                <td>Generation</td>
                <td>{overall:.3f}</td>
                <td>Faithfulness, Relevance, Factuality, Coherence</td>
            </tr>
"""
        
        if "end_to_end" in results:
            overall = results["end_to_end"].get("overall_score", 0)
            html += f"""            <tr>
                <td>End-to-End</td>
                <td>{overall:.3f}</td>
                <td>Task Completion, User Satisfaction, Efficiency</td>
            </tr>
"""
        
        if "human_eval" in results and "overall_scores" in results["human_eval"]:
            relevance = results["human_eval"]["overall_scores"].get("relevance", 0)
            html += f"""            <tr>
                <td>Human Evaluation</td>
                <td>{relevance:.3f}</td>
                <td>Relevance, Accuracy, Completeness, Coherence</td>
            </tr>
"""
        
        html += """        </table>
    </div>
"""
        
        # Add sections for each category
        for category, files in plot_files.items():
            if files:
                html += f"""    <div id="{category}" class="section">
        <h2>{category.title()} Evaluation</h2>
        <div class="plots-container">
"""
                
                for plot_name, plot_file in files.items():
                    # Use relative path for images
                    rel_path = os.path.relpath(plot_file, os.path.dirname(output_file))
                    html += f"""            <div class="plot-item">
                <h3>{plot_name.replace('_', ' ').title()}</h3>
                <img src="{rel_path}" alt="{plot_name}" width="600">
            </div>
"""
                
                html += """        </div>
    </div>
"""
        
        html += """</body>
</html>
"""
        
        # Write the HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_file 
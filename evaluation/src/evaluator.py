"""
RAG Evaluator

This module provides the main RAGEvaluator class that ties together all
components of the RAG evaluation framework.
"""

import os
import json
from typing import List, Dict, Any, Union, Optional, Set, Tuple, Callable
from datetime import datetime

from .retrieval_metrics import RetrievalMetrics
from .generation_metrics import GenerationMetrics
from .end_to_end_metrics import EndToEndMetrics
from .human_eval_tools import HumanEvaluationProcessor
from .visualization import RAGVisualizer


class RAGEvaluator:
    """
    Main class for evaluating RAG systems, integrating retrieval, generation,
    end-to-end, and human evaluation metrics.
    """
    
    def __init__(
        self,
        retrieval_metrics: Optional[List[str]] = None,
        generation_metrics: Optional[List[str]] = None,
        end_to_end_metrics: Optional[List[str]] = None,
        use_semantic_similarity: bool = True,
        semantic_model_name: str = "all-MiniLM-L6-v2",
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the RAG evaluator.
        
        Args:
            retrieval_metrics: List of retrieval metrics to compute
            generation_metrics: List of generation metrics to compute
            end_to_end_metrics: List of end-to-end metrics to compute
            use_semantic_similarity: Whether to use semantic similarity for relevance
            semantic_model_name: Name of the sentence transformer model to use
            output_dir: Directory to save evaluation results and visualizations
        """
        self.retrieval_metrics = retrieval_metrics or [
            "precision", "recall", "f1", "map", "ndcg", "context_relevance"
        ]
        self.generation_metrics = generation_metrics or [
            "faithfulness", "answer_relevance", "hallucination", "factuality",
            "coherence", "conciseness"
        ]
        self.end_to_end_metrics = end_to_end_metrics or [
            "task_completion", "user_satisfaction", "efficiency", "robustness"
        ]
        self.use_semantic_similarity = use_semantic_similarity
        self.semantic_model_name = semantic_model_name
        self.output_dir = output_dir
        
        # Initialize metric calculators
        self.retrieval_metric_calculator = RetrievalMetrics(
            use_semantic_similarity=use_semantic_similarity,
            semantic_model_name=semantic_model_name
        )
        
        self.generation_metric_calculator = GenerationMetrics(
            use_semantic_similarity=use_semantic_similarity,
            semantic_model_name=semantic_model_name
        )
        
        self.end_to_end_metric_calculator = EndToEndMetrics()
        
        self.human_eval_processor = HumanEvaluationProcessor()
        
        self.visualizer = RAGVisualizer()
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_retrieval(
        self,
        queries: List[str],
        retrieved_docs_list: List[List[str]],
        relevant_docs_list: List[List[str]],
        k_values: List[int] = [1, 3, 5, 10],
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval performance.
        
        Args:
            queries: List of query texts
            retrieved_docs_list: List of retrieved document lists for each query
            relevant_docs_list: List of relevant document lists for each query
            k_values: List of k values to calculate metrics at
            save_results: Whether to save results to file
            
        Returns:
            Dictionary of retrieval evaluation results
        """
        print("Evaluating retrieval performance...")
        
        results = self.retrieval_metric_calculator.evaluate_all(
            queries=queries,
            retrieved_docs_list=retrieved_docs_list,
            relevant_docs_list=relevant_docs_list,
            k_values=k_values
        )
        
        # Save results if requested
        if save_results and self.output_dir:
            output_file = os.path.join(self.output_dir, "retrieval_results.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"Retrieval evaluation results saved to {output_file}")
        
        return results
    
    def evaluate_generation(
        self,
        queries: List[str],
        generated_texts: List[str],
        retrieved_contexts_list: List[List[str]],
        reference_texts: Optional[List[str]] = None,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate generation performance.
        
        Args:
            queries: List of query texts
            generated_texts: List of generated answers
            retrieved_contexts_list: List of retrieved contexts for each query
            reference_texts: Optional list of reference answers
            save_results: Whether to save results to file
            
        Returns:
            Dictionary of generation evaluation results
        """
        print("Evaluating generation performance...")
        
        results = self.generation_metric_calculator.evaluate_all(
            queries=queries,
            generated_texts=generated_texts,
            retrieved_contexts_list=retrieved_contexts_list,
            reference_texts=reference_texts
        )
        
        # Save results if requested
        if save_results and self.output_dir:
            output_file = os.path.join(self.output_dir, "generation_results.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"Generation evaluation results saved to {output_file}")
        
        return results
    
    def evaluate_end_to_end(
        self,
        outputs: List[str],
        expected_outputs: List[str],
        user_ratings: Optional[List[float]] = None,
        efficiency_data: Optional[Dict[str, Any]] = None,
        robustness_data: Optional[Dict[str, Any]] = None,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate end-to-end performance.
        
        Args:
            outputs: List of system outputs
            expected_outputs: List of expected outputs
            user_ratings: Optional list of user ratings
            efficiency_data: Optional efficiency metrics data
            robustness_data: Optional robustness metrics data
            save_results: Whether to save results to file
            
        Returns:
            Dictionary of end-to-end evaluation results
        """
        print("Evaluating end-to-end performance...")
        
        results = self.end_to_end_metric_calculator.evaluate_all(
            outputs=outputs,
            expected_outputs=expected_outputs,
            user_ratings=user_ratings,
            efficiency_data=efficiency_data,
            robustness_data=robustness_data
        )
        
        # Save results if requested
        if save_results and self.output_dir:
            output_file = os.path.join(self.output_dir, "end_to_end_results.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"End-to-end evaluation results saved to {output_file}")
        
        return results
    
    def evaluate_human_feedback(
        self,
        results_file: str,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Process and evaluate human feedback.
        
        Args:
            results_file: Path to human evaluation results file
            save_results: Whether to save processed results to file
            
        Returns:
            Dictionary of human evaluation metrics
        """
        print("Processing human evaluation results...")
        
        # Load human evaluation results
        human_eval_results = self.human_eval_processor.load_results(results_file)
        
        # Calculate metrics
        metrics = self.human_eval_processor.calculate_metrics(human_eval_results)
        
        # Save results if requested
        if save_results and self.output_dir:
            output_file = os.path.join(self.output_dir, "human_eval_metrics.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
            print(f"Human evaluation metrics saved to {output_file}")
        
        return metrics
    
    def measure_efficiency(
        self,
        query_func: Callable,
        queries: List[str],
        measure_memory: bool = True,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Measure efficiency metrics.
        
        Args:
            query_func: Function that executes a query and returns results
            queries: List of queries to evaluate
            measure_memory: Whether to measure memory usage
            save_results: Whether to save results to file
            
        Returns:
            Dictionary of efficiency metrics
        """
        print("Measuring efficiency metrics...")
        
        results = self.end_to_end_metric_calculator.measure_efficiency(
            query_func=query_func,
            queries=queries,
            measure_memory=measure_memory
        )
        
        # Save results if requested
        if save_results and self.output_dir:
            output_file = os.path.join(self.output_dir, "efficiency_results.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"Efficiency metrics saved to {output_file}")
        
        return results
    
    def evaluate_robustness(
        self,
        query_func: Callable,
        standard_queries: List[str],
        adversarial_queries: List[str],
        evaluation_func: Callable,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate robustness.
        
        Args:
            query_func: Function that executes a query and returns results
            standard_queries: List of standard queries
            adversarial_queries: List of adversarial queries
            evaluation_func: Function to evaluate quality of responses
            save_results: Whether to save results to file
            
        Returns:
            Dictionary of robustness metrics
        """
        print("Evaluating robustness...")
        
        results = self.end_to_end_metric_calculator.robustness(
            query_func=query_func,
            standard_queries=standard_queries,
            adversarial_queries=adversarial_queries,
            evaluation_func=evaluation_func
        )
        
        # Save results if requested
        if save_results and self.output_dir:
            output_file = os.path.join(self.output_dir, "robustness_results.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"Robustness metrics saved to {output_file}")
        
        return results
    
    def visualize_results(
        self,
        results: Dict[str, Dict[str, Any]],
        show_plots: bool = True,
        save_dashboard: bool = True,
    ) -> Optional[str]:
        """
        Visualize evaluation results.
        
        Args:
            results: Dictionary mapping categories to results
            show_plots: Whether to display plots
            save_dashboard: Whether to save an HTML dashboard
            
        Returns:
            Path to the dashboard HTML file if saved, None otherwise
        """
        print("Visualizing evaluation results...")
        
        # Create plots directory if not exists
        plots_dir = None
        if self.output_dir:
            plots_dir = os.path.join(self.output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
        
        # Generate plots for each category
        if "retrieval" in results:
            self.visualizer.plot_retrieval_metrics(
                retrieval_results=results["retrieval"],
                output_dir=plots_dir,
                filename_prefix="retrieval",
                show_plots=show_plots
            )
        
        if "generation" in results:
            self.visualizer.plot_generation_metrics(
                generation_results=results["generation"],
                output_dir=plots_dir,
                filename_prefix="generation",
                show_plots=show_plots
            )
        
        if "end_to_end" in results:
            self.visualizer.plot_end_to_end_metrics(
                end_to_end_results=results["end_to_end"],
                output_dir=plots_dir,
                filename_prefix="end_to_end",
                show_plots=show_plots
            )
        
        if "human_eval" in results:
            self.visualizer.plot_human_evaluation_metrics(
                human_eval_results=results["human_eval"],
                output_dir=plots_dir,
                filename_prefix="human_eval",
                show_plots=show_plots
            )
        
        # Create and save dashboard if requested
        if save_dashboard and self.output_dir:
            dashboard_file = os.path.join(self.output_dir, "rag_evaluation_dashboard.html")
            self.visualizer.create_dashboard(
                results=results,
                output_file=dashboard_file,
                title="RAG Evaluation Dashboard"
            )
            print(f"Evaluation dashboard saved to {dashboard_file}")
            return dashboard_file
        
        return None
    
    def evaluate(
        self,
        rag_system: Optional[Union[str, Any]] = None,
        benchmark_dataset: Optional[str] = None,
        num_samples: int = 100,
        custom_query_func: Optional[Callable] = None,
        evaluate_retrieval: bool = True,
        evaluate_generation: bool = True,
        evaluate_end_to_end: bool = True,
        measure_system_efficiency: bool = True,
        visualize: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform a comprehensive evaluation of a RAG system.
        
        Args:
            rag_system: RAG system instance or path to load it from
            benchmark_dataset: Path to benchmark dataset
            num_samples: Number of samples to evaluate
            custom_query_func: Custom function to query the RAG system
            evaluate_retrieval: Whether to evaluate retrieval performance
            evaluate_generation: Whether to evaluate generation performance
            evaluate_end_to_end: Whether to evaluate end-to-end performance
            measure_system_efficiency: Whether to measure system efficiency
            visualize: Whether to visualize results
            
        Returns:
            Dictionary of evaluation results by category
        """
        print(f"Starting comprehensive evaluation of RAG system...")
        start_time = datetime.now()
        
        results = {}
        
        # Load benchmark dataset if provided
        if benchmark_dataset:
            print(f"Loading benchmark dataset from {benchmark_dataset}...")
            with open(benchmark_dataset, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            # Limit to num_samples
            dataset = dataset[:num_samples]
            
            # Extract data from dataset
            queries = [item['query'] for item in dataset]
            expected_answers = [item.get('answer', '') for item in dataset]
            relevant_docs = [item.get('relevant_docs', []) for item in dataset]
        else:
            print("No benchmark dataset provided. Using test queries.")
            # Use sample queries for testing
            queries = [
                "What is retrieval-augmented generation?",
                "How do transformers work?",
                "What are the limitations of large language models?",
                "Explain the concept of attention in deep learning."
            ]
            expected_answers = [""] * len(queries)
            relevant_docs = [[]] * len(queries)
        
        # Set up query function
        query_func = custom_query_func
        retrieved_docs_list = []
        context_list = []
        generated_texts = []
        
        if query_func is None and rag_system is not None:
            # Create a wrapper for the RAG system
            if isinstance(rag_system, str):
                # Assume it's a path and would be loaded by the user's code
                print("RAG system needs to be loaded. Using dummy query function.")
                
                def dummy_query_func(query):
                    return {
                        "answer": f"This is a dummy answer to: {query}",
                        "retrieved_docs": ["Doc1", "Doc2"],
                        "context": "Dummy context"
                    }
                
                query_func = dummy_query_func
            else:
                # Use the provided RAG system instance
                print("Using provided RAG system instance.")
                
                def rag_query_func(query):
                    try:
                        # Try to use a standard RAG interface
                        if hasattr(rag_system, 'query'):
                            return rag_system.query(query)
                        elif hasattr(rag_system, '__call__'):
                            return rag_system(query)
                        else:
                            raise AttributeError("RAG system has no query method or is not callable")
                    except Exception as e:
                        print(f"Error querying RAG system: {str(e)}")
                        return {
                            "answer": "",
                            "retrieved_docs": [],
                            "context": ""
                        }
                
                query_func = rag_query_func
        
        # Execute queries if we have a query function
        if query_func is not None:
            print(f"Executing {len(queries)} queries...")
            
            for query in queries:
                try:
                    response = query_func(query)
                    
                    # Extract data based on response format
                    if isinstance(response, dict):
                        answer = response.get('answer', '')
                        docs = response.get('retrieved_docs', [])
                        context = response.get('context', '')
                    else:
                        # Assume the response is the answer
                        answer = str(response)
                        docs = []
                        context = ""
                    
                    retrieved_docs_list.append(docs)
                    context_list.append([context] if context else [])
                    generated_texts.append(answer)
                    
                except Exception as e:
                    print(f"Error executing query '{query}': {str(e)}")
                    retrieved_docs_list.append([])
                    context_list.append([])
                    generated_texts.append("")
        
        # Evaluate retrieval if requested
        if evaluate_retrieval and retrieved_docs_list:
            results['retrieval'] = self.evaluate_retrieval(
                queries=queries,
                retrieved_docs_list=retrieved_docs_list,
                relevant_docs_list=relevant_docs
            )
        
        # Evaluate generation if requested
        if evaluate_generation and generated_texts:
            results['generation'] = self.evaluate_generation(
                queries=queries,
                generated_texts=generated_texts,
                retrieved_contexts_list=context_list or retrieved_docs_list,
                reference_texts=expected_answers if expected_answers and expected_answers[0] else None
            )
        
        # Evaluate end-to-end if requested
        if evaluate_end_to_end and generated_texts:
            results['end_to_end'] = self.evaluate_end_to_end(
                outputs=generated_texts,
                expected_outputs=expected_answers
            )
        
        # Measure efficiency if requested
        if measure_system_efficiency and query_func is not None:
            efficiency_results = self.measure_efficiency(
                query_func=query_func,
                queries=queries[:min(10, len(queries))]  # Limit to 10 queries for efficiency
            )
            
            # Include efficiency in end-to-end results
            if 'end_to_end' in results:
                results['end_to_end']['efficiency'] = efficiency_results
            else:
                results['end_to_end'] = {'efficiency': efficiency_results}
        
        # Visualize results if requested
        if visualize:
            self.visualize_results(results=results)
        
        # Calculate overall duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"Evaluation completed in {duration:.2f} seconds.")
        
        # Save combined results
        if self.output_dir:
            output_file = os.path.join(self.output_dir, "evaluation_results.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                # Add metadata to results
                full_results = {
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "num_queries": len(queries),
                        "duration_seconds": duration
                    },
                    "results": results
                }
                json.dump(full_results, f, indent=2)
            print(f"Combined evaluation results saved to {output_file}")
        
        return results
    
    def summary(self, results: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate a summary of evaluation results.
        
        Args:
            results: Dictionary of evaluation results by category
            
        Returns:
            Text summary of the results
        """
        summary = "=== RAG EVALUATION SUMMARY ===\n\n"
        
        # Retrieval results
        if 'retrieval' in results:
            summary += "RETRIEVAL METRICS:\n"
            if 'map' in results['retrieval']:
                summary += f"- Mean Average Precision (MAP): {results['retrieval']['map']:.3f}\n"
            
            if 'precision' in results['retrieval']:
                for k, score in results['retrieval']['precision'].items():
                    summary += f"- Precision{k}: {score:.3f}\n"
            
            if 'recall' in results['retrieval']:
                for k, score in results['retrieval']['recall'].items():
                    summary += f"- Recall{k}: {score:.3f}\n"
            
            if 'f1' in results['retrieval']:
                for k, score in results['retrieval']['f1'].items():
                    summary += f"- F1{k}: {score:.3f}\n"
            
            summary += "\n"
        
        # Generation results
        if 'generation' in results and 'overall_scores' in results['generation']:
            summary += "GENERATION METRICS:\n"
            
            for metric, score in results['generation']['overall_scores'].items():
                summary += f"- {metric.capitalize()}: {score:.3f}\n"
            
            summary += "\n"
        
        # End-to-end results
        if 'end_to_end' in results:
            summary += "END-TO-END METRICS:\n"
            
            if 'task_completion' in results['end_to_end']:
                summary += f"- Task Completion Rate: {results['end_to_end']['task_completion']:.3f}\n"
            
            if 'overall_score' in results['end_to_end']:
                summary += f"- Overall Score: {results['end_to_end']['overall_score']:.3f}\n"
            
            if 'efficiency' in results['end_to_end']:
                eff = results['end_to_end']['efficiency']
                if 'average_latency' in eff:
                    summary += f"- Average Latency: {eff['average_latency']:.3f} seconds\n"
                if 'throughput' in eff:
                    summary += f"- Throughput: {eff['throughput']:.2f} queries/second\n"
            
            summary += "\n"
        
        # Human evaluation results
        if 'human_eval' in results and 'overall_scores' in results['human_eval']:
            summary += "HUMAN EVALUATION METRICS:\n"
            
            for metric, score in results['human_eval']['overall_scores'].items():
                summary += f"- {metric.capitalize()}: {score:.3f}\n"
            
            if 'overall_system_preference' in results['human_eval']:
                pref = results['human_eval']['overall_system_preference']
                if 'winner' in pref:
                    summary += f"- Preferred System: {pref['winner']}\n"
            
            summary += "\n"
        
        return summary 
"""
Cache Dashboard Module

This module provides monitoring capabilities for the RAGNITE caching infrastructure,
including metrics collection, visualization, and performance analytics.
"""

import logging
import time
import json
import os
import threading
from typing import Dict, Any, List, Tuple, Optional, Union, Set, Callable
from datetime import datetime, timedelta
from pathlib import Path
import statistics
from collections import defaultdict, deque

try:
    from .cache_manager import CacheManager
    from .embedding_cache import EmbeddingCache
    from .semantic_cache import SemanticQueryCache
    from .result_cache import ResultCache
    from .prompt_cache import PromptCache
except ImportError:
    from cache_manager import CacheManager
    from embedding_cache import EmbeddingCache
    from semantic_cache import SemanticQueryCache
    from result_cache import ResultCache
    from prompt_cache import PromptCache

# Configure logging
logger = logging.getLogger(__name__)

class CacheMetricsCollector:
    """
    Collector for cache-related metrics across different cache components.
    """
    
    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        embedding_cache: Optional[EmbeddingCache] = None,
        semantic_cache: Optional[SemanticQueryCache] = None,
        result_cache: Optional[ResultCache] = None,
        prompt_cache: Optional[PromptCache] = None,
        collection_interval: int = 60,  # seconds
        history_size: int = 24,  # hours
        metrics_dir: Optional[str] = None
    ):
        """
        Initialize the metrics collector.
        
        Args:
            cache_manager: Optional central cache manager
            embedding_cache: Optional embedding cache
            semantic_cache: Optional semantic cache
            result_cache: Optional result cache
            prompt_cache: Optional prompt cache
            collection_interval: Interval for collecting metrics in seconds
            history_size: Number of hours of metrics to keep in memory
            metrics_dir: Directory to store metrics data
        """
        # Cache components
        self.cache_manager = cache_manager
        self.embedding_cache = embedding_cache
        self.semantic_cache = semantic_cache
        self.result_cache = result_cache
        self.prompt_cache = prompt_cache
        
        # Configuration
        self.collection_interval = collection_interval
        self.history_size = history_size
        self.metrics_dir = metrics_dir
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        
        # Metrics history (deques for fixed-size history)
        max_history_points = (history_size * 3600) // collection_interval
        self.metrics_history = deque(maxlen=max_history_points)
        
        # Current metrics
        self.current_metrics = {}
        
        # Performance tracking
        self.performance_stats = {
            'hit_rates': defaultdict(list),
            'latencies': defaultdict(list),
            'sizes': defaultdict(list),
            'cost_savings': []
        }
        
        # Start collection thread
        self._stop_collection = False
        self._collection_thread = None
        if collection_interval > 0:
            self._start_collection_thread()
    
    def _start_collection_thread(self) -> None:
        """Start the metrics collection thread."""
        def collection_worker():
            while not self._stop_collection:
                try:
                    self.collect_metrics()
                    
                    # Calculate and update performance stats
                    self._update_performance_stats()
                    
                    # Persist metrics if enabled
                    if self.metrics_dir:
                        self._persist_current_metrics()
                except Exception as e:
                    logger.error(f"Error in metrics collection: {str(e)}")
                
                time.sleep(self.collection_interval)
        
        self._collection_thread = threading.Thread(target=collection_worker, daemon=True)
        self._collection_thread.start()
    
    def stop_collection(self) -> None:
        """Stop the metrics collection thread."""
        self._stop_collection = True
        if self._collection_thread:
            self._collection_thread.join(timeout=2.0)
    
    def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect metrics from all cache components.
        
        Returns:
            Dictionary with combined metrics
        """
        metrics = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat()
        }
        
        # Collect from cache manager
        if self.cache_manager:
            metrics['cache_manager'] = self.cache_manager.get_metrics()
        
        # Collect from embedding cache
        if self.embedding_cache:
            metrics['embedding_cache'] = self.embedding_cache.get_metrics()
        
        # Collect from semantic cache
        if self.semantic_cache:
            metrics['semantic_cache'] = self.semantic_cache.get_metrics()
        
        # Collect from result cache
        if self.result_cache:
            metrics['result_cache'] = self.result_cache.get_metrics()
        
        # Collect from prompt cache
        if self.prompt_cache:
            metrics['prompt_cache'] = self.prompt_cache.get_metrics()
        
        # Add to history
        self.metrics_history.append(metrics)
        
        # Update current metrics
        self.current_metrics = metrics
        
        return metrics
    
    def _update_performance_stats(self) -> None:
        """Update performance statistics based on current metrics."""
        metrics = self.current_metrics
        
        # Extract hit rates
        if 'cache_manager' in metrics:
            cm_metrics = metrics['cache_manager']
            
            for cache_type in ['embedding_cache', 'semantic_cache', 'result_cache', 'prompt_cache']:
                hit_rate_key = f"{cache_type}_hit_rate"
                if hit_rate_key in cm_metrics:
                    self.performance_stats['hit_rates'][cache_type].append(cm_metrics[hit_rate_key])
        
        # Extract from individual caches
        for cache_type in ['embedding_cache', 'semantic_cache', 'result_cache', 'prompt_cache']:
            if cache_type in metrics:
                cache_metrics = metrics[cache_type]
                
                # Hit rates
                if 'hit_rate' in cache_metrics:
                    self.performance_stats['hit_rates'][cache_type].append(cache_metrics['hit_rate'])
                elif 'overall_hit_rate' in cache_metrics:
                    self.performance_stats['hit_rates'][cache_type].append(cache_metrics['overall_hit_rate'])
                
                # Sizes
                if 'size' in cache_metrics:
                    self.performance_stats['sizes'][cache_type].append(cache_metrics['size'])
                
                # Cost savings (embedding cache specific)
                if cache_type == 'embedding_cache' and 'estimated_cost_saved' in cache_metrics:
                    self.performance_stats['cost_savings'].append(cache_metrics['estimated_cost_saved'])
        
        # Limit the size of performance stats arrays
        max_stats_points = (self.history_size * 3600) // self.collection_interval
        for category, stat_dict in self.performance_stats.items():
            if category == 'cost_savings':
                self.performance_stats[category] = self.performance_stats[category][-max_stats_points:]
            else:
                for key in stat_dict:
                    stat_dict[key] = stat_dict[key][-max_stats_points:]
    
    def _persist_current_metrics(self) -> None:
        """Persist the current metrics to disk."""
        if not self.metrics_dir:
            return
        
        timestamp = int(time.time())
        date_str = datetime.now().strftime('%Y%m%d')
        
        # Create daily directory
        daily_dir = os.path.join(self.metrics_dir, date_str)
        os.makedirs(daily_dir, exist_ok=True)
        
        # Save metrics
        metrics_file = os.path.join(daily_dir, f"metrics_{timestamp}.json")
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.current_metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Error persisting metrics: {str(e)}")
        
        # Save summary to latest file
        latest_file = os.path.join(self.metrics_dir, "latest_metrics.json")
        try:
            with open(latest_file, 'w') as f:
                json.dump(self.current_metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving latest metrics: {str(e)}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get the most recent metrics.
        
        Returns:
            Dictionary with current metrics
        """
        return self.current_metrics
    
    def get_metrics_history(self, hours: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get historical metrics.
        
        Args:
            hours: Number of hours of history to return (None for all available)
            
        Returns:
            List of metrics dictionaries
        """
        if hours is None or hours >= self.history_size:
            return list(self.metrics_history)
        
        # Calculate how many data points to return
        points_per_hour = 3600 // self.collection_interval
        num_points = hours * points_per_hour
        
        return list(self.metrics_history)[-num_points:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance statistics.
        
        Returns:
            Dictionary with performance summary
        """
        summary = {}
        
        # Calculate hit rate statistics
        hit_rates = self.performance_stats['hit_rates']
        hit_rate_summary = {}
        
        for cache_type, rates in hit_rates.items():
            if rates:
                hit_rate_summary[cache_type] = {
                    'current': rates[-1],
                    'average': statistics.mean(rates),
                    'min': min(rates),
                    'max': max(rates),
                    'trend': 'improving' if len(rates) > 1 and rates[-1] > rates[0] else 'declining' if len(rates) > 1 and rates[-1] < rates[0] else 'stable'
                }
        
        summary['hit_rates'] = hit_rate_summary
        
        # Calculate size statistics
        sizes = self.performance_stats['sizes']
        size_summary = {}
        
        for cache_type, size_data in sizes.items():
            if size_data:
                size_summary[cache_type] = {
                    'current': size_data[-1],
                    'average': statistics.mean(size_data),
                    'max': max(size_data),
                    'growth_rate': (size_data[-1] - size_data[0]) / len(size_data) if len(size_data) > 1 else 0
                }
        
        summary['sizes'] = size_summary
        
        # Cost savings
        cost_savings = self.performance_stats['cost_savings']
        if cost_savings:
            summary['cost_savings'] = {
                'total': sum(cost_savings),
                'average_per_interval': statistics.mean(cost_savings),
                'projected_monthly': statistics.mean(cost_savings) * (3600 * 24 * 30 / self.collection_interval)
            }
        
        return summary
    
    def get_analytics_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive analytics report.
        
        Returns:
            Dictionary with analytics report
        """
        # Get the most recent metrics and performance summary
        current_metrics = self.get_current_metrics()
        performance_summary = self.get_performance_summary()
        
        # Calculate overall cache efficiency
        hit_rates = performance_summary.get('hit_rates', {})
        overall_hit_rate = 0.0
        hit_rate_count = 0
        
        for cache_type, stats in hit_rates.items():
            overall_hit_rate += stats.get('current', 0.0)
            hit_rate_count += 1
        
        overall_hit_rate = overall_hit_rate / hit_rate_count if hit_rate_count > 0 else 0.0
        
        # Generate recommendations
        recommendations = []
        
        # Low hit rate recommendations
        for cache_type, stats in hit_rates.items():
            current_rate = stats.get('current', 0.0)
            if current_rate < 0.5:  # Less than 50% hit rate
                if cache_type == 'embedding_cache':
                    recommendations.append(
                        f"Consider increasing {cache_type} size as hit rate is only {current_rate:.1%}"
                    )
                elif cache_type == 'semantic_cache':
                    recommendations.append(
                        f"Consider adjusting semantic similarity threshold or cache size (current hit rate: {current_rate:.1%})"
                    )
                else:
                    recommendations.append(
                        f"Review {cache_type} configuration as hit rate is only {current_rate:.1%}"
                    )
        
        # Size-related recommendations
        sizes = performance_summary.get('sizes', {})
        for cache_type, stats in sizes.items():
            growth_rate = stats.get('growth_rate', 0.0)
            if growth_rate > 0.1:  # Growing quickly
                recommendations.append(
                    f"{cache_type} is growing rapidly, consider implementing cache eviction or increasing size"
                )
        
        # Build the report
        report = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'overall_cache_efficiency': overall_hit_rate,
            'performance_summary': performance_summary,
            'current_metrics': current_metrics,
            'recommendations': recommendations,
            'collection_info': {
                'interval_seconds': self.collection_interval,
                'history_hours': self.history_size,
                'data_points': len(self.metrics_history)
            }
        }
        
        return report


class CacheDashboard:
    """
    Dashboard for monitoring and analyzing cache performance.
    """
    
    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        embedding_cache: Optional[EmbeddingCache] = None,
        semantic_cache: Optional[SemanticQueryCache] = None,
        result_cache: Optional[ResultCache] = None,
        prompt_cache: Optional[PromptCache] = None,
        metrics_collector: Optional[CacheMetricsCollector] = None,
        metrics_dir: Optional[str] = None,
        collection_interval: int = 60,  # seconds
        history_size: int = 24,  # hours
        enable_web_ui: bool = False,
        web_port: int = 8050
    ):
        """
        Initialize the cache dashboard.
        
        Args:
            cache_manager: Optional central cache manager
            embedding_cache: Optional embedding cache
            semantic_cache: Optional semantic cache
            result_cache: Optional result cache
            prompt_cache: Optional prompt cache
            metrics_collector: Optional existing metrics collector
            metrics_dir: Directory to store metrics data
            collection_interval: Interval for collecting metrics in seconds
            history_size: Number of hours of metrics to keep in memory
            enable_web_ui: Whether to enable the web UI
            web_port: Port for the web UI server
        """
        # Use provided collector or create new one
        if metrics_collector:
            self.metrics_collector = metrics_collector
        else:
            self.metrics_collector = CacheMetricsCollector(
                cache_manager=cache_manager,
                embedding_cache=embedding_cache,
                semantic_cache=semantic_cache,
                result_cache=result_cache,
                prompt_cache=prompt_cache,
                collection_interval=collection_interval,
                history_size=history_size,
                metrics_dir=metrics_dir
            )
        
        # Store references to cache components
        self.cache_manager = cache_manager
        self.embedding_cache = embedding_cache
        self.semantic_cache = semantic_cache
        self.result_cache = result_cache
        self.prompt_cache = prompt_cache
        
        # Web UI configuration
        self.enable_web_ui = enable_web_ui
        self.web_port = web_port
        self.web_app = None
        
        # Initialize web UI if enabled
        if enable_web_ui:
            self._setup_web_ui()
    
    def _setup_web_ui(self):
        """Set up the web-based dashboard interface."""
        try:
            # Defer imports to avoid dependencies if web UI not used
            import dash
            from dash import dcc, html
            import plotly.express as px
            import plotly.graph_objects as go
            import pandas as pd
            
            # Create Dash app
            app = dash.Dash(__name__, title="RAGNITE Cache Dashboard")
            
            # Define layout
            app.layout = html.Div([
                html.H1("RAGNITE Cache Dashboard"),
                html.Div([
                    html.H2("Cache Performance"),
                    dcc.Graph(id='hit-rate-graph'),
                    dcc.Graph(id='cache-size-graph'),
                    html.Div(id='cost-savings-info', className='info-box')
                ]),
                html.Div([
                    html.H2("Cache Health"),
                    html.Div(id='health-metrics', className='metrics-container')
                ]),
                html.Div([
                    html.H2("Recommendations"),
                    html.Ul(id='recommendations', className='recommendations-list')
                ]),
                dcc.Interval(
                    id='interval-component',
                    interval=5*1000,  # in milliseconds (5 seconds)
                    n_intervals=0
                )
            ])
            
            # Define callbacks to update components
            @app.callback(
                [
                    dash.Output('hit-rate-graph', 'figure'),
                    dash.Output('cache-size-graph', 'figure'),
                    dash.Output('cost-savings-info', 'children'),
                    dash.Output('health-metrics', 'children'),
                    dash.Output('recommendations', 'children')
                ],
                [dash.Input('interval-component', 'n_intervals')]
            )
            def update_dashboard(n):
                # Get analytics report
                report = self.metrics_collector.get_analytics_report()
                history = self.metrics_collector.get_metrics_history(hours=6)  # Last 6 hours
                
                # Create dataframes for plotting
                hit_rate_data = []
                size_data = []
                
                for entry in history:
                    timestamp = entry.get('timestamp', 0)
                    dt = datetime.fromtimestamp(timestamp)
                    
                    # Process each cache type
                    for cache_type in ['embedding_cache', 'semantic_cache', 'result_cache', 'prompt_cache']:
                        if cache_type in entry:
                            # Hit rate
                            hit_rate = entry[cache_type].get('hit_rate', entry[cache_type].get('overall_hit_rate', 0))
                            hit_rate_data.append({
                                'timestamp': dt,
                                'cache_type': cache_type,
                                'hit_rate': hit_rate
                            })
                            
                            # Size
                            size = entry[cache_type].get('size', 0)
                            size_data.append({
                                'timestamp': dt,
                                'cache_type': cache_type,
                                'size': size
                            })
                
                # Create figures
                hit_rate_df = pd.DataFrame(hit_rate_data) if hit_rate_data else pd.DataFrame(columns=['timestamp', 'cache_type', 'hit_rate'])
                size_df = pd.DataFrame(size_data) if size_data else pd.DataFrame(columns=['timestamp', 'cache_type', 'size'])
                
                hit_rate_fig = px.line(
                    hit_rate_df,
                    x='timestamp',
                    y='hit_rate',
                    color='cache_type',
                    title='Cache Hit Rates Over Time',
                    labels={'hit_rate': 'Hit Rate', 'timestamp': 'Time'},
                    range_y=[0, 1]
                )
                
                size_fig = px.line(
                    size_df,
                    x='timestamp',
                    y='size',
                    color='cache_type',
                    title='Cache Size Over Time',
                    labels={'size': 'Size (items)', 'timestamp': 'Time'}
                )
                
                # Cost savings info
                cost_savings = report.get('performance_summary', {}).get('cost_savings', {})
                cost_savings_html = html.Div([
                    html.H3("Estimated Cost Savings"),
                    html.P(f"Total: ${cost_savings.get('total', 0):.4f}"),
                    html.P(f"Projected Monthly: ${cost_savings.get('projected_monthly', 0):.2f}")
                ])
                
                # Health metrics
                health_metrics_html = html.Div([
                    html.Div([
                        html.H3("Overall Cache Efficiency"),
                        html.Div(f"{report.get('overall_cache_efficiency', 0):.1%}", className='metric-value')
                    ], className='metric-box'),
                    html.Div([
                        html.H3("Total Cache Entries"),
                        html.Div(str(sum(s.get('current', 0) for s in report.get('performance_summary', {}).get('sizes', {}).values())), className='metric-value')
                    ], className='metric-box')
                ])
                
                # Recommendations
                recommendations = report.get('recommendations', [])
                recommendations_html = [html.Li(rec) for rec in recommendations]
                if not recommendations_html:
                    recommendations_html = [html.Li("No recommendations at this time.")]
                
                return hit_rate_fig, size_fig, cost_savings_html, health_metrics_html, recommendations_html
            
            # Store the app
            self.web_app = app
            
        except ImportError as e:
            logger.error(f"Could not set up web UI: {str(e)}")
            logger.error("Make sure dash, plotly, and pandas are installed")
            self.enable_web_ui = False
    
    def start_web_ui(self) -> None:
        """Start the web UI server."""
        if not self.enable_web_ui or self.web_app is None:
            logger.warning("Web UI is not enabled or could not be initialized")
            return
        
        # Start the server in a separate thread
        def run_server():
            self.web_app.run_server(debug=False, port=self.web_port)
        
        threading.Thread(target=run_server, daemon=True).start()
        logger.info(f"Cache dashboard web UI started on port {self.web_port}")
    
    def print_summary(self) -> None:
        """Print a summary of cache performance to the console."""
        report = self.metrics_collector.get_analytics_report()
        
        print("\n===== RAGNITE CACHE DASHBOARD =====")
        print(f"Report Time: {report.get('datetime')}")
        print(f"Overall Cache Efficiency: {report.get('overall_cache_efficiency', 0):.1%}")
        print("\n--- Hit Rates ---")
        
        hit_rates = report.get('performance_summary', {}).get('hit_rates', {})
        for cache_type, stats in hit_rates.items():
            print(f"{cache_type}: {stats.get('current', 0):.1%} (trend: {stats.get('trend', 'unknown')})")
        
        print("\n--- Cache Sizes ---")
        sizes = report.get('performance_summary', {}).get('sizes', {})
        for cache_type, stats in sizes.items():
            print(f"{cache_type}: {stats.get('current', 0)} items")
        
        print("\n--- Recommendations ---")
        recommendations = report.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        else:
            print("No recommendations at this time.")
        
        print("\n===================================\n")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get all dashboard data for external consumption.
        
        Returns:
            Dictionary with all dashboard data
        """
        return {
            'current_metrics': self.metrics_collector.get_current_metrics(),
            'performance_summary': self.metrics_collector.get_performance_summary(),
            'analytics_report': self.metrics_collector.get_analytics_report(),
            'metrics_history': self.metrics_collector.get_metrics_history()
        }
    
    def cleanup(self) -> None:
        """Clean up resources used by the dashboard."""
        self.metrics_collector.stop_collection() 
"""
Dashboard implementation for Fantasy League optimization problem.

This module provides Dash-based dashboards for visualizing performance metrics
across different optimization algorithms applied to the Fantasy League problem.
"""

import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Define color schemes for consistency across dashboards
ALGORITHM_COLORS = {
    "HillClimbing": "#1f77b4",  # Blue
    "SimulatedAnnealing": "#ff7f0e",  # Orange
    "GeneticAlgorithm": "#2ca02c",  # Green
    "IslandGA": "#d62728",  # Red
}

CONFIG_COLORS = {
    "HillClimbing_Std": "#1f77b4",
    "HillClimbing_Intensive": "#aec7e8",
    "HillClimbing_RandomRestart_Test": "#7fc97f",
    "SimulatedAnnealing_Std": "#ff7f0e",
    "SimulatedAnnealing_Enhanced": "#ffbb78",
    "GA_Tournament_OnePoint": "#2ca02c",
    "GA_Ranking_Uniform": "#98df8a",
    "GA_TwoPointCrossover_Test": "#d62728",
    "GA_Hybrid_Optimized": "#ff9896",
    "GA_Island_Model_Test": "#9467bd",
}

# Helper functions for dashboard components
def load_metrics_data(metrics_dir: str) -> pd.DataFrame:
    """
    Load metrics data from JSON files in the specified directory.
    
    Args:
        metrics_dir: Directory containing metrics JSON files
        
    Returns:
        DataFrame containing metrics data
    """
    all_metrics = []
    
    for filename in os.listdir(metrics_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(metrics_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    metrics_data = json.load(f)
                
                # Extract key metrics for the DataFrame
                metrics_row = {
                    'algorithm_name': metrics_data.get('algorithm_name', 'Unknown'),
                    'config_name': metrics_data.get('config_name', 'Unknown'),
                    'run_id': metrics_data.get('run_id', 'Unknown'),
                    'best_fitness': metrics_data.get('metrics', {}).get('best_fitness', float('inf')),
                    'runtime_seconds': metrics_data.get('metrics', {}).get('runtime_seconds', None),
                    'iterations': metrics_data.get('metrics', {}).get('iterations', None),
                    'function_evaluations': metrics_data.get('metrics', {}).get('function_evaluations', None),
                    'convergence_rate': metrics_data.get('metrics', {}).get('convergence_rate', None),
                    'solution_stability': metrics_data.get('metrics', {}).get('solution_stability', None),
                    'raw_data': metrics_data.get('raw_data', {}),
                    'full_metrics': metrics_data.get('metrics', {})
                }
                
                # Add algorithm-specific metrics
                if metrics_data.get('algorithm_name') == 'HillClimbing':
                    metrics_row.update({
                        'neighbors_generated': metrics_data.get('metrics', {}).get('neighbors_generated', None),
                        'neighbors_evaluated': metrics_data.get('metrics', {}).get('neighbors_evaluated', None),
                        'local_optima_count': metrics_data.get('metrics', {}).get('local_optima_count', None),
                        'plateau_length': metrics_data.get('metrics', {}).get('plateau_length', None),
                        'improvement_rate': metrics_data.get('metrics', {}).get('improvement_rate', None),
                    })
                elif metrics_data.get('algorithm_name') == 'SimulatedAnnealing':
                    metrics_row.update({
                        'acceptance_rate': metrics_data.get('metrics', {}).get('acceptance_rate', None),
                        'worse_solutions_accepted': metrics_data.get('metrics', {}).get('worse_solutions_accepted', None),
                        'cooling_efficiency': metrics_data.get('metrics', {}).get('cooling_efficiency', None),
                        'temperature_impact': metrics_data.get('metrics', {}).get('temperature_impact', None),
                    })
                elif metrics_data.get('algorithm_name') in ['GeneticAlgorithm', 'IslandGA']:
                    metrics_row.update({
                        'generations': metrics_data.get('metrics', {}).get('generations', None),
                        'population_size': metrics_data.get('metrics', {}).get('population_size', None),
                        'crossover_success_rate': metrics_data.get('metrics', {}).get('crossover_success_rate', None),
                        'mutation_impact': metrics_data.get('metrics', {}).get('mutation_impact', None),
                        'selection_pressure': metrics_data.get('metrics', {}).get('selection_pressure', None),
                    })
                
                # Add Island GA specific metrics
                if metrics_data.get('algorithm_name') == 'IslandGA':
                    metrics_row.update({
                        'num_islands': metrics_data.get('metrics', {}).get('num_islands', None),
                        'migration_events': metrics_data.get('metrics', {}).get('migration_events', None),
                        'migration_impact': metrics_data.get('metrics', {}).get('migration_impact', None),
                        'migration_success_rate': metrics_data.get('metrics', {}).get('migration_success_rate', None),
                        'topology_efficiency': metrics_data.get('metrics', {}).get('topology_efficiency', None),
                    })
                
                all_metrics.append(metrics_row)
            except Exception as e:
                logger.error(f"Error loading metrics from {file_path}: {e}")
    
    if not all_metrics:
        return pd.DataFrame()
    
    return pd.DataFrame(all_metrics)


def create_global_comparison_dashboard(metrics_df: pd.DataFrame):
    """
    Create a dashboard for global comparison of all algorithms.
    
    Args:
        metrics_df: DataFrame containing metrics data
        
    Returns:
        Dash app layout for global comparison
    """
    if metrics_df.empty:
        return html.Div("No metrics data available for comparison.")
    
    # Create figures for key metrics comparison
    best_fitness_fig = px.box(
        metrics_df, 
        x='config_name', 
        y='best_fitness',
        color='algorithm_name',
        title='Best Fitness by Algorithm Configuration',
        labels={'best_fitness': 'Best Fitness (lower is better)', 'config_name': 'Configuration'},
        color_discrete_map=ALGORITHM_COLORS
    )
    
    runtime_fig = px.box(
        metrics_df, 
        x='config_name', 
        y='runtime_seconds',
        color='algorithm_name',
        title='Runtime by Algorithm Configuration',
        labels={'runtime_seconds': 'Runtime (seconds)', 'config_name': 'Configuration'},
        color_discrete_map=ALGORITHM_COLORS
    )
    
    evaluations_fig = px.box(
        metrics_df, 
        x='config_name', 
        y='function_evaluations',
        color='algorithm_name',
        title='Function Evaluations by Algorithm Configuration',
        labels={'function_evaluations': 'Function Evaluations', 'config_name': 'Configuration'},
        color_discrete_map=ALGORITHM_COLORS
    )
    
    # Create radar chart for algorithm comparison
    # Aggregate metrics by algorithm type
    agg_metrics = metrics_df.groupby('algorithm_name').agg({
        'best_fitness': 'mean',
        'runtime_seconds': 'mean',
        'function_evaluations': 'mean',
        'convergence_rate': 'mean',
        'solution_stability': 'mean'
    }).reset_index()
    
    # Normalize metrics for radar chart (0-1 scale where 1 is best)
    for col in agg_metrics.columns:
        if col != 'algorithm_name' and not agg_metrics[col].isna().all():
            if col == 'best_fitness' or col == 'runtime_seconds' or col == 'function_evaluations' or col == 'solution_stability':
                # Lower is better for these metrics
                min_val = agg_metrics[col].min()
                max_val = agg_metrics[col].max()
                if max_val > min_val:
                    agg_metrics[f'{col}_norm'] = 1 - ((agg_metrics[col] - min_val) / (max_val - min_val))
                else:
                    agg_metrics[f'{col}_norm'] = 1.0
            else:
                # Higher is better for these metrics
                min_val = agg_metrics[col].min()
                max_val = agg_metrics[col].max()
                if max_val > min_val:
                    agg_metrics[f'{col}_norm'] = (agg_metrics[col] - min_val) / (max_val - min_val)
                else:
                    agg_metrics[f'{col}_norm'] = 1.0
    
    # Create radar chart
    radar_fig = go.Figure()
    
    categories = ['Solution Quality', 'Computational Efficiency', 'Convergence Speed', 'Solution Stability', 'Overall Performance']
    
    for _, row in agg_metrics.iterrows():
        algorithm = row['algorithm_name']
        
        # Calculate overall performance as average of other normalized metrics
        overall = np.mean([
            row.get('best_fitness_norm', 0),
            row.get('runtime_seconds_norm', 0),
            row.get('function_evaluations_norm', 0),
            row.get('convergence_rate_norm', 0),
            row.get('solution_stability_norm', 0)
        ])
        
        radar_fig.add_trace(go.Scatterpolar(
            r=[
                row.get('best_fitness_norm', 0),
                row.get('runtime_seconds_norm', 0),
                row.get('function_evaluations_norm', 0),
                row.get('solution_stability_norm', 0),
                overall
            ],
            theta=categories,
            fill='toself',
            name=algorithm,
            line_color=ALGORITHM_COLORS.get(algorithm, '#000000')
        ))
    
    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title='Algorithm Performance Comparison',
        showlegend=True
    )
    
    # Create convergence plot
    convergence_fig = go.Figure()
    
    # Get one representative run for each configuration
    for config in metrics_df['config_name'].unique():
        config_data = metrics_df[metrics_df['config_name'] == config]
        if not config_data.empty:
            # Get the run with median best fitness
            median_idx = config_data['best_fitness'].argmin()
            run_data = config_data.iloc[median_idx]
            
            # Extract convergence data
            if 'best_fitness_history' in run_data.get('full_metrics', {}):
                y_data = run_data['full_metrics']['best_fitness_history']
                x_data = list(range(len(y_data)))
                
                convergence_fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines',
                    name=config,
                    line=dict(color=CONFIG_COLORS.get(config, '#000000'))
                ))
    
    convergence_fig.update_layout(
        title='Convergence Comparison',
        xaxis_title='Iterations',
        yaxis_title='Best Fitness',
        legend_title='Configuration'
    )
    
    # Create dashboard layout
    layout = html.Div([
        html.H1("Fantasy League Optimization - Global Algorithm Comparison"),
        
        html.Div([
            html.Div([
                dcc.Graph(figure=radar_fig)
            ], className='six columns'),
            
            html.Div([
                dcc.Graph(figure=convergence_fig)
            ], className='six columns')
        ], className='row'),
        
        html.Div([
            html.Div([
                dcc.Graph(figure=best_fitness_fig)
            ], className='four columns'),
            
            html.Div([
                dcc.Graph(figure=runtime_fig)
            ], className='four columns'),
            
            html.Div([
                dcc.Graph(figure=evaluations_fig)
            ], className='four columns')
        ], className='row'),
        
        html.Div([
            html.H3("Statistical Summary"),
            
            html.Div(id='statistical-summary', children=[
                # Will be populated by callback
                html.P("Select metrics to compare in the dropdowns below."),
                
                html.Div([
                    html.Label("Select Primary Metric:"),
                    dcc.Dropdown(
                        id='primary-metric-dropdown',
                        options=[
                            {'label': 'Best Fitness', 'value': 'best_fitness'},
                            {'label': 'Runtime (seconds)', 'value': 'runtime_seconds'},
                            {'label': 'Function Evaluations', 'value': 'function_evaluations'},
                            {'label': 'Convergence Rate', 'value': 'convergence_rate'},
                            {'label': 'Solution Stability', 'value': 'solution_stability'}
                        ],
                        value='best_fitness'
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'margin-right': '5%'}),
                
                html.Div([
                    html.Label("Select Secondary Metric:"),
                    dcc.Dropdown(
                        id='secondary-metric-dropdown',
                        options=[
                            {'label': 'Best Fitness', 'value': 'best_fitness'},
                            {'label': 'Runtime (seconds)', 'value': 'runtime_seconds'},
                            {'label': 'Function Evaluations', 'value': 'function_evaluations'},
                            {'label': 'Convergence Rate', 'value': 'convergence_rate'},
                            {'label': 'Solution Stability', 'value': 'solution_stability'}
                        ],
                        value='runtime_seconds'
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div(id='metrics-correlation-output')
            ])
        ], className='row')
    ])
    
    return layout


def create_hill_climbing_dashboard(metrics_df: pd.DataFrame):
    """
    Create a dashboard for Hill Climbing algorithm analysis.
    
    Args:
        metrics_df: DataFrame containing metrics data
        
    Returns:
        Dash app layout for Hill Climbing analysis
    """
    # Filter for Hill Climbing data
    hc_df = metrics_df[metrics_df['algorithm_name'] == 'HillClimbing']
    
    if hc_df.empty:
        return html.Div("No Hill Climbing metrics data available.")
    
    # Create figures for Hill Climbing specific metrics
    neighbors_fig = px.bar(
        hc_df,
        x='config_name',
        y=['neighbors_generated', 'neighbors_evaluated'],
        barmode='group',
        title='Neighbor Generation and Evaluation',
        labels={
            'value': 'Count',
            'config_name': 'Configuration',
            'variable': 'Metric'
        }
    )
    
    local_optima_fig = px.bar(
        hc_df,
        x='config_name',
        y='local_optima_count',
        title='Local Optima Encountered',
        labels={
            'local_optima_count': 'Count',
            'config_name': 'Configuration'
        },
        color='config_name',
        color_discrete_map=CONFIG_COLORS
    )
    
    # Create plateau analysis figure
    plateau_fig = px.scatter(
        hc_df,
        x='plateau_length',
        y='best_fitness',
        color='config_name',
        size='iterations',
        title='Plateau Analysis',
        labels={
            'plateau_length': 'Plateau Length',
            'best_fitness': 'Best Fitness',
            'iterations': 'Total Iterations'
        },
        color_discrete_map=CONFIG_COLORS
    )
    
    # Create improvement rate figure
    improvement_fig = px.box(
        hc_df,
        x='config_name',
        y='improvement_rate',
        title='Improvement Rate (improvements per neighbor evaluated)',
        labels={
            'improvement_rate': 'Improvement Rate',
            'config_name': 'Configuration'
        },
        color='config_name',
        color_discrete_map=CONFIG_COLORS
    )
    
    # Create convergence plot for each configuration
    convergence_fig = go.Figure()
    
    for config in hc_df['config_name'].unique():
        config_data = hc_df[hc_df['config_name'] == config]
        if not config_data.empty:
            # Get the run with median best fitness
            median_idx = config_data['best_fitness'].argmin()
            run_data = config_data.iloc[median_idx]
            
            # Extract convergence data
            if 'best_fitness_history' in run_data.get('full_metrics', {}):
                y_data = run_data['full_metrics']['best_fitness_history']
                x_data = list(range(len(y_data)))
                
                convergence_fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines',
                    name=config,
                    line=dict(color=CONFIG_COLORS.get(config, '#000000'))
                ))
    
    convergence_fig.update_layout(
        title='Convergence Comparison',
        xaxis_title='Iterations',
        yaxis_title='Best Fitness',
        legend_title='Configuration'
    )
    
    # Create dashboard layout
    layout = html.Div([
        html.H1("Fantasy League Optimization - Hill Climbing Analysis"),
        
        html.Div([
            html.Div([
                dcc.Graph(figure=convergence_fig)
            ], className='six columns'),
            
            html.Div([
                dcc.Graph(figure=plateau_fig)
            ], className='six columns')
        ], className='row'),
        
        html.Div([
            html.Div([
                dcc.Graph(figure=neighbors_fig)
            ], className='six columns'),
            
            html.Div([
                dcc.Graph(figure=local_optima_fig)
            ], className='six columns')
        ], className='row'),
        
        html.Div([
            html.Div([
                dcc.Graph(figure=improvement_fig)
            ], className='twelve columns')
        ], className='row'),
        
        html.Div([
            html.H3("Hill Climbing Configuration Comparison"),
            
            html.Div(id='hc-config-comparison', children=[
                html.P("Select configurations to compare:"),
                
                dcc.Checklist(
                    id='hc-config-checklist',
                    options=[{'label': config, 'value': config} for config in hc_df['config_name'].unique()],
                    value=list(hc_df['config_name'].unique())[:2] if len(hc_df['config_name'].unique()) >= 2 else list(hc_df['config_name'].unique()),
                    inline=True
                ),
                
                html.Div(id='hc-config-comparison-output')
            ])
        ], className='row')
    ])
    
    return layout


def create_simulated_annealing_dashboard(metrics_df: pd.DataFrame):
    """
    Create a dashboard for Simulated Annealing algorithm analysis.
    
    Args:
        metrics_df: DataFrame containing metrics data
        
    Returns:
        Dash app layout for Simulated Annealing analysis
    """
    # Filter for Simulated Annealing data
    sa_df = metrics_df[metrics_df['algorithm_name'] == 'SimulatedAnnealing']
    
    if sa_df.empty:
        return html.Div("No Simulated Annealing metrics data available.")
    
    # Create figures for Simulated Annealing specific metrics
    acceptance_fig = px.box(
        sa_df,
        x='config_name',
        y='acceptance_rate',
        title='Solution Acceptance Rate',
        labels={
            'acceptance_rate': 'Acceptance Rate',
            'config_name': 'Configuration'
        },
        color='config_name',
        color_discrete_map=CONFIG_COLORS
    )
    
    worse_solutions_fig = px.bar(
        sa_df,
        x='config_name',
        y='worse_solutions_accepted',
        title='Worse Solutions Accepted',
        labels={
            'worse_solutions_accepted': 'Count',
            'config_name': 'Configuration'
        },
        color='config_name',
        color_discrete_map=CONFIG_COLORS
    )
    
    # Create temperature impact figure
    temp_impact_fig = px.scatter(
        sa_df,
        x='temperature_impact',
        y='cooling_efficiency',
        color='config_name',
        size='function_evaluations',
        title='Temperature Impact vs Cooling Efficiency',
        labels={
            'temperature_impact': 'Temperature Impact',
            'cooling_efficiency': 'Cooling Efficiency',
            'function_evaluations': 'Function Evaluations'
        },
        color_discrete_map=CONFIG_COLORS
    )
    
    # Create convergence plot for each configuration
    convergence_fig = go.Figure()
    
    for config in sa_df['config_name'].unique():
        config_data = sa_df[sa_df['config_name'] == config]
        if not config_data.empty:
            # Get the run with median best fitness
            median_idx = config_data['best_fitness'].argmin()
            run_data = config_data.iloc[median_idx]
            
            # Extract convergence data
            if 'best_fitness_history' in run_data.get('full_metrics', {}):
                y_data = run_data['full_metrics']['best_fitness_history']
                x_data = list(range(len(y_data)))
                
                convergence_fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines',
                    name=config,
                    line=dict(color=CONFIG_COLORS.get(config, '#000000'))
                ))
    
    convergence_fig.update_layout(
        title='Convergence Comparison',
        xaxis_title='Iterations',
        yaxis_title='Best Fitness',
        legend_title='Configuration'
    )
    
    # Create temperature vs acceptance rate figure
    # For this, we need to extract temperature history and acceptance rate history
    # from a representative run
    temp_acceptance_fig = go.Figure()
    
    for config in sa_df['config_name'].unique():
        config_data = sa_df[sa_df['config_name'] == config]
        if not config_data.empty:
            # Get the run with median best fitness
            median_idx = config_data['best_fitness'].argmin()
            run_data = config_data.iloc[median_idx]
            
            # Extract temperature history
            if 'temperature_history' in run_data.get('full_metrics', {}):
                temp_history = run_data['full_metrics']['temperature_history']
                
                # Create bins for temperature ranges
                temp_ranges = np.linspace(min(temp_history), max(temp_history), 10)
                
                # Calculate acceptance rate for each temperature bin
                # This is a simplified version - in a real implementation, you would
                # need to extract the actual acceptance data per temperature
                acceptance_by_temp = []
                for i in range(len(temp_ranges) - 1):
                    # Placeholder - in real implementation, calculate actual acceptance rate
                    acceptance_by_temp.append(np.random.uniform(0.1, 1.0))
                
                temp_midpoints = [(temp_ranges[i] + temp_ranges[i+1])/2 for i in range(len(temp_ranges)-1)]
                
                temp_acceptance_fig.add_trace(go.Scatter(
                    x=temp_midpoints,
                    y=acceptance_by_temp,
                    mode='lines+markers',
                    name=config,
                    line=dict(color=CONFIG_COLORS.get(config, '#000000'))
                ))
    
    temp_acceptance_fig.update_layout(
        title='Temperature vs Acceptance Rate',
        xaxis_title='Temperature',
        yaxis_title='Acceptance Rate',
        legend_title='Configuration'
    )
    
    # Create dashboard layout
    layout = html.Div([
        html.H1("Fantasy League Optimization - Simulated Annealing Analysis"),
        
        html.Div([
            html.Div([
                dcc.Graph(figure=convergence_fig)
            ], className='six columns'),
            
            html.Div([
                dcc.Graph(figure=temp_acceptance_fig)
            ], className='six columns')
        ], className='row'),
        
        html.Div([
            html.Div([
                dcc.Graph(figure=acceptance_fig)
            ], className='six columns'),
            
            html.Div([
                dcc.Graph(figure=worse_solutions_fig)
            ], className='six columns')
        ], className='row'),
        
        html.Div([
            html.Div([
                dcc.Graph(figure=temp_impact_fig)
            ], className='twelve columns')
        ], className='row'),
        
        html.Div([
            html.H3("Simulated Annealing Parameter Analysis"),
            
            html.Div(id='sa-parameter-analysis', children=[
                html.P("Select a configuration to analyze:"),
                
                dcc.Dropdown(
                    id='sa-config-dropdown',
                    options=[{'label': config, 'value': config} for config in sa_df['config_name'].unique()],
                    value=sa_df['config_name'].unique()[0] if not sa_df['config_name'].unique().empty else None
                ),
                
                html.Div(id='sa-parameter-analysis-output')
            ])
        ], className='row')
    ])
    
    return layout


def create_genetic_algorithm_dashboard(metrics_df: pd.DataFrame):
    """
    Create a dashboard for Genetic Algorithm analysis.
    
    Args:
        metrics_df: DataFrame containing metrics data
        
    Returns:
        Dash app layout for Genetic Algorithm analysis
    """
    # Filter for Genetic Algorithm data (including Island GA)
    ga_df = metrics_df[metrics_df['algorithm_name'].isin(['GeneticAlgorithm', 'IslandGA'])]
    
    if ga_df.empty:
        return html.Div("No Genetic Algorithm metrics data available.")
    
    # Create figures for Genetic Algorithm specific metrics
    crossover_fig = px.box(
        ga_df,
        x='config_name',
        y='crossover_success_rate',
        title='Crossover Success Rate',
        labels={
            'crossover_success_rate': 'Success Rate',
            'config_name': 'Configuration'
        },
        color='config_name',
        color_discrete_map=CONFIG_COLORS
    )
    
    mutation_fig = px.box(
        ga_df,
        x='config_name',
        y='mutation_impact',
        title='Mutation Impact',
        labels={
            'mutation_impact': 'Impact',
            'config_name': 'Configuration'
        },
        color='config_name',
        color_discrete_map=CONFIG_COLORS
    )
    
    # Create selection pressure figure
    selection_fig = px.box(
        ga_df,
        x='config_name',
        y='selection_pressure',
        title='Selection Pressure',
        labels={
            'selection_pressure': 'Pressure',
            'config_name': 'Configuration'
        },
        color='config_name',
        color_discrete_map=CONFIG_COLORS
    )
    
    # Create convergence plot for each configuration
    convergence_fig = go.Figure()
    
    for config in ga_df['config_name'].unique():
        config_data = ga_df[ga_df['config_name'] == config]
        if not config_data.empty:
            # Get the run with median best fitness
            median_idx = config_data['best_fitness'].argmin()
            run_data = config_data.iloc[median_idx]
            
            # Extract convergence data
            if 'best_fitness_history' in run_data.get('full_metrics', {}):
                y_data = run_data['full_metrics']['best_fitness_history']
                x_data = list(range(len(y_data)))
                
                convergence_fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines',
                    name=config,
                    line=dict(color=CONFIG_COLORS.get(config, '#000000'))
                ))
    
    convergence_fig.update_layout(
        title='Convergence Comparison',
        xaxis_title='Iterations',
        yaxis_title='Best Fitness',
        legend_title='Configuration'
    )
    
    # Create population diversity figure
    # For this, we need to extract population diversity history
    # from a representative run
    diversity_fig = go.Figure()
    
    for config in ga_df['config_name'].unique():
        config_data = ga_df[ga_df['config_name'] == config]
        if not config_data.empty:
            # Get the run with median best fitness
            median_idx = config_data['best_fitness'].argmin()
            run_data = config_data.iloc[median_idx]
            
            # Extract population diversity history
            if 'population_diversity' in run_data.get('full_metrics', {}):
                y_data = run_data['full_metrics']['population_diversity']
                x_data = list(range(len(y_data)))
                
                diversity_fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines',
                    name=config,
                    line=dict(color=CONFIG_COLORS.get(config, '#000000'))
                ))
    
    diversity_fig.update_layout(
        title='Population Diversity Over Generations',
        xaxis_title='Generation',
        yaxis_title='Diversity',
        legend_title='Configuration'
    )
    
    # Create dashboard layout
    layout = html.Div([
        html.H1("Fantasy League Optimization - Genetic Algorithm Analysis"),
        
        html.Div([
            html.Div([
                dcc.Graph(figure=convergence_fig)
            ], className='six columns'),
            
            html.Div([
                dcc.Graph(figure=diversity_fig)
            ], className='six columns')
        ], className='row'),
        
        html.Div([
            html.Div([
                dcc.Graph(figure=crossover_fig)
            ], className='four columns'),
            
            html.Div([
                dcc.Graph(figure=mutation_fig)
            ], className='four columns'),
            
            html.Div([
                dcc.Graph(figure=selection_fig)
            ], className='four columns')
        ], className='row'),
        
        html.Div([
            html.H3("Genetic Algorithm Operator Analysis"),
            
            html.Div(id='ga-operator-analysis', children=[
                html.P("Select a configuration to analyze:"),
                
                dcc.Dropdown(
                    id='ga-config-dropdown',
                    options=[{'label': config, 'value': config} for config in ga_df['config_name'].unique()],
                    value=ga_df['config_name'].unique()[0] if not ga_df['config_name'].unique().empty else None
                ),
                
                html.Div(id='ga-operator-analysis-output')
            ])
        ], className='row')
    ])
    
    return layout


def create_island_ga_dashboard(metrics_df: pd.DataFrame):
    """
    Create a dashboard for Island Genetic Algorithm analysis.
    
    Args:
        metrics_df: DataFrame containing metrics data
        
    Returns:
        Dash app layout for Island GA analysis
    """
    # Filter for Island GA data
    island_df = metrics_df[metrics_df['algorithm_name'] == 'IslandGA']
    
    if island_df.empty:
        return html.Div("No Island Genetic Algorithm metrics data available.")
    
    # Create figures for Island GA specific metrics
    migration_fig = px.box(
        island_df,
        x='config_name',
        y='migration_impact',
        title='Migration Impact',
        labels={
            'migration_impact': 'Impact',
            'config_name': 'Configuration'
        },
        color='config_name',
        color_discrete_map=CONFIG_COLORS
    )
    
    migration_success_fig = px.box(
        island_df,
        x='config_name',
        y='migration_success_rate',
        title='Migration Success Rate',
        labels={
            'migration_success_rate': 'Success Rate',
            'config_name': 'Configuration'
        },
        color='config_name',
        color_discrete_map=CONFIG_COLORS
    )
    
    # Create topology efficiency figure
    topology_fig = px.box(
        island_df,
        x='config_name',
        y='topology_efficiency',
        title='Topology Efficiency',
        labels={
            'topology_efficiency': 'Efficiency',
            'config_name': 'Configuration'
        },
        color='config_name',
        color_discrete_map=CONFIG_COLORS
    )
    
    # Create convergence plot for each configuration
    convergence_fig = go.Figure()
    
    for config in island_df['config_name'].unique():
        config_data = island_df[island_df['config_name'] == config]
        if not config_data.empty:
            # Get the run with median best fitness
            median_idx = config_data['best_fitness'].argmin()
            run_data = config_data.iloc[median_idx]
            
            # Extract convergence data
            if 'best_fitness_history' in run_data.get('full_metrics', {}):
                y_data = run_data['full_metrics']['best_fitness_history']
                x_data = list(range(len(y_data)))
                
                convergence_fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines',
                    name=config,
                    line=dict(color=CONFIG_COLORS.get(config, '#000000'))
                ))
    
    convergence_fig.update_layout(
        title='Convergence Comparison',
        xaxis_title='Iterations',
        yaxis_title='Best Fitness',
        legend_title='Configuration'
    )
    
    # Create inter-island diversity figure
    # This would typically show diversity between islands over time
    # For this mockup, we'll create a placeholder
    diversity_fig = go.Figure()
    
    for config in island_df['config_name'].unique():
        # In a real implementation, extract actual inter-island diversity data
        x_data = list(range(20))
        y_data = [np.random.uniform(0.5, 1.0) - 0.05 * i for i in x_data]
        
        diversity_fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines',
            name=config,
            line=dict(color=CONFIG_COLORS.get(config, '#000000'))
        ))
    
    diversity_fig.update_layout(
        title='Inter-Island Diversity Over Time',
        xaxis_title='Migration Events',
        yaxis_title='Inter-Island Diversity',
        legend_title='Configuration'
    )
    
    # Create dashboard layout
    layout = html.Div([
        html.H1("Fantasy League Optimization - Island Genetic Algorithm Analysis"),
        
        html.Div([
            html.Div([
                dcc.Graph(figure=convergence_fig)
            ], className='six columns'),
            
            html.Div([
                dcc.Graph(figure=diversity_fig)
            ], className='six columns')
        ], className='row'),
        
        html.Div([
            html.Div([
                dcc.Graph(figure=migration_fig)
            ], className='four columns'),
            
            html.Div([
                dcc.Graph(figure=migration_success_fig)
            ], className='four columns'),
            
            html.Div([
                dcc.Graph(figure=topology_fig)
            ], className='four columns')
        ], className='row'),
        
        html.Div([
            html.H3("Island GA Migration Analysis"),
            
            html.Div(id='island-migration-analysis', children=[
                html.P("Select a configuration to analyze:"),
                
                dcc.Dropdown(
                    id='island-config-dropdown',
                    options=[{'label': config, 'value': config} for config in island_df['config_name'].unique()],
                    value=island_df['config_name'].unique()[0] if not island_df['config_name'].unique().empty else None
                ),
                
                html.Div(id='island-migration-analysis-output')
            ])
        ], className='row')
    ])
    
    return layout


def create_main_dashboard(metrics_dir: str):
    """
    Create the main dashboard application.
    
    Args:
        metrics_dir: Directory containing metrics JSON files
        
    Returns:
        Dash application
    """
    # Load metrics data
    metrics_df = load_metrics_data(metrics_dir)
    
    # Create Dash app
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    
    # Define app layout with tabs
    app.layout = html.Div([
        html.H1("Fantasy League Optimization Dashboard"),
        
        dcc.Tabs(id='dashboard-tabs', value='global-comparison', children=[
            dcc.Tab(label='Global Comparison', value='global-comparison'),
            dcc.Tab(label='Hill Climbing', value='hill-climbing'),
            dcc.Tab(label='Simulated Annealing', value='simulated-annealing'),
            dcc.Tab(label='Genetic Algorithm', value='genetic-algorithm'),
            dcc.Tab(label='Island GA', value='island-ga')
        ]),
        
        html.Div(id='dashboard-content')
    ])
    
    # Define callback to update dashboard content based on selected tab
    @app.callback(
        Output('dashboard-content', 'children'),
        Input('dashboard-tabs', 'value')
    )
    def update_dashboard(tab):
        if tab == 'global-comparison':
            return create_global_comparison_dashboard(metrics_df)
        elif tab == 'hill-climbing':
            return create_hill_climbing_dashboard(metrics_df)
        elif tab == 'simulated-annealing':
            return create_simulated_annealing_dashboard(metrics_df)
        elif tab == 'genetic-algorithm':
            return create_genetic_algorithm_dashboard(metrics_df)
        elif tab == 'island-ga':
            return create_island_ga_dashboard(metrics_df)
        else:
            return html.Div("Unknown tab selected.")
    
    # Define callback for metrics correlation analysis
    @app.callback(
        Output('metrics-correlation-output', 'children'),
        [Input('primary-metric-dropdown', 'value'),
         Input('secondary-metric-dropdown', 'value')]
    )
    def update_metrics_correlation(primary_metric, secondary_metric):
        if primary_metric == secondary_metric:
            return html.Div("Please select different metrics for comparison.")
        
        if primary_metric not in metrics_df.columns or secondary_metric not in metrics_df.columns:
            return html.Div("One or both selected metrics are not available in the data.")
        
        # Calculate correlation
        correlation = metrics_df[[primary_metric, secondary_metric]].corr().iloc[0, 1]
        
        # Create scatter plot
        correlation_fig = px.scatter(
            metrics_df,
            x=primary_metric,
            y=secondary_metric,
            color='algorithm_name',
            title=f'Correlation between {primary_metric} and {secondary_metric}: {correlation:.3f}',
            labels={
                primary_metric: primary_metric.replace('_', ' ').title(),
                secondary_metric: secondary_metric.replace('_', ' ').title()
            },
            color_discrete_map=ALGORITHM_COLORS
        )
        
        return html.Div([
            html.P(f"Correlation coefficient: {correlation:.3f}"),
            dcc.Graph(figure=correlation_fig)
        ])
    
    # Define callback for Hill Climbing configuration comparison
    @app.callback(
        Output('hc-config-comparison-output', 'children'),
        Input('hc-config-checklist', 'value')
    )
    def update_hc_config_comparison(selected_configs):
        if not selected_configs:
            return html.Div("Please select at least one configuration.")
        
        # Filter data for selected configurations
        filtered_df = metrics_df[
            (metrics_df['algorithm_name'] == 'HillClimbing') & 
            (metrics_df['config_name'].isin(selected_configs))
        ]
        
        if filtered_df.empty:
            return html.Div("No data available for selected configurations.")
        
        # Create comparison table
        comparison_data = filtered_df.groupby('config_name').agg({
            'best_fitness': ['mean', 'std', 'min'],
            'runtime_seconds': ['mean', 'std'],
            'function_evaluations': ['mean', 'std'],
            'neighbors_evaluated': ['mean', 'sum'],
            'local_optima_count': ['mean', 'sum']
        }).reset_index()
        
        # Format table for display
        table_data = []
        for _, row in comparison_data.iterrows():
            table_data.append({
                'Configuration': row['config_name'],
                'Avg Best Fitness': f"{row[('best_fitness', 'mean')]:.4f} ± {row[('best_fitness', 'std')]:.4f}",
                'Min Best Fitness': f"{row[('best_fitness', 'min')]:.4f}",
                'Avg Runtime (s)': f"{row[('runtime_seconds', 'mean')]:.2f} ± {row[('runtime_seconds', 'std')]:.2f}",
                'Avg Evaluations': f"{row[('function_evaluations', 'mean')]:.0f} ± {row[('function_evaluations', 'std')]:.0f}",
                'Total Neighbors': f"{row[('neighbors_evaluated', 'sum')]:.0f}",
                'Total Local Optima': f"{row[('local_optima_count', 'sum')]:.0f}"
            })
        
        return html.Div([
            html.Table(
                # Header
                [html.Tr([html.Th(col) for col in table_data[0].keys()])] +
                # Rows
                [html.Tr([html.Td(table_data[i][col]) for col in table_data[0].keys()]) for i in range(len(table_data))],
                style={'border-collapse': 'collapse', 'width': '100%'}
            )
        ])
    
    # Define callback for Simulated Annealing parameter analysis
    @app.callback(
        Output('sa-parameter-analysis-output', 'children'),
        Input('sa-config-dropdown', 'value')
    )
    def update_sa_parameter_analysis(selected_config):
        if not selected_config:
            return html.Div("Please select a configuration.")
        
        # Filter data for selected configuration
        filtered_df = metrics_df[
            (metrics_df['algorithm_name'] == 'SimulatedAnnealing') & 
            (metrics_df['config_name'] == selected_config)
        ]
        
        if filtered_df.empty:
            return html.Div("No data available for selected configuration.")
        
        # Get parameters for this configuration
        params = filtered_df.iloc[0]['full_metrics'].get('parameters', {})
        
        # Create parameter table
        param_items = []
        for key, value in params.items():
            param_items.append(html.Tr([
                html.Td(key),
                html.Td(str(value))
            ]))
        
        # Create temperature vs fitness figure
        # For this, we need to extract temperature history and fitness history
        # from a representative run
        temp_fitness_fig = go.Figure()
        
        # Get the run with median best fitness
        median_idx = filtered_df['best_fitness'].argmin()
        run_data = filtered_df.iloc[median_idx]
        
        # Extract temperature and fitness history
        if 'temperature_history' in run_data.get('full_metrics', {}) and 'raw_data' in run_data:
            temp_history = run_data['full_metrics']['temperature_history']
            fitness_history = run_data['raw_data'].get('fitness_history', [])
            
            # Ensure lengths match (temperature history might be shorter)
            min_length = min(len(temp_history), len(fitness_history))
            
            if min_length > 0:
                # Sample points to avoid overcrowding
                sample_indices = np.linspace(0, min_length - 1, min(100, min_length)).astype(int)
                
                temp_fitness_fig.add_trace(go.Scatter(
                    x=[temp_history[i] for i in sample_indices],
                    y=[fitness_history[i] for i in sample_indices],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=[i for i in sample_indices],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Iteration')
                    )
                ))
                
                temp_fitness_fig.update_layout(
                    title='Temperature vs Fitness',
                    xaxis_title='Temperature',
                    yaxis_title='Fitness',
                    showlegend=False
                )
        
        return html.Div([
            html.H4("Configuration Parameters"),
            
            html.Table(
                [html.Tr([html.Th("Parameter"), html.Th("Value")])] +
                param_items,
                style={'border-collapse': 'collapse', 'width': '50%', 'margin-bottom': '20px'}
            ),
            
            dcc.Graph(figure=temp_fitness_fig)
        ])
    
    # Define callback for Genetic Algorithm operator analysis
    @app.callback(
        Output('ga-operator-analysis-output', 'children'),
        Input('ga-config-dropdown', 'value')
    )
    def update_ga_operator_analysis(selected_config):
        if not selected_config:
            return html.Div("Please select a configuration.")
        
        # Filter data for selected configuration
        filtered_df = metrics_df[
            (metrics_df['algorithm_name'].isin(['GeneticAlgorithm', 'IslandGA'])) & 
            (metrics_df['config_name'] == selected_config)
        ]
        
        if filtered_df.empty:
            return html.Div("No data available for selected configuration.")
        
        # Get parameters for this configuration
        params = filtered_df.iloc[0]['full_metrics'].get('parameters', {})
        
        # Create parameter table
        param_items = []
        for key, value in params.items():
            # Skip complex objects like functions
            if not callable(value) and not isinstance(value, dict) and not isinstance(value, list):
                param_items.append(html.Tr([
                    html.Td(key),
                    html.Td(str(value))
                ]))
        
        # Create fitness distribution figure
        # For this, we need to extract fitness values from a representative run
        fitness_dist_fig = go.Figure()
        
        # Get the run with median best fitness
        median_idx = filtered_df['best_fitness'].argmin()
        run_data = filtered_df.iloc[median_idx]
        
        # Extract fitness history
        if 'raw_data' in run_data and 'fitness_history' in run_data['raw_data']:
            fitness_history = run_data['raw_data']['fitness_history']
            
            # Create histogram
            fitness_dist_fig.add_trace(go.Histogram(
                x=fitness_history,
                nbinsx=30,
                marker_color='rgba(0, 0, 255, 0.7)'
            ))
            
            fitness_dist_fig.update_layout(
                title='Fitness Distribution',
                xaxis_title='Fitness',
                yaxis_title='Count',
                showlegend=False
            )
        
        return html.Div([
            html.H4("Configuration Parameters"),
            
            html.Table(
                [html.Tr([html.Th("Parameter"), html.Th("Value")])] +
                param_items,
                style={'border-collapse': 'collapse', 'width': '50%', 'margin-bottom': '20px'}
            ),
            
            dcc.Graph(figure=fitness_dist_fig)
        ])
    
    # Define callback for Island GA migration analysis
    @app.callback(
        Output('island-migration-analysis-output', 'children'),
        Input('island-config-dropdown', 'value')
    )
    def update_island_migration_analysis(selected_config):
        if not selected_config:
            return html.Div("Please select a configuration.")
        
        # Filter data for selected configuration
        filtered_df = metrics_df[
            (metrics_df['algorithm_name'] == 'IslandGA') & 
            (metrics_df['config_name'] == selected_config)
        ]
        
        if filtered_df.empty:
            return html.Div("No data available for selected configuration.")
        
        # Get parameters for this configuration
        params = filtered_df.iloc[0]['full_metrics'].get('parameters', {})
        
        # Create parameter table
        param_items = []
        for key, value in params.items():
            # Skip complex objects like functions
            if not callable(value) and not isinstance(value, dict) and not isinstance(value, list):
                param_items.append(html.Tr([
                    html.Td(key),
                    html.Td(str(value))
                ]))
        
        # Create island fitness comparison figure
        # For this, we need to extract per-island fitness data
        # from a representative run
        island_fitness_fig = go.Figure()
        
        # Get the run with median best fitness
        median_idx = filtered_df['best_fitness'].argmin()
        run_data = filtered_df.iloc[median_idx]
        
        # Extract island fitness history
        if 'raw_data' in run_data and 'island_best_fitness' in run_data['raw_data']:
            island_fitness = run_data['raw_data']['island_best_fitness']
            
            for island_idx, fitness_history in island_fitness.items():
                island_fitness_fig.add_trace(go.Scatter(
                    x=list(range(len(fitness_history))),
                    y=fitness_history,
                    mode='lines',
                    name=f'Island {island_idx}'
                ))
            
            island_fitness_fig.update_layout(
                title='Island Fitness Comparison',
                xaxis_title='Generation',
                yaxis_title='Best Fitness',
                legend_title='Island'
            )
        
        return html.Div([
            html.H4("Configuration Parameters"),
            
            html.Table(
                [html.Tr([html.Th("Parameter"), html.Th("Value")])] +
                param_items,
                style={'border-collapse': 'collapse', 'width': '50%', 'margin-bottom': '20px'}
            ),
            
            dcc.Graph(figure=island_fitness_fig)
        ])
    
    return app


def run_dashboard(metrics_dir: str, host: str = '0.0.0.0', port: int = 8050, debug: bool = False):
    """
    Run the dashboard application.
    
    Args:
        metrics_dir: Directory containing metrics JSON files
        host: Host to run the server on
        port: Port to run the server on
        debug: Whether to run in debug mode
    """
    app = create_main_dashboard(metrics_dir)
    app.run_server(host=host, port=port, debug=debug)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Fantasy League Optimization Dashboard')
    parser.add_argument('--metrics-dir', type=str, required=True, help='Directory containing metrics JSON files')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8050, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    run_dashboard(args.metrics_dir, args.host, args.port, args.debug)

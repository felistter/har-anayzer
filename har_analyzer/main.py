import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple

def load_har_file(file_path: str) -> List[Dict]:
    """Load and parse HAR file, extracting only GraphQL POST requests."""
    with open(file_path, 'r') as f:
        har_data = json.load(f)
    
    graphql_entries = []
    for entry in har_data['log']['entries']:
        if ('graphql' in entry['request']['url'] and 
            entry['request']['method'] == 'POST' and
            'postData' in entry['request']):
            try:
                # Parse GraphQL operation details
                post_data = json.loads(entry['request']['postData']['text'])
                operation_name = post_data.get('operationName', 'Unknown')
                query = post_data.get('query', '')
                
                # Determine operation type
                operation_type = 'unknown'
                if query.strip().startswith('mutation'):
                    operation_type = 'mutation'
                elif query.strip().startswith('query'):
                    operation_type = 'query'
                
                # Calculate timing metrics
                timings = entry['timings']
                total_time = sum(v for v in timings.values() if v > 0)
                
                graphql_entries.append({
                    'operation_name': operation_name,
                    'operation_type': operation_type,
                    'total_time': total_time,
                    'wait_time': timings['wait'],
                    'send_time': timings['send'],
                    'receive_time': timings['receive'],
                    'response_size': entry['response']['bodySize'],
                    'timestamp': datetime.fromisoformat(entry['startedDateTime'].replace('Z', '+00:00')),
                    'status': entry['response']['status']
                })
            except json.JSONDecodeError:
                continue
    
    return graphql_entries

def analyze_operations(entries: List[Dict]) -> Tuple[pd.DataFrame, Dict]:
    """Analyze GraphQL operations and return insights."""
    df = pd.DataFrame(entries)
    
    # Calculate statistical metrics
    stats = {
        'total_operations': len(df),
        'unique_operations': df['operation_name'].nunique(),
        'avg_response_time': df['total_time'].mean(),
        'p95_response_time': df['total_time'].quantile(0.95),
        'median_response_time': df['total_time'].median()
    }
    
    # Calculate per-operation metrics
    operation_metrics = df.groupby('operation_name').agg({
        'total_time': ['count', 'mean', 'std', 'max', 'min'],
        'response_size': 'mean',
        'operation_type': 'first'
    }).reset_index()
    
    # Calculate coefficient of variation to identify unstable operations
    operation_metrics['cv'] = (
        operation_metrics[('total_time', 'std')] / 
        operation_metrics[('total_time', 'mean')]
    )
    
    return df, operation_metrics, stats

def identify_problematic_queries(df: pd.DataFrame, operation_metrics: pd.DataFrame) -> Dict:
    """Identify operations that need attention."""
    problems = {
        'slow_queries': [],
        'unstable_queries': [],
        'frequent_queries': [],
        'large_responses': []
    }
    
    # Identify slow queries (above 95th percentile)
    p95_time = df['total_time'].quantile(0.95)
    slow_ops = operation_metrics[
        operation_metrics[('total_time', 'mean')] > p95_time
    ]
    problems['slow_queries'] = slow_ops['operation_name'].tolist()
    
    # Identify unstable queries (high coefficient of variation)
    unstable_ops = operation_metrics[
        operation_metrics['cv'] > 0.5  # More than 50% variation
    ]
    problems['unstable_queries'] = unstable_ops['operation_name'].tolist()
    
    # Identify frequently called queries
    frequent_ops = operation_metrics[
        operation_metrics[('total_time', 'count')] > operation_metrics[('total_time', 'count')].quantile(0.75)
    ]
    problems['frequent_queries'] = frequent_ops['operation_name'].tolist()
    
    # Identify large response sizes
    large_ops = operation_metrics[
        operation_metrics[('response_size', 'mean')] > operation_metrics[('response_size', 'mean')].quantile(0.95)
    ]
    problems['large_responses'] = large_ops['operation_name'].tolist()
    
    return problems

def plot_top_slow_operations(df: pd.DataFrame, output_prefix: str, n_top: int = 50):
    """Plot top N slowest queries with frequency-based coloring."""
    plt.figure(figsize=(12, 12), dpi=300)
    
    # Calculate average times and counts
    operation_stats = df.groupby('operation_name').agg({
        'total_time': 'mean',
        'operation_name': 'count'  # This gives us the frequency
    }).rename(columns={'operation_name': 'frequency'})
    
    # Get top N slowest operations
    top_slow = operation_stats.nlargest(n_top, 'total_time').iloc[::-1]
    
    # Normalize frequencies for color scaling
    max_freq = top_slow['frequency'].max()
    normalized_freq = top_slow['frequency'] / max_freq
    
    # Create color palette - using Blues colormap for frequency indication
    colors = plt.cm.Reds(normalized_freq)
    
    # Create bar plot
    bars = plt.barh(y=range(len(top_slow)), width=top_slow['total_time'], color=colors)
    
    # Add frequency annotations to each bar
    for i, (time, freq) in enumerate(zip(top_slow['total_time'], top_slow['frequency'])):
        plt.text(time + 20, i, f'({freq} calls)', 
                va='center', fontsize=8, alpha=0.7)
    
    # Add 1000ms threshold line
    plt.axvline(x=1000, color='red', linestyle='--', alpha=0.7, 
                label='1000ms threshold')
    
    # Customize the plot
    plt.yticks(range(len(top_slow)), top_slow.index, fontsize=8)
    plt.xlabel('Average Response Time (ms)', fontsize=10)
    plt.title('Top 50 Slowest GraphQL queries\nColor intensity indicates relative frequency', 
              fontsize=12, pad=20)
    
    # Add stats in the corner
    stats_text = (f"Total unique operations: {len(df['operation_name'].unique())}\n"
                  f"Total calls: {len(df)}\n"
                  f"Avg response time: {df['total_time'].mean():.0f}ms")
    plt.text(0.95, 0.02, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
             fontsize=8)
    
    # Add legend for threshold line
    plt.legend(loc='upper right')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{output_prefix}.png')
    plt.close()

def visualize_analysis(df: pd.DataFrame, output_prefix: str):
    """Generate visualizations for the analysis."""
    # Set style
    # plt.style.use('seaborn')
    
    # 1. Response time distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='total_time', bins=30)
    plt.title('Distribution of Response Times')
    plt.xlabel('Response Time (ms)')
    plt.ylabel('Count')
    plt.savefig(f'{output_prefix}_response_times.png')
    plt.close()
    
    # 2. Top 10 slowest operations (average)
    # plt.figure(figsize=(12, 12), dpi=300)
    # top_slow = df.groupby('operation_name')['total_time'].mean().nlargest(50)
    # sns.barplot(x=top_slow.values, y=top_slow.index)
    # plt.axvline(x=1000, color='red', linestyle='--')
    # plt.title('Top 50 Slowest Operations (Average)')
    # plt.xlabel('Average Response Time (ms)')
    # plt.tight_layout()
    # plt.savefig(f'{output_prefix}_top_50_slow.png')
    # plt.close()
    plot_top_slow_operations(df, f'{output_prefix}_top_50_slow')
    
    # 3. Operation timing breakdown
    plt.figure(figsize=(12, 6))
    timing_data = df[['wait_time', 'send_time', 'receive_time']].mean()
    plt.pie(timing_data, labels=timing_data.index, autopct='%1.1f%%')
    plt.title('Average Time Distribution')
    plt.savefig(f'{output_prefix}_timing_breakdown.png')
    plt.close()

def main(har_file: str, output_prefix: str):
    """Main function to analyze HAR file and generate report."""
    # Load and analyze data
    entries = load_har_file(har_file)
    df, operation_metrics, stats = analyze_operations(entries)
    problems = identify_problematic_queries(df, operation_metrics)
    
    # Generate visualizations
    visualize_analysis(df, output_prefix)
    
    # Print analysis report
    print("\n=== GraphQL Operations Analysis ===")
    print(f"\nTotal Operations: {stats['total_operations']}")
    print(f"Unique Operations: {stats['unique_operations']}")
    print(f"Average Response Time: {stats['avg_response_time']:.2f}ms")
    print(f"95th Percentile Response Time: {stats['p95_response_time']:.2f}ms")
    print(f"Median Response Time: {stats['median_response_time']:.2f}ms")
    
    print("\nOperations Requiring Attention:")
    print("\n1. Slow Queries (Above 95th percentile):")
    for op in problems['slow_queries']:
        print(f"  - {op}")
    
    print("\n2. Unstable Queries (High variance):")
    for op in problems['unstable_queries']:
        print(f"  - {op}")
    
    print("\n3. Frequent Queries (Top 25% by count):")
    for op in problems['frequent_queries']:
        print(f"  - {op}")
    
    print("\n4. Large Response Sizes:")
    for op in problems['large_responses']:
        print(f"  - {op}")
    
    # Export detailed metrics to CSV
    operation_metrics.to_csv(f'{output_prefix}_metrics.csv')

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script.py <har_file> <output_prefix>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])
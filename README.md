# GraphQL HAR Performance Analyzer

A Python tool for analyzing and visualizing GraphQL performance data from HAR files. Helps identify slow, unstable, and frequently called operations in GraphQL API.

## Features

- 📊 Analyzes GraphQL operations timing and patterns
- 🔍 Identifies problematic queries needing optimization
- 📈 Generates visual performance reports
- 📉 Shows timing distribution and breakdowns
- 🎯 Highlights operations exceeding performance thresholds

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yaradigitallabs/har-anayzer.git
cd har-anayzer
```

2. Install dependencies:
Install poetry as described in the [official documentation](https://python-poetry.org/docs/#installation).
```bash
poetry install
```

## Usage

1. Record a HAR file using Chrome DevTools:
   - Open DevTools (F12)
   - Go to Network tab
   - Check "Preserve log"
   - Filter by "Fetch/XHR"
   - Perform your operations
   - Right-click → "Save all as HAR with content"

2. Run the analyzer:
```bash
poetry run python har_analyzer/main.py input.har output_prefix
```

This will generate:
- `output_prefix_metrics.csv` - Detailed metrics for all operations
- `output_prefix_response_times.png` - Distribution of response times
- `output_prefix_top_50_slow.png` - Top 50 slowest operations with frequency indicators
- `output_prefix_timing_breakdown.png` - Timing phase analysis

## Understanding the Output

### Top 50 Slowest Operations Chart
- Bar length shows average response time
- Color intensity indicates call frequency (darker = more calls)
- Red dashed line marks 1000ms threshold
- Numbers in parentheses show call count
- Summary stats in bottom right corner

### Analysis Report
The tool identifies:
- Slow queries (above 95th percentile)
- Unstable queries (high variance)
- Frequently called queries
- Queries with large response sizes

Example output:
```
=== GraphQL Operations Analysis ===
Total Operations: 1500
Unique Operations: 45
Average Response Time: 235.7ms
95th Percentile Response Time: 892.3ms
```

## Notes

- Only analyzes GraphQL POST requests
- Requires HAR files with complete request/response data
- Best used with Chrome DevTools HAR export
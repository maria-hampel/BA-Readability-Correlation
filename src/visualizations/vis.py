import pandas as pd
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Function to parse and extract metrics from a single JSON string
def parse_json_string(json_string):
    try:
        json_string = json_string.replace("'", "\"")
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Problematic JSON string: {json_string}")
        return {}

# Function to process metrics from all lines
def process_metrics(df):
    all_metrics = []

    for _, row in df.iterrows():
        metrics = []
        for col in df.columns[2:10]:  # Columns containing JSON strings
            parsed_metric = parse_json_string(row[col])
            if isinstance(parsed_metric, dict):  # Only append if parsing was successful and is a dictionary
                metrics.append(parsed_metric)
            else:
                print(f"Skipping non-dictionary metric: {parsed_metric}")
        all_metrics.append(metrics)

    return all_metrics

# Function to visualize each metric
def visualize_metrics(all_metrics):
    num_lines = len(all_metrics)
    metric_dict = defaultdict(list)

    # Collect all metrics by key
    for metrics in all_metrics:
        for metric in metrics:
            if isinstance(metric, dict):
                comps = metric.get('comps', {})
                for key, value in comps.items():
                    metric_dict[key].append(float(value))
            else:
                print(f"Skipping non-dictionary metric: {metric}")

    # Plot each metric
    for metric, values in metric_dict.items():
        plt.figure(figsize=(10, 6))
        plt.plot(range(num_lines), values, marker='o', linestyle='-')
        plt.xlabel('Document Index')
        plt.ylabel('Value')
        plt.title(f'Metric: {metric}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Main function
def main():
    file_path = 'data/api/beir-arguana/batch_1.csv'  # Replace with your actual file path
    df = pd.read_csv(file_path)
    all_metrics = process_metrics(df)
    visualize_metrics(all_metrics)

if __name__ == "__main__":
    main()

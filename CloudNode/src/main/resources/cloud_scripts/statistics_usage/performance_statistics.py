import json
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import csv
import sys


# Customize font globally for better readability
plt.rcParams.update({
    'font.size': 12,  # Default font size
    'font.family': 'Arial',  # Change font family
    'axes.titlesize': 16,  # Title font size
    'axes.labelsize': 14,  # Axis labels font size
    'xtick.labelsize': 12,  # X-ticks font size
    'ytick.labelsize': 12,  # Y-ticks font size
})


def plot_performance_data(edge_performance_map):
    cloud_result = []
    dates = set()
    fog_performances_across_edges = []

    for edge_name, performance_results in edge_performance_map.items():
        # Transform date to ensure format is correct
        dates_results = [datetime.strptime(result['date'].strip('"[]'), '%Y-%m-%d').strftime('%Y-%m-%d')
                         for result in performance_results]
        for date in dates_results:
            dates.add(date)

        for i, result in enumerate(performance_results):
            if len(fog_performances_across_edges) <= i:
                fog_performances_across_edges.append([])
            fog_performances_across_edges[i].append(result['fogModelPerformance'])

    for performances in fog_performances_across_edges:
        non_zero_performances = [perf for perf in performances if perf != 0.0]
        if non_zero_performances:
            if len(non_zero_performances) == 1:
                avg_performance = non_zero_performances[0]  # Keep the single non-zero value
            else:
                avg_performance = np.mean(non_zero_performances)  # Mean of non-zero values
        else:
            avg_performance = 0.0  # If all performances are 0.0, set to 0.0 (or another default value)

        cloud_result.append(avg_performance)

    dates_list = list(dates)
    dates_list.sort(key=lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%Y-%m-%d'))

    # Remove the first entry from both lists for clarity (if needed)
    cloud_result.pop(0)
    dates_list.pop(0)

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot(dates_list, cloud_result, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)

    # Formatting the plot
    plt.xlabel('dates', fontsize=14, weight='bold')
    plt.ylabel('mse', fontsize=14, weight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    # Set a tighter layout
    plt.tight_layout()

    # Save the plot as an image
    plt.savefig('/app/images/cloud_performance.png', format='png', dpi=300)
    plt.close()

    # Create CSV with the performance data
    create_performance_csv(dates_list, cloud_result, '/app/images/cloud_performance_data.csv')


def create_performance_csv(dates, performances, filename_csv):
    # Prepare the table data for CSV
    column_labels = ['Iteration'] + dates
    table_data = ['Cloud MSE'] + performances

    # Write to a CSV file
    with open(filename_csv, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the column headers (dates)
        writer.writerow(column_labels)

        # Write the performance data row
        writer.writerow(table_data)

    print(f"CSV saved as {filename_csv}")


if __name__ == '__main__':
    edge_performance_json = sys.argv[1]

    # Convert JSON data to Python objects
    edge_performance_map_arg = json.loads(edge_performance_json)

    plot_performance_data(edge_performance_map_arg)

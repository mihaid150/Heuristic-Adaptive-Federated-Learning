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
    """
    Plot the cloud performance data (Mean Squared Error) based on edge performance data.

    Args:
        edge_performance_map (dict): Dictionary containing edge performance data, where keys are edge names
                                     and values are lists of performance metrics.
                                     Each performance metric should have 'date' and 'fogModelPerformance' fields.

    Outputs:
        - A PNG file with the performance chart saved to '/app/images/cloud_performance.png'.
        - A CSV file with the performance data saved to '/app/images/cloud_performance_data.csv'.
    """
    cloud_result = []
    dates = set()
    fog_performances_across_edges = []

    for edge_name, performance_results in edge_performance_map.items():
        # Extract and format dates from performance results
        dates_results = [datetime.strptime(result['date'].strip('"[]'), '%Y-%m-%d').strftime('%Y-%m-%d')
                         for result in performance_results]
        for date in dates_results:
            dates.add(date)

        # Collect fog model performances
        for i, result in enumerate(performance_results):
            if len(fog_performances_across_edges) <= i:
                fog_performances_across_edges.append([])
            fog_performances_across_edges[i].append(result['fogModelPerformance'])

    for performances in fog_performances_across_edges:
        # Compute the average performance of non-zero values
        non_zero_performances = [perf for perf in performances if perf != 0.0]
        if non_zero_performances:
            avg_performance = np.mean(non_zero_performances) if len(non_zero_performances) > 1 else non_zero_performances[0]
        else:
            avg_performance = 0.0  # Default value if all performances are zero
        cloud_result.append(avg_performance)

    # Sort dates and synchronize with performances
    dates_list = sorted(dates, key=lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%Y-%m-%d'))

    # Remove the first entry from both lists for clarity (optional)
    cloud_result.pop(0)
    dates_list.pop(0)

    # Plot the cloud performance data
    plt.figure(figsize=(12, 8))
    plt.plot(dates_list, cloud_result, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)

    # Formatting the plot
    plt.xlabel('Dates', fontsize=14, weight='bold')
    plt.ylabel('MSE', fontsize=14, weight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(['Cloud MSE'], loc='best')

    # Save the plot as an image
    plt.tight_layout()
    plt.savefig('/app/images/cloud_performance.png', format='png', dpi=300)
    plt.close()

    # Save performance data to a CSV file
    create_performance_csv(dates_list, cloud_result, '/app/images/cloud_performance_data.csv')


def create_performance_csv(dates, performances, filename_csv):
    """
    Save the performance data to a CSV file.

    Args:
        dates (list): List of date strings.
        performances (list): List of performance values corresponding to the dates.
        filename_csv (str): Path to save the CSV file.

    Outputs:
        - A CSV file saved to the specified path.
    """
    column_labels = ['Iteration'] + dates
    table_data = ['Cloud MSE'] + performances

    with open(filename_csv, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the column headers (dates)
        writer.writerow(column_labels)

        # Write the performance data row
        writer.writerow(table_data)

    print(f"CSV saved as {filename_csv}")


if __name__ == '__main__':
    """
    Main script to process and plot cloud performance data.

    Args:
        sys.argv[1] (str): JSON string containing edge performance data.

    Outputs:
        - Generates a performance chart and saves it as a PNG file.
        - Saves the performance data to a CSV file.
    """
    edge_performance_json = sys.argv[1]

    # Convert JSON data to Python objects
    edge_performance_map_arg = json.loads(edge_performance_json)

    # Process and plot performance data
    plot_performance_data(edge_performance_map_arg)

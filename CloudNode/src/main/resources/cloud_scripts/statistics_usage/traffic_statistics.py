import sys
import json
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import csv


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)


# Customize font globally for better readability
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})


def plot_traffic(data, dates, filename, label_prefix, ylabel):
    """
    Plot traffic data (cloud, fog, edge) over time and save the plot.

    Args:
        data (dict): Dictionary containing traffic data for 'cloud', 'fog', and 'edge'.
        dates (list): List of datetime objects representing the x-axis.
        filename (str): Path to save the plot image.
        label_prefix (str): Prefix for the legend labels (e.g., 'Incoming', 'Outgoing').
        ylabel (str): Label for the y-axis (e.g., 'kB').
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(dates, data['cloud'], label=f'{label_prefix} Cloud', color='blue', marker='o', linestyle='-', linewidth=2,
            markersize=8)
    ax.plot(dates, data['fog'], label=f'{label_prefix} Fog', color='green', marker='o', linestyle='-', linewidth=2,
            markersize=8)
    ax.plot(dates, data['edge'], label=f'{label_prefix} Edge', color='red', marker='o', linestyle='-', linewidth=2,
            markersize=8)

    ax.set_xlabel('Date', fontsize=14, weight='bold')
    ax.set_ylabel(ylabel, fontsize=14, weight='bold')
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend(loc='best', fontsize=12)

    # Format x-axis to show dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.tick_params(axis='x', rotation=45)

    # Save the chart
    plt.tight_layout()
    plt.savefig(filename, format='png', dpi=300)
    plt.close()


def create_csv(data, dates, filename_csv):
    """
    Create a CSV file for traffic data.

    Args:
        data (dict): Dictionary containing traffic data for 'cloud', 'fog', and 'edge'.
        dates (list): List of datetime objects representing the column headers.
        filename_csv (str): Path to save the CSV file.
    """
    # Prepare the CSV data: dates as the column headers, cloud/fog/edge as rows
    column_labels = ['Iteration'] + [date.strftime('%Y-%m-%d') for date in dates]
    cloud_row = ['Cloud'] + list(data['cloud'])
    fog_row = ['Fog'] + list(data['fog'])
    edge_row = ['Edge'] + list(data['edge'])

    # Write to a CSV file
    with open(filename_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(column_labels)  # Write headers (dates)
        writer.writerow(cloud_row)  # Write Cloud row
        writer.writerow(fog_row)  # Write Fog row
        writer.writerow(edge_row)  # Write Edge row

    print(f"CSV saved as {filename_csv}")


def plot_traffic_data(cloud_incoming, cloud_outgoing, fog_incoming, fog_outgoing, edge_incoming_map, edge_outgoing_map):
    """
    Process and plot incoming and outgoing traffic data for cloud, fog, and edge nodes.

    Args:
        cloud_incoming (list): Incoming traffic data for the cloud.
        cloud_outgoing (list): Outgoing traffic data for the cloud.
        fog_incoming (list of lists): Incoming traffic data for multiple fog nodes.
        fog_outgoing (list of lists): Outgoing traffic data for multiple fog nodes.
        edge_incoming_map (dict): Incoming traffic data for multiple edges, keyed by edge name.
        edge_outgoing_map (dict): Outgoing traffic data for multiple edges, keyed by edge name.

    Outputs:
        - Saves plots as PNG files ('/app/images/in_saga.png' and '/app/images/out_saga.png').
        - Saves traffic data as CSV files ('/app/images/in_saga.csv' and '/app/images/out_saga.csv').
    """
    logging.info('Starting to plot traffic data.')

    # Generate dates for x-axis, starting from 2013-07-10
    start_date = datetime.strptime("2013-07-10", "%Y-%m-%d")
    num_iterations = max(len(cloud_incoming), len(cloud_outgoing))
    dates = [start_date + timedelta(days=i) for i in range(num_iterations)]

    # Prepare data for incoming traffic
    incoming_data = {
        'cloud': np.array(cloud_incoming),
        'fog': np.mean(np.array(fog_incoming), axis=0),
        'edge': np.mean([np.mean(np.array(traffic), axis=0) for traffic in edge_incoming_map.values()], axis=0)
    }

    # Prepare data for outgoing traffic
    outgoing_data = {
        'cloud': np.array(cloud_outgoing),
        'fog': np.mean(np.array(fog_outgoing), axis=0),
        'edge': np.mean([np.mean(np.array(traffic), axis=0) for traffic in edge_outgoing_map.values()], axis=0)
    }

    # Plot and save incoming traffic
    plot_traffic(incoming_data, dates, '/app/images/in_saga.png', 'Incoming', 'kB')

    # Plot and save outgoing traffic
    plot_traffic(outgoing_data, dates, '/app/images/out_saga.png', 'Outgoing', 'kB')

    # Create CSV files for traffic data
    create_csv(incoming_data, dates, '/app/images/in_saga.csv')
    create_csv(outgoing_data, dates, '/app/images/out_saga.csv')


if __name__ == '__main__':
    """
    Main function to process input JSON data and generate traffic plots and CSV files.

    Args:
        sys.argv[1]: JSON string for cloud incoming traffic.
        sys.argv[2]: JSON string for cloud outgoing traffic.
        sys.argv[3]: JSON string for fog incoming traffic.
        sys.argv[4]: JSON string for fog outgoing traffic.
        sys.argv[5]: JSON string for edge incoming traffic.
        sys.argv[6]: JSON string for edge outgoing traffic.
    """
    logging.info('Script started. Parsing input JSON data.')

    # Receive JSON data from arguments
    cloud_incoming_traffic_json = sys.argv[1]
    cloud_outgoing_traffic_json = sys.argv[2]
    fog_incoming_traffic_list_json = sys.argv[3]
    fog_outgoing_traffic_list_json = sys.argv[4]
    edge_incoming_traffic_list_map_json = sys.argv[5]
    edge_outgoing_traffic_list_map_json = sys.argv[6]

    # Convert JSON data to Python objects
    cloud_incoming_traffic = json.loads(cloud_incoming_traffic_json)
    cloud_outgoing_traffic = json.loads(cloud_outgoing_traffic_json)
    fog_incoming_traffic_list = json.loads(fog_incoming_traffic_list_json)
    fog_outgoing_traffic_list = json.loads(fog_outgoing_traffic_list_json)
    edge_incoming_traffic_list_map = json.loads(edge_incoming_traffic_list_map_json)
    edge_outgoing_traffic_list_map = json.loads(edge_outgoing_traffic_list_map_json)

    # Process and plot traffic data
    plot_traffic_data(cloud_incoming_traffic, cloud_outgoing_traffic, fog_incoming_traffic_list,
                      fog_outgoing_traffic_list, edge_incoming_traffic_list_map, edge_outgoing_traffic_list_map)

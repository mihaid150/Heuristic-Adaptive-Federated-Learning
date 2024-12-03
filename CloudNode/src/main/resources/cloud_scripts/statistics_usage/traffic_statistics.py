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
    'font.family': 'DejaVu Sans',  # Use the default Matplotlib font
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})


def plot_traffic(data, dates, filename, label_prefix, ylabel):
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(dates, data['cloud'], label=f'{label_prefix} Cloud', color='blue', marker='o', linestyle='-', linewidth=2,
            markersize=8)
    ax.plot(dates, data['fog'], label=f'{label_prefix} Fog', color='green', marker='o', linestyle='-', linewidth=2,
            markersize=8)
    ax.plot(dates, data['edge'], label=f'{label_prefix} Edge', color='red', marker='o', linestyle='-', linewidth=2,
            markersize=8)

    ax.set_xlabel('date', fontsize=14, weight='bold')
    ax.set_ylabel(ylabel, fontsize=14, weight='bold')
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend(loc='best', fontsize=12)

    # Format x-axis to show dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.tick_params(axis='x', rotation=45)

    # Tight layout and save chart
    plt.tight_layout()
    plt.savefig(filename, format='png', dpi=300)
    plt.close()


def create_csv(data, dates, filename_csv):
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
    logging.info('Starting to plot traffic data.')

    # Generate dates for x-axis, starting from 2013-07-10
    start_date = datetime.strptime("2013-07-10", "%Y-%m-%d")
    num_iterations = max(len(cloud_incoming), len(cloud_outgoing))
    dates = [start_date + timedelta(days=i) for i in range(num_iterations)]

    # Prepare data for incoming traffic (cloud, fog, edge)
    incoming_data = {
        'cloud': np.array(cloud_incoming),
        'fog': np.mean(np.array(fog_incoming), axis=0),
        'edge': np.mean([np.mean(np.array(traffic), axis=0) for traffic in edge_incoming_map.values()], axis=0)
    }

    # Prepare data for outgoing traffic (cloud, fog, edge)
    outgoing_data = {
        'cloud': np.array(cloud_outgoing),
        'fog': np.mean(np.array(fog_outgoing), axis=0),
        'edge': np.mean([np.mean(np.array(traffic), axis=0) for traffic in edge_outgoing_map.values()], axis=0)
    }

    # Plot incoming traffic separately
    plot_traffic(incoming_data, dates, '/app/images/in_saga.png', 'Incoming', 'kB')

    # Plot outgoing traffic separately
    plot_traffic(outgoing_data, dates, '/app/images/out_saga.png', 'Outgoing', 'kB')

    # Create CSV files for incoming and outgoing traffic
    create_csv(incoming_data, dates, '/app/images/in_saga.csv')
    create_csv(outgoing_data, dates, '/app/images/out_saga.csv')


if __name__ == '__main__':
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

    plot_traffic_data(cloud_incoming_traffic, cloud_outgoing_traffic, fog_incoming_traffic_list,
                      fog_outgoing_traffic_list, edge_incoming_traffic_list_map, edge_outgoing_traffic_list_map)

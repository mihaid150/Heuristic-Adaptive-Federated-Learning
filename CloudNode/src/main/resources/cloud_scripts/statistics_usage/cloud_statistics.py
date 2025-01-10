import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.ticker as ticker
import csv

# Customize font globally for better readability
plt.rcParams.update({
    'font.size': 12,  # Default font size
    'font.family': 'Arial',  # Change font family
    'axes.titlesize': 16,  # Title font size
    'axes.labelsize': 14,  # Axis labels font size
    'xtick.labelsize': 12,  # X-ticks font size
    'ytick.labelsize': 12,  # Y-ticks font size
})


def plot_elapsed_time(elapsed_time, dates):
    """
    Plots elapsed time against dates and saves the plot as a PNG image.
    Also creates a corresponding CSV file containing the data.

    Args:
        elapsed_time (list of float): List of elapsed times in seconds.
        dates (list of str): List of date strings in the format 'YYYY-MM-DD'.

    Returns:
        None
    """
    print(f"Original dates: {dates}")
    date_objects = [datetime.strptime(date.strip('"[]'), '%Y-%m-%d') for date in dates]
    print(f"Processed date objects: {date_objects}")

    date_nums = mdates.date2num(date_objects)

    plt.figure(figsize=(12, 8))
    plt.plot(date_nums, elapsed_time, marker='o', linestyle='-', color='blue', linewidth=2, markersize=8)

    plt.xlabel('Dates', fontsize=14, weight='bold')
    plt.ylabel('Seconds', fontsize=14, weight='bold')
    plt.xticks(date_nums, [date.strftime('%Y-%m-%d') for date in date_objects], rotation=45, ha='right')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()

    plt.savefig('/app/images/time_saga.png', format='png', dpi=300)
    plt.close()

    # Create a CSV table with the data
    create_elapsed_time_csv(elapsed_time, date_objects, '/app/images/time_saga_table.csv')


def create_elapsed_time_csv(elapsed_time, dates, filename_csv):
    """
    Creates a CSV file containing the elapsed time and corresponding dates.

    Args:
        elapsed_time (list of float): List of elapsed times in seconds.
        dates (list of datetime): List of datetime objects representing the dates.
        filename_csv (str): File path to save the CSV.

    Returns:
        None
    """
    # Prepare the table data for CSV
    column_labels = ['Iteration'] + [date.strftime('%Y-%m-%d') for date in dates]
    table_data = ['Seconds'] + elapsed_time

    # Write to a CSV file
    with open(filename_csv, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the column headers (dates)
        writer.writerow(column_labels)

        # Write the elapsed times row
        writer.writerow(table_data)

    print(f"CSV saved as {filename_csv}")


if __name__ == "__main__":
    """
    Main script to process elapsed times and dates, and generate a plot and a CSV.

    Args:
        sys.argv[1] (str): Comma-separated string of elapsed times in seconds.
        sys.argv[2] (str): Comma-separated string of dates in 'YYYY-MM-DD' format.

    Returns:
        None
    """
    elapsed_times_args = sys.argv[1]
    dates_args = sys.argv[2]

    print(f"Elapsed times args: {elapsed_times_args}")
    print(f"Dates args: {dates_args}")

    elapsed_times = list(map(float, elapsed_times_args.split(',')))
    dates_arg = dates_args.split(',')

    plot_elapsed_time(elapsed_times, dates_arg)

import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.ticker as ticker
import csv

# customize font globally for better readability
plt.rcParams.update({
    'font.size': 12,  # default font size
    'font.family': 'Arial',  # change font family
    'axes.titlesize': 16,  # title font size
    'axes.labelsize': 14,  # axis labels font size
    'xtick.labelsize': 12,  # x-ticks font size
    'ytick.labelsize': 12,  # y-ticks font size
})


def plot_elapsed_time(elapsed_time, dates):
    print(f"Original dates: {dates}")
    date_objects = [datetime.strptime(date.strip('"[]'), '%Y-%m-%d') for date in dates]
    print(f"Processed date objects: {date_objects}")

    date_nums = mdates.date2num(date_objects)

    plt.figure(figsize=(12, 8))
    plt.plot(date_nums, elapsed_time, marker='o', linestyle='-', color='blue', linewidth=2, markersize=8)

    # plt.title('Elapsed Time per Iteration', fontsize=18, weight='bold')
    plt.xlabel('dates', fontsize=14, weight='bold')
    plt.ylabel('seconds', fontsize=14, weight='bold')
    plt.xticks(date_nums, [date.strftime('%Y-%m-%d') for date in date_objects], rotation=45, ha='right')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()

    plt.savefig('/app/images/time_saga.png', format='png', dpi=300)
    plt.close()

    # create a CSV table with the data
    create_elapsed_time_csv(elapsed_time, date_objects, '/app/images/time_saga_table.csv')


def create_elapsed_time_csv(elapsed_time, dates, filename_csv):
    # prepare the table data for CSV
    column_labels = ['Iteration'] + [date.strftime('%Y-%m-%d') for date in dates]
    table_data = ['seconds'] + elapsed_time

    # write to a CSV file
    with open(filename_csv, mode='w', newline='') as file:
        writer = csv.writer(file)

        # write the column headers (dates)
        writer.writerow(column_labels)

        # write the elapsed times row
        writer.writerow(table_data)

    print(f"CSV saved as {filename_csv}")


if __name__ == "__main__":
    elapsed_times_args = sys.argv[1]
    dates_args = sys.argv[2]

    print(f"Elapsed times args: {elapsed_times_args}")
    print(f"Dates args: {dates_args}")

    elapsed_times = list(map(float, elapsed_times_args.split(',')))
    dates_arg = dates_args.split(',')

    plot_elapsed_time(elapsed_times, dates_arg)

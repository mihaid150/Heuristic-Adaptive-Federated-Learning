import csv
import json
from datetime import datetime
import sys


def save_to_csv(data, output_file_path):
    """
    Save parsed JSON data to a CSV file.

    Args:
        data (list): List of dictionaries containing data to be saved.
                     Each dictionary must contain the keys: 'mac', 'standard', 'datetime', and 'value'.
        output_file_path (str): Path where the output CSV file will be saved.

    Notes:
        - The function assumes the `datetime` field is in ISO 8601 format (e.g., 'YYYY-MM-DDTHH:MM:SS').
        - If a datetime parsing error occurs, the corresponding row will be skipped.
    """
    headers = ['mac', 'standard', 'datetime', 'value']

    with open(output_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)

        for entry in data:
            try:
                # Adjusting the datetime parsing format to match ISO 8601
                dt = datetime.strptime(entry['datetime'], '%Y-%m-%dT%H:%M:%S')
            except ValueError as e:
                print(f"Error parsing date {entry['datetime']}: {e}")
                continue  # Skip this entry or handle appropriately

            row = [
                entry['mac'],
                entry['standard'],
                dt.strftime('%Y-%m-%d %H:%M:%S'),
                entry['value']
            ]
            writer.writerow(row)


if __name__ == "__main__":
    """
    Main script for converting JSON data to CSV.

    Command-line Arguments:
        1. input_json_path: Path to the input JSON file.
        2. output_csv_path: Path to save the output CSV file.

    Example Usage:
        python script.py input.json output.csv
    """
    input_json_path = sys.argv[1]
    output_csv_path = sys.argv[2]

    # Load the JSON data from the specified file
    with open(input_json_path, 'r') as json_file:
        parsed_data = json.load(json_file)

    # Save the JSON data to CSV format
    save_to_csv(parsed_data, output_csv_path)

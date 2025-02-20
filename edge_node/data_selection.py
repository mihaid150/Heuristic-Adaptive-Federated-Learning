import pandas as pd
from shared.logging_config import logger


def filter_data_by_interval_date(file_path: str, filtering_column_name: str, start_date: str, end_date: str,
                                 output_file_path: str):
    """
    Filters data from a CSV file based on a date range and saves the filtered data to a new CSV file.

    :param file_path: Path to the input CSV file.
    :param filtering_column_name: Column name containing date values for filtering.
    :param start_date: Start date for the filter (inclusive).
    :param end_date: End date for the filter (inclusive).
    :param output_file_path: Path to save the filtered CSV file.
    :return: Path to the saved filtered CSV file.
    """
    chunk_size = 10000
    is_first_chunk = True  # flag to handle headers for the output file

    try:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # convert the filtering column to datetime, ignoring errors
            chunk[filtering_column_name] = pd.to_datetime(chunk[filtering_column_name], errors='coerce')

            # filter the chunk based on the date range
            filtered_chunk = chunk[(chunk[filtering_column_name] >= pd.to_datetime(start_date)) &
                                   (chunk[filtering_column_name] <= pd.to_datetime(end_date))]

            # if the filtered chunk is not empty, append it to the output file
            if not filtered_chunk.empty:
                filtered_chunk.to_csv(output_file_path, mode='a', index=False, header=is_first_chunk)
                is_first_chunk = False  # ensure headers are written only for the first chunk

        return output_file_path

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None


def filter_data_by_day_date(file_path: str, filtering_column_name: str, day_date: str,
                            output_file_path: str):
    """
    Filters data from a CSV file for a specific day and saves the filtered data.
    This function will return only the entries for the given day (e.g., 48 half‑hourly records).

    :param file_path: Path to the input CSV file.
    :param filtering_column_name: Column name containing date values for filtering.
    :param day_date: The specific date for filtering (e.g., "2018-04-21").
    :param output_file_path: Path to save the filtered CSV file.
    :return: Path to the saved filtered CSV file.
    """
    chunk_size = 10000
    is_first_chunk = True  # flag to handle headers for the output file

    try:
        # Parse the provided day_date and compute the end of the day.
        start = pd.to_datetime(day_date)
        end = start + pd.Timedelta(days=1)  # only include the given day

        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # Convert the filtering column to datetime, ignoring errors
            chunk[filtering_column_name] = pd.to_datetime(chunk[filtering_column_name], errors='coerce')

            # Filter the chunk: all rows with a timestamp >= start and < end
            filtered_chunk = chunk[(chunk[filtering_column_name] >= start) &
                                   (chunk[filtering_column_name] < end)]

            # If the filtered chunk is not empty, append it to the output file
            if not filtered_chunk.empty:
                filtered_chunk.to_csv(output_file_path, mode='a', index=False, header=is_first_chunk)
                is_first_chunk = False  # ensure headers are written only once

        return output_file_path

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None

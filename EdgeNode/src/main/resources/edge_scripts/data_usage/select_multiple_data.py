import pandas as pd
import argparse
import logging
import sys


def setup_logging():
    """
    Configures the logging settings for the script.

    Logging is set to INFO level with a standardized message format including timestamps, levels, and messages.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def filter_data(file_path, start_date, end_date):
    """
    Filters rows from a CSV file within a specified date range.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file to be processed.
    start_date : str
        The start date (inclusive) in 'YYYY-MM-DD' format.
    end_date : str
        The end date (inclusive) in 'YYYY-MM-DD' format.

    Returns:
    --------
    None
        Prints the filtered rows in CSV format to the console.

    Raises:
    -------
    pd.errors.ParserError
        If there is an error while parsing the CSV file.
    FileNotFoundError
        If the specified file does not exist.
    Exception
        For any other unexpected errors during processing.
    """
    logging.info(f'Start date: {start_date}, End date: {end_date}')
    chunk_size = 10000
    try:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            logging.info('Processing chunk...')
            chunk['DateTime'] = pd.to_datetime(chunk['DateTime'], errors='coerce')
            filtered_chunk = chunk[(chunk['DateTime'] >= pd.to_datetime(start_date)) &
                                   (chunk['DateTime'] <= pd.to_datetime(end_date))]
            if not filtered_chunk.empty:
                print(filtered_chunk.to_csv(index=False, header=False))
    except pd.errors.ParserError as pe:
        logging.error('Parser error occurred while reading the file: %s', pe)
        raise
    except FileNotFoundError as fnf:
        logging.error('File not found: %s', fnf)
        raise
    except Exception as ex:
        logging.error('An unexpected error occurred during processing: %s', ex)
        raise


def main():
    """
    Main entry point for the script.

    Parses command-line arguments and filters data from the specified CSV file within the given date range.
    Configures logging and handles any unhandled exceptions.

    Command-Line Arguments:
    ------------------------
    file_path : str
        Path to the CSV file.
    start_date : str
        Start date (inclusive) in 'YYYY-MM-DD' format.
    end_date : str
        End date (inclusive) in 'YYYY-MM-DD' format.

    Exits:
    ------
    0 : Successful execution.
    1 : Unhandled exception occurred.
    """
    setup_logging()
    parser = argparse.ArgumentParser(description='Filter CSV data within a time interval')
    parser.add_argument('file_path', type=str, help='Path to the CSV file')
    parser.add_argument('start_date', type=str, help='Start date (inclusive) in YYYY-MM-DD format')
    parser.add_argument('end_date', type=str, help='End date (inclusive) in YYYY-MM-DD format')
    args = parser.parse_args()

    try:
        filter_data(args.file_path, args.start_date, args.end_date)
    except Exception as ex:
        logging.error('Unhandled exception occurred: %s', ex)
        sys.exit(1)


if __name__ == "__main__":
    """
    Entry point for the script execution.

    Calls the main function to process the provided arguments and perform data filtering.
    """
    main()

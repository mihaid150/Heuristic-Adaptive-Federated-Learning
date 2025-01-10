import pandas as pd
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def filter_data(file_path, date, lclid):
    """
    Filters a CSV file to extract rows that match the given date, the next day, and the specified LCLid.

    Parameters:
    -----------
    file_path : str
        Full path to the CSV file to be filtered.
    date : str
        The date in 'YYYY-MM-DD' format to filter by.
    lclid : str
        The LCLid value to filter by.

    Returns:
    --------
    None
        Prints the filtered rows as CSV format to the console.

    Raises:
    -------
    Exception
        If an error occurs while reading or processing the file.
    """
    try:
        chunk_size = 10000
        logging.info(f"Starting to process the file: {file_path}")
        request_date = pd.to_datetime(date)
        next_date = request_date + pd.Timedelta(days=1)

        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            logging.info("Reading a chunk of data...")
            chunk['DateTime'] = pd.to_datetime(chunk['DateTime'], errors='coerce')

            # Filter for both the requested date and the next date
            filtered_chunk = chunk[
                ((chunk['DateTime'].dt.date == request_date.date()) | (chunk['DateTime'].dt.date == next_date.date())) &
                (chunk['LCLid'] == lclid)
                ]

            if not filtered_chunk.empty:
                logging.info(
                    f"Filtered data found for dates: {request_date.date()} and {next_date.date()} and LCLid: {lclid}")
                print(filtered_chunk.to_csv(index=False, header=False))

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.error("Traceback:", exc_info=True)


if __name__ == "__main__":
    """
    Command-line interface for filtering CSV data by date and LCLid.

    Arguments:
    ----------
    file_path : str
        Path to the CSV file to be processed.
    date : str
        Date in 'YYYY-MM-DD' format to filter by.
    lclid : str
        LCLid to filter by.
    """
    parser = argparse.ArgumentParser(description='Filter CSV data by date and LCLid')
    parser.add_argument('file_path', type=str, help='Full path to the CSV file')
    parser.add_argument('date', type=str, help='Date in YYYY-MM-DD format')
    parser.add_argument('lclid', type=str, help='LCLid to filter by')
    args = parser.parse_args()
    logging.info(f"Arguments received - file_path: {args.file_path}, date: {args.date}, lclid: {args.lclid}")
    filter_data(args.file_path, args.date, args.lclid)

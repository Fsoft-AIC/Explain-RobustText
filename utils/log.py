import logging
import os

def setup_logging(log_folder):
    log_file_path = os.path.join(log_folder, 'my_log.log')

    # Configure logging settings
    logging.basicConfig(filename=log_file_path, level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s', force=True)

    # Create a handler to log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Set the desired logging level

    # Create a formatter and attach it to the console handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add the console handler to the root logger
    logging.getLogger().addHandler(console_handler)
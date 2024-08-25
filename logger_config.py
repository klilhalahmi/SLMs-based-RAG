# logger_config.py

import logging

# Configure the logger


def setup_logger(log_file='tinies_vs_giant.log', level=logging.INFO):
    logger = logging.getLogger('tinies_vs_giant')
    logger.setLevel(level)

    # File handler - logs to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Console handler - logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Formatter - specify the format of the logs
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Initialize the logger
logger = setup_logger()

import logging
import os

class CustomLogger:
    def __init__(self, name: str, level: str, log_file: str, log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
        """
        Initializes the logger.
        :param name: Name of the logger.
        :param level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        :param log_file: Path to the log file.
        :param log_format: Format of the log messages (optional, defaults to a standard format).
        """
        self.logger = logging.getLogger(name)
        self.set_level(level)
        
        # Ensure the log directory and file exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist

        # Create a file handler
        file_handler = logging.FileHandler(log_file)
        
        # Set the logging format
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        self.logger.addHandler(file_handler)
    
    def set_level(self, level: str):
        """
        Sets the logging level.
        :param level: Logging level as a string.
        """
        level_dict = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        
        # Set the logger level
        self.logger.setLevel(level_dict.get(level.upper(), logging.INFO))
    
    def get_logger(self):
        """
        Returns the logger instance.
        :return: Logger object.
        """
        return self.logger

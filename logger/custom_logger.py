import logging
import sys


class CustomLogger:

    def __init__(self, name: str):
        """
        Args:
            name:

        Returns: None: Instantiates Logger 
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        stdout_handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(stdout_handler)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s.%(funcName)s - Line %(lineno)d: %(message)s')
        stdout_handler.setFormatter(formatter)

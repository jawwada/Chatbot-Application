
import logging
from logging.handlers import RotatingFileHandler
import os

log_file_path = 'logs/flask-logs.log'
log_level = logging.INFO
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Check if the directory exists, and create it if it doesn't
# Check if the file exists, and create it if it doesn't
if not os.path.isfile(log_file_path):
    # Open the file in write mode, which will create the file if it doesn't exist
    with open(log_file_path, 'w') as f:
        pass  # 'pass' simply allows an empty block; the file is created by opening it

# File handler
file_handler = RotatingFileHandler(log_file_path, maxBytes=10000000, backupCount=5)
file_handler.setLevel(log_level)
file_handler.setFormatter(log_format)


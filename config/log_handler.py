
import logging
from logging.handlers import RotatingFileHandler


log_file_path = 'logs/flask-logs.log'
log_level = logging.INFO
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handler
file_handler = RotatingFileHandler(log_file_path, maxBytes=10000000, backupCount=5)
file_handler.setLevel(log_level)
file_handler.setFormatter(log_format)


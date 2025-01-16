import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir='logs'):
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamp for unique log file name
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # This will also print to console
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def info(self, message):
        """Log info level messages"""
        self.logger.info(message)
    
    def error(self, message):
        """Log error level messages"""
        self.logger.error(message)
    
    def warning(self, message):
        """Log warning level messages"""
        self.logger.warning(message)
    
    def debug(self, message):
        """Log debug level messages"""
        self.logger.debug(message) 
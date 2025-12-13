from .config import get_settings
import logging

def setup_logger(name: str):
    settings = get_settings()
    logger = logging.getLogger(name)
    logger.setLevel(settings.LOG_LEVEL)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
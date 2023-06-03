from pathlib import Path
from config import config
import logging.config


BASE_DIR = Path(__file__).resolve().parent
log_file_path = str(Path(BASE_DIR, 'log/', config.log_file))

LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'simple': {
            'format': '%(asctime)s %(levelname)s %(message)s'
            },
    },
    'handlers': {
        'handler': {
            'level': config.log_level,
            'formatter': 'simple',
            'class': 'logging.handlers.WatchedFileHandler',
            'filename': log_file_path,
        }
    },
    'loggers': {
        'asynclogger': {
            'handlers': ['handler'],
            'level': config.log_level,
            'propagate': True,
        }
    }
}

logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger('asynclogger')

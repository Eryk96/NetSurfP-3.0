import yaml
import logging
import logging.config
from pathlib import Path

from .saving import log_path

LOG_LEVEL = logging.INFO


def setup_logging(run_config: str, log_config: str = "logging.yml"):
    """ Setup ``logging.config``

    Args:
        run_config: path to configuration file for run
        log_config : path to configuration file for logging
    """
    log_config = Path(log_config)

    if not log_config.exists():
        logging.basicConfig(level=LOG_LEVEL)
        logger = logging.getLogger("setup")
        logger.warning(f'"{log_config}" not found. Using basicConfig.')
        return

    with open(log_config, "rt") as f:
        config = yaml.safe_load(f.read())

    # modify logging paths based on run config
    run_path = log_path(run_config)
    for _, handler in config["handlers"].items():
        if "filename" in handler:
            handler["filename"] = str(run_path / handler["filename"])

    logging.config.dictConfig(config)


def setup_logger(name):
    log = logging.getLogger(f'nsp3.{name}')
    log.setLevel(LOG_LEVEL)
    return log

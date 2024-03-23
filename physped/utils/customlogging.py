"""Custom logging module for the project."""

import logging
import os
import time

from IPython.display import HTML, display


def generate_logger(params={}):
    log_level = params.get("level", "INFO")
    display = params.get("display", "screen")
    log_levels = {
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    logging.basicConfig()
    log = logging.getLogger(__name__)
    if log.handlers:
        log.handlers = []

    if display == "screen":
        handler = DisplayHandler()
        handler.setFormatter(HTMLFormatter())
    elif display == "file":
        handler = logging.FileHandler(os.path.join("..", "logs", "log_validation.log"))
        handler.setFormatter(CustomFormatter())
    elif display == "term":
        handler = logging.StreamHandler()
        handler.setFormatter(CustomFormatter())
    else:
        print('Invalid display option. Choose between "screen", "term" and "file".')

    log.addHandler(handler)
    log.setLevel(log_levels[log_level.upper()])
    log.propagate = False


def format_seconds_to_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


class DisplayHandler(logging.Handler):
    def emit(self, record):
        message = self.format(record)
        display(message)


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "[%(asctime)s][%(adjustedTime)s][" "%(levelname)s][%(filename)s:%(lineno)d] - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        record.adjustedTime = format_seconds_to_time(record.relativeCreated / 1000)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class HTMLFormatter(logging.Formatter):
    level_colors = {
        logging.DEBUG: "lightblue",
        logging.INFO: "dodgerblue",
        logging.WARNING: "goldenrod",
        logging.ERROR: "crimson",
        logging.CRITICAL: "firebrick",
    }

    def __init__(self):
        super().__init__(
            '<span style="font-weight: bold; color: green">{asctime}</span> '
            '[<span style="font-weight: bold; color: {levelcolor}">{levelname}</span>] '
            "{message}",
            style="{",
        )

    def format(self, record):
        record.levelcolor = self.level_colors.get(record.levelno, "black")
        return HTML(super().format(record))

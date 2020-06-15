import logging
import sys

# Multiple calls to logging.getLogger('someLogger') return a
# reference to the same logger object.  This is true not only
# within the same module, but also across modules as long as
# it is in the same Python interpreter process.

FORMATTER = logging.Formatter(
)


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_logger():
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s — %(name)s — %(levelname)s —"
                               "%(funcName)s:%(lineno)d — %(message)s")

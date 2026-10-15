"""Quiet imports; optional console progress without creating log files."""
import logging

logger = logging.getLogger('DCLens')
logger.addHandler(logging.NullHandler())


def configure(verbose=False):
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
    logger.propagate = False

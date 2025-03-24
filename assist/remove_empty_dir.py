#!/usr/bin/env python3

import os
import argparse
import sys
import logging
from seqlbtoolkit.io import set_logging

logger = logging.getLogger(__name__)


def remove_empty_folders(directory, excluded_names=None):
    """
    Recursively remove all empty folders under the given directory,
    except those whose path contains any folder in excluded_names.

    :param directory:      The top-level directory to clean up.
    :param excluded_names: A set/list of folder names to exclude if encountered in the path.
                           If a folder's path contains any of these names, skip it.
    """
    if excluded_names is None:
        excluded_names = set()
    else:
        # Convert to a set for faster membership checks, and ensure everything is string
        excluded_names = set(str(name) for name in excluded_names)

    # Traverse from the bottom up so that subdirectories are processed first
    for root, dirs, files in os.walk(directory, topdown=False):
        # Check if this folder path should be excluded
        # We split the path into components and see if there's any overlap with excluded_names
        path_parts = root.split(os.sep)
        if set(path_parts).intersection(excluded_names):
            # If this folder (or any parent folder in its path) is excluded, skip it
            continue

        # If there are no files and no subdirectories, remove the directory
        if not dirs and not files:
            os.rmdir(root)
            logger.info(f"Removed empty folder: {root}")


def main():

    directory = "output"

    # Verify that the directory exists
    if not os.path.isdir(directory):
        logger.info(f"Error: The path '{directory}' does not exist or is not a directory.")
        sys.exit(1)

    exclude = []
    remove_empty_folders(directory, exclude)


if __name__ == "__main__":
    set_logging()
    main()

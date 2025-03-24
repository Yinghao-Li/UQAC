import os
import shutil
import logging
from seqlbtoolkit.io import set_logging

logger = logging.getLogger(__name__)


def sync_folders(
    source_folder,
    destination_folder,
    remove_extras=False,
    include_extensions=None,
    exclude_extensions=None,
    show_progress=False,
):
    """
    Sync files and folders from source_folder to destination_folder.

    :param source_folder:       Path to the source directory.
    :param destination_folder:  Path to the destination directory.
    :param remove_extras:       If True, files/folders in the destination that
                                do not exist in the source will be removed.
    :param include_extensions:  List of file extensions to explicitly include.
                                Example: [".txt", ".py"].
                                If None or empty, include all file types.
    :param exclude_extensions:  List of file extensions to exclude from sync.
                                Example: [".log", ".tmp"].
    :param show_progress:       If True, display a progress indicator in the console.
    """

    # Normalize user input (so we can assume sets later)
    if include_extensions:
        include_extensions = set(ext.lower() for ext in include_extensions)
    if exclude_extensions:
        exclude_extensions = set(ext.lower() for ext in exclude_extensions)

    # Ensure source and destination exist
    if not os.path.isdir(source_folder):
        raise NotADirectoryError(f"Source folder does not exist: {source_folder}")

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder, exist_ok=True)

    # --------------------------------------------------
    # First pass: determine total files to be processed
    # --------------------------------------------------
    total_eligible_files = 0
    for root, dirs, files in os.walk(source_folder):
        for f in files:
            # Check extension to see if it passes inclusion/exclusion filters
            _, ext = os.path.splitext(f)
            ext_lower = ext.lower()

            # If we have an inclusion list, skip files not in it
            if include_extensions and ext_lower not in include_extensions:
                continue

            # If we have an exclusion list, skip those that match
            if exclude_extensions and ext_lower in exclude_extensions:
                continue

            total_eligible_files += 1

    # Keep track of all items found in the source for later removal (if remove_extras=True)
    source_items = set()

    # --------------------------------------------------
    # Second pass: actually copy the files
    # --------------------------------------------------
    processed_files = 0  # For progress tracking

    # Walk the source folder again to copy
    for root, dirs, files in os.walk(source_folder):
        relative_path = os.path.relpath(root, source_folder)
        dest_path = os.path.join(destination_folder, relative_path)

        # Ensure the destination subdirectory exists
        if not os.path.exists(dest_path):
            os.makedirs(dest_path, exist_ok=True)

        # Sync subdirectories
        for d in dirs:
            source_items.add(os.path.join(relative_path, d))
            subdir_source = os.path.join(root, d)
            subdir_dest = os.path.join(dest_path, d)
            if not os.path.exists(subdir_dest):
                os.makedirs(subdir_dest, exist_ok=True)

        # Sync files
        for f in files:
            file_source = os.path.join(root, f)

            # Check extension
            _, ext = os.path.splitext(f)
            ext_lower = ext.lower()

            if include_extensions and ext_lower not in include_extensions:
                continue
            if exclude_extensions and ext_lower in exclude_extensions:
                continue

            # We've decided to copy this file
            source_items.add(os.path.join(relative_path, f))
            file_dest = os.path.join(dest_path, f)

            if not os.path.exists(file_dest) or os.path.getmtime(file_source) > os.path.getmtime(file_dest):
                shutil.copy2(file_source, file_dest)

            # Update progress
            processed_files += 1
            if show_progress:
                logger.info(
                    f"Copying... {processed_files}/{total_eligible_files} "
                    f"({processed_files / total_eligible_files * 100:.2f}%)",
                    end="\r",  # overwrite the line each time
                    flush=True,
                )

    # logger.info a final newline if we were showing progress
    if show_progress and total_eligible_files > 0:
        logger.info()

    # --------------------------------------------------
    # Remove extras if requested
    # --------------------------------------------------
    if remove_extras:
        for root, dirs, files in os.walk(destination_folder):
            relative_path = os.path.relpath(root, destination_folder)
            if relative_path == ".":
                relative_path = ""

            # Check subdirectories
            for d in dirs:
                dest_subdir_path = os.path.join(relative_path, d)
                if dest_subdir_path not in source_items:
                    full_path = os.path.join(root, d)
                    shutil.rmtree(full_path, ignore_errors=True)

            # Check files
            for f in files:
                dest_file_path = os.path.join(relative_path, f)
                if dest_file_path not in source_items:
                    full_path = os.path.join(root, f)
                    os.remove(full_path)


def main():
    source = "/localscratch/yli3100/LLM-Uncertainty/output/math/DeepSeek-R1-Distill-Llama-8B/"
    destination = "/net/csefiles/chaozhanglab/yli3100/LLM-Uncertainty/output/math/DeepSeek-R1-Distill-Llama-8B/"

    # Example usage: only copy .txt and .json, exclude anything .log,
    # and show a progress indicator.
    sync_folders(
        source,
        destination,
        remove_extras=False,
        include_extensions=[".txt", ".json"],
        exclude_extensions=[".log"],
        show_progress=True,
    )


if __name__ == "__main__":
    set_logging()
    main()

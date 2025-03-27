import os


def get_file_paths(base_dir, suffix):
    file_paths = []

    if os.path.exists(base_dir):
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith(suffix):
                    file_paths.append(os.path.join(root, file))
        return file_paths
    else:
        raise FileNotFoundError(f"Directory {base_dir} does not exist.")
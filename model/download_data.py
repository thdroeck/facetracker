import os
import kagglehub
import shutil


def download_and_prepare_data():
    # Download latest version
    path = kagglehub.dataset_download(
        "freak2209/face-data",
    )

    print("Path to dataset files:", path)

    # 1. Define your source path (the one printed by kagglehub)
    # Replace 'path' with the variable from your download code
    src_path = os.path.join(path, "Custom_Data")

    # 2. Define your destination path
    dest_path = "./data"

    # 3. Create the 'Data' folder if it doesn't exist yet
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    # 4. Move the 'images' and 'labels' folders
    folders_to_move = ["images", "labels"]

    for folder in folders_to_move:
        source = os.path.join(src_path, folder)
        destination = os.path.join(dest_path, folder)

        # Check if the folder exists in source before moving
        if os.path.exists(source):
            # Move the folder
            shutil.move(source, destination)
            print(f"Successfully moved {folder} to {dest_path}")
        else:
            print(f"Error: {folder} not found in {src_path}")

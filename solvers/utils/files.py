import os


def get_exports_folder(*sub_folders) -> str:
    sub_folders_strings = (str(folder_name) for folder_name in sub_folders)
    folder_path = os.path.join("exports", *sub_folders_strings)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    return folder_path

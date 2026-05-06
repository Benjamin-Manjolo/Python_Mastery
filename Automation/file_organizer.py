import os
import shutil

# 🎯 Folder you want to organize
FOLDER = "Downloads"

# 📁 File categories
FILE_TYPES = {
    "Images": [".jpg", ".jpeg", ".png", ".gif"],
    "Videos": [".mp4", ".mkv", ".mov"],
    "Documents": [".pdf", ".docx", ".txt", ".xlsx"],
    "Music": [".mp3", ".wav"],
}

def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_category(file_name):
    _, ext = os.path.splitext(file_name)

    for category, extensions in FILE_TYPES.items():
        if ext.lower() in extensions:
            return category

    return "Others"

def organize_files():
    if not os.path.exists(FOLDER):
        print("Folder does not exist!")
        return

    for file in os.listdir(FOLDER):
        file_path = os.path.join(FOLDER, file)

        # skip folders
        if os.path.isdir(file_path):
            continue

        category = get_category(file)

        target_folder = os.path.join(FOLDER, category)
        create_folder_if_not_exists(target_folder)

        shutil.move(file_path, os.path.join(target_folder, file))
        print(f"Moved: {file} → {category}/")

if __name__ == "__main__":
    organize_files()
    print("Done organizing 🎉")
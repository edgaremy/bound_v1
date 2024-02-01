import os

def count_lines_in_txt_files(folder_path):
    total_lines = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    total_lines += len(f.readlines())

    return total_lines

# Example usage:
folder_path = "/mnt/disk1/datasets/Lepinoc_2022/splitted_dataset/Task_Lepinoc/labels"
total_lines = count_lines_in_txt_files(folder_path)
print(f"Total number of lines in txt files: {total_lines}")

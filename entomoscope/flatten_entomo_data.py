import os
import sys
import shutil

def flatten_directory(directory):
    """
    Flattens the structure inside each subfolder of the given directory.
    All files from nested directories are moved to the root of each subfolder.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)
    
    print(f"Flattening directory structure in: {directory}")
    
    # Get all immediate subdirectories
    root_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
    for root_dir in root_dirs:
        root_path = os.path.join(directory, root_dir)
        print(f"Processing: {root_path}")
        
        # Collect all files from subdirectories
        files_to_move = []
        for root, dirs, files in os.walk(root_path):
            # Skip the top-level directory
            if root == root_path:
                continue
                
            for file in files:
                source = os.path.join(root, file)
                # Check for duplicate filenames
                target = os.path.join(root_path, file)
                counter = 1
                base_name, ext = os.path.splitext(file)
                while os.path.exists(target):
                    new_name = f"{base_name}_{counter}{ext}"
                    target = os.path.join(root_path, new_name)
                    counter += 1
                
                files_to_move.append((source, target))
        
        # Move all files
        for source, target in files_to_move:
            print(f"  Moving {source} → {target}")
            shutil.move(source, target)
        
        # Remove empty directories
        for root, dirs, files in os.walk(root_path, topdown=False):
            if root == root_path:
                continue
            try:
                if not os.listdir(root):
                    print(f"  Removing empty directory: {root}")
                    os.rmdir(root)
            except Exception as e:
                print(f"  Error removing directory {root}: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python flatten_entomo_data.py <directory_path>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    flatten_directory(input_dir)
    print("Flattening complete!")
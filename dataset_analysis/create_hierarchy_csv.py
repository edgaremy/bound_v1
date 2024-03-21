import os
import csv
import get_hierarchy as hierarchy

def create_hierarchy_csv(directory_path, output_csv):
    hierarchy_data = []
    
    for folder_name in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, folder_name)
        
        if os.path.isdir(folder_path):
            hierarchy_list = hierarchy.get_hierarchy_from_name(folder_name)
            
            if hierarchy_list:
                hierarchy_data.append(hierarchy_list)
    
    # Sort the hierarchy_data list by the specie name (order of rows needs to match order of classes in the model)
    hierarchy_data.sort(key=lambda x: x[4])
    
    # Check if the output_csv file exists
    file_exists = os.path.isfile(output_csv)
    
    with open(output_csv, 'a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header if the file doesn't exist
        if not file_exists:
            writer.writerow(['class', 'order', 'family', 'genus', 'specie'])
        
        writer.writerows(hierarchy_data)

# Example usage:
create_hierarchy_csv('/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/train', '306_hierarchy.csv')

import os

def generate_class_to_idx_mapping(directory):
    classes = sorted(os.listdir(directory))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    return class_to_idx

# def write_mapping_to_file(mapping, output_file):
#     with open(output_file, 'w') as file:
#         for cls, idx in mapping.items():
#             file.write(f'{cls}: {idx}\n')

def write_mapping_to_file(mapping, output_file):
    with open(output_file, 'w') as file:
        # file.write("{")
        for cls, idx in mapping.items():
            file.write(f"{cls}\n")#: {idx},\n")
        # file.write("}")

# Example usage
directory = '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/train/'
output_file = './torch_benchmark/class_to_idx_mapping.txt'

mapping = generate_class_to_idx_mapping(directory)
write_mapping_to_file(mapping, output_file)
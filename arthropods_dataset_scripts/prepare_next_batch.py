import os
import shutil
from convert_json_to_dataset import convert_json_to_labels
from convert_labels_to_csv import convert_labels_to_csv
from get_data_from_inat import get_images_from_inat
from limit_one_line_of_csv_v2 import keep_each_element_number_n_with_different_observation
from merge_datasets import create_yaml, merge_datasets
from split_dataset_detect import split_dataset
from yolov8_predict import find_last_predict_number, predict_images_with_yolov8
from yolov8_train import find_last_model_number, train_yolov8


## MAIN SCRIPT TO CALL OTHER FUNCTIONS IN THE SAME FOLDER ##

# Current Dataset Number (where the new Annotation JSON-files are):
DATASET_NUMBER = 19

# Convert JSON to Labels:
print("\n##### CONVERTING JSON TO LABELS #####\n")
json_folder = '/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT' + str(DATASET_NUMBER) + '/dataset(prediction)/json_validated'
labels_output_folder = '/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT' + str(DATASET_NUMBER) + '/dataset(prediction)/labels_validated'
convert_json_to_labels(json_folder, labels_output_folder)

# Split Newly Annotated Dataset:
print("\n##### SPLITTING NEWLY CREATED DATASET #####\n")
labels_input_folder = '/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT' + str(DATASET_NUMBER) + '/dataset(prediction)/labels_validated'
labels_output_folder = '/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT' + str(DATASET_NUMBER) + '/dataset/labels'
image_input_folder = '/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT' + str(DATASET_NUMBER) + '/Pictures'
image_output_folder = '/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT' + str(DATASET_NUMBER) + '/dataset/images'
split_dataset(labels_input_folder, labels_output_folder, image_input_folder, image_output_folder, train_ratio = 0.9, val_ratio = 0.1)

# Add Previously Annotated Dataset to the New Dataset:
print("\n##### MERGING OLD DATASET WITH NEW DATASET #####\n")
source_dataset = '/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT' + str(DATASET_NUMBER - 1) + '/dataset'
target_dataset = '/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT' + str(DATASET_NUMBER) + '/dataset'
merge_datasets(source_dataset, target_dataset)

# Create YAML File:
print("\n##### CREATING DATASET YAML FILE #####\n")
yaml_path = '/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT' + str(DATASET_NUMBER) + '/dataset/Arthropods_wave' + str(DATASET_NUMBER) + '.yaml'
create_yaml(yaml_path)

# Get List of Images from NEXT observations and Download them from iNaturalist:
print("\n##### GETTING & DOWNLOADING LIST OF IMAGES FOR NEXT ANNOTATIONS (ASYNC) #####\n")
input_file = 'requested_CSVs/photos_to_scrap.csv'
input_file_2 = 'requested_CSVs/french_arthro_observations_list.csv'
family_file = 'requested_CSVs/biggest_french_member_by_obs.csv'
file_list = 'requested_CSVs/photos_to_scrap_NUMBER' + str(DATASET_NUMBER + 1) + '.csv'
dest_folder = "/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT" + str(DATASET_NUMBER + 1) + "/"
keep_each_element_number_n_with_different_observation(input_file, input_file_2, family_file, file_list, DATASET_NUMBER + 1)
get_images_from_inat(file_list, dest_folder)

# Train YOLOv8 on the New Complete Dataset:
print("\n##### MEANWHILE, TRAINING YOLOv8 ON THE NEW DATASET #####\n")
train_yolov8(yaml_path, epoch=100)

# Predict Images with YOLOv8:
print("\n##### PREDICT ON DOWNLOADED IMAGES WITH NEW TRAINED MODEL #####\n")
print("Model folder used: train" + str(find_last_model_number()))
model_path = "runs/detect/train" + str(find_last_model_number()) + "/weights/best.pt" # load best.pt or last.pt of local model
folder_path = "/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT" + str(DATASET_NUMBER + 1) + "/Pictures"
predict_images_with_yolov8(model_path, folder_path)
print("Prediction done.")

# Copy predicted labels to the new dataset:
print("\n##### COPYING PREDICTED LABELS TO THE NEW DATASET #####\n")
print("Predict folder used: predict" + str(find_last_predict_number()))
labels_input_folder = "/home/eremy/Documents/CODE/bound_v1/runs/detect/predict" + str(find_last_predict_number()) + "/labels"
labels_output_folder = "/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT" + str(DATASET_NUMBER + 1) + "/dataset(prediction)/labels"
os.makedirs(labels_output_folder, exist_ok=True)
for root, dirs, files in os.walk(labels_input_folder):
    for file in files:
        file_path = os.path.join(root, file)
        shutil.copy(file_path, labels_output_folder)
print("Copy done.")

# Convert labels to CSV:
print("\n##### CONVERTING LABELS TO CSV #####\n")
images_path = "/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT" + str(DATASET_NUMBER + 1) + "/Pictures"
labels_folder = "/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT" + str(DATASET_NUMBER + 1) + "/dataset(prediction)/labels"
output_csv = "/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT" + str(DATASET_NUMBER + 1) + "/dataset(prediction)/labels.csv"
convert_labels_to_csv(images_path, labels_folder, output_csv)
print("Conversion done.")

# Compress the batch to annotate:
print("\n##### COMPRESSING BATCH TO ANNOTATE #####\n")
# Copy Pictures file into a temporary images folder with labels.csv:
archive_folder = "/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT" + str(DATASET_NUMBER + 1) + "/dataset(prediction)/Arthropods_wave" + str(DATASET_NUMBER + 1)
os.makedirs(archive_folder, exist_ok=True)
source1 = "/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT" + str(DATASET_NUMBER + 1) + "/Pictures"
source2 = "/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT" + str(DATASET_NUMBER + 1) + "/dataset(prediction)/labels.csv"
shutil.copytree(source1, archive_folder + "/images")
shutil.copy(source2, archive_folder + "/labels.csv")
# Compress the folder:
shutil.make_archive(archive_folder, 'zip', archive_folder)
# Delete the temporary folder:
shutil.rmtree(archive_folder)
print("Compression done.")

# Create JSON directory for next annotations:
os.makedirs('/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT' + str(DATASET_NUMBER + 1) + '/dataset(prediction)/json_validated', exist_ok=True)

print("\n##### NEXT BATCH IS READY #####\n")
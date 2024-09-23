import glob
import fiftyone as fo

images_patt = "/path/to/images/*"

# Ex: your custom label format
annotations = {
    "/path/to/images/000001.jpg": [
        {"bbox": ..., "label": ...},
        ...
    ],
    ...
}

# Create dataset
dataset = fo.Dataset(name="my-detection-dataset")

# Persist the dataset on disk in order to 
# be able to load it in one line in the future
dataset.persistent = True

# Add your samples to the dataset
for filepath in glob.glob(images_patt):
    sample = fo.Sample(filepath=filepath)

    # Convert detections to FiftyOne format
    detections = []
    for obj in annotations[filepath]:
        label = obj["label"]

        # Bounding box coordinates should be relative values
        # in [0, 1] in the following format:
        # [top-left-x, top-left-y, width, height]
        bounding_box = obj["bbox"]

        detections.append(
            fo.Detection(label=label, bounding_box=bounding_box)
        )

    # Store detections in a field name of your choice
    sample["ground_truth"] = fo.Detections(detections=detections)

    dataset.add_sample(sample)


export_dir = "/path/for/coco-detection-dataset"
label_field = "ground_truth"  # for example

# Export the dataset
dataset.export(
    export_dir=export_dir,
    dataset_type=fo.types.COCODetectionDataset,
    label_field=label_field,
)

# Convert a COCO detection dataset to CVAT image format
fiftyone convert \
    --input-dir /mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT1/dataset \
    --input-type fiftyone.types.YOLOv5Dataset \
    --output-dir /mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT1/tmp \
    --output-type fiftyone.types.COCODetectionDataset
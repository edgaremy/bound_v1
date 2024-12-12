import os
import shutil
import math

def create_final_split(input_data_dir, output_dataset_dir):
    # Make output directories
    os.makedirs(os.path.join(output_dataset_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dataset_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dataset_dir, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_dataset_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dataset_dir, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dataset_dir, 'labels', 'test'), exist_ok=True)

    images_dir = os.path.join(input_data_dir, 'images')
    labels_dir = os.path.join(input_data_dir, 'labels')

    images = os.listdir(images_dir)
    labels = os.listdir(labels_dir)

     # only keep images with associated label
    images = [image for image in images if image.split('.')[0] + '.txt' in labels]

    # get list of species:
    species = [image.split('_')[0] for image in images]

    # # Print some stats:
    # # number of species with 22 images
    # print('Number of species with 22 images:', len(list(set([specie for specie in species if species.count(specie) == 22]))))
    # # number of species with 10-21 images
    # print('Number of species with 10-21 images:', len(list(set([specie for specie in species if 10 <= species.count(specie) <= 21]))))
    # # number of species with 5-9 images
    # print('Number of species with 5-9 images:', len(list(set([specie for specie in species if 5 <= species.count(specie) <= 9]))))
    # # number of species with 2-4 images
    # print('Number of species with 2-4 images:', len(list(set([specie for specie in species if 2 <= species.count(specie) <= 4]))))
    # # number of species with 1 image
    # print('Number of species with 1 image:', len(list(set([specie for specie in species if species.count(specie) == 1]))))
    # return

    species = list(set(species)) # remove duplicates
    # split each specie into train, val, test
    for specie in species:
        # get images for the current specie (only images with associated label)
        images_specie = [image for image in images if image.startswith(specie + '_')]
        # get number of images for the current specie
        number_images = len(images_specie)

        if number_images < 2:
            # print(f'Specie {specie} does not have enough images, only {number_images}')
            shutil.copy(os.path.join(images_dir, images_specie[0]), os.path.join(output_dataset_dir, 'images', 'train', images_specie[0]))
        else:
            test_nb = math.ceil(0.10 * number_images)
            if number_images == 2: #or number_images == 3:
                val_nb = 0
            else:
                val_nb = math.ceil(0.10*number_images)
            train_nb = number_images - test_nb - val_nb

            print(specie, number_images, train_nb, val_nb, test_nb)

            # copy images to the corresponding split
            for i, image in enumerate(images_specie):
                if i < train_nb:
                    shutil.copy(os.path.join(images_dir, image), os.path.join(output_dataset_dir, 'images', 'train', image))
                    shutil.copy(os.path.join(labels_dir, image.split('.')[0] + '.txt'), os.path.join(output_dataset_dir, 'labels', 'train', image.split('.')[0] + '.txt'))
                elif i < train_nb + val_nb:
                    shutil.copy(os.path.join(images_dir, image), os.path.join(output_dataset_dir, 'images', 'val', image))
                    shutil.copy(os.path.join(labels_dir, image.split('.')[0] + '.txt'), os.path.join(output_dataset_dir, 'labels', 'val', image.split('.')[0] + '.txt'))
                else:
                    shutil.copy(os.path.join(images_dir, image), os.path.join(output_dataset_dir, 'images', 'test', image))
                    shutil.copy(os.path.join(labels_dir, image.split('.')[0] + '.txt'), os.path.join(output_dataset_dir, 'labels', 'test', image.split('.')[0] + '.txt'))
# Usage example
input_data_dir = '/mnt/disk1/datasets/iNaturalist/Arthropods/ALL_IMAGES'
output_dataset_dir = '/mnt/disk1/datasets/iNaturalist/Arthropods/FINAL_DATASET'
create_final_split(input_data_dir, output_dataset_dir)
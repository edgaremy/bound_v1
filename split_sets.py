import shutil
import os
import random# get image names and set train/test split ratio
import glob
import shutil
from PIL import Image, ExifTags
from datetime import datetime

ratio_train = 0.8
ratio_val = 0.1
# -> the remaining part is for test

src = '/home/edgarremy/Documents/CODE/bound_v1/coco_converted_tiled/'
dst = '/home/edgarremy/Documents/CODE/bound_v1/splitted_dataset/Task_Lepinoc/'

imnames = glob.glob(f'{src}images/*.jpg')

def random_split():
    # split dataset for train, val and test
    train = []
    val = []
    test = []
    for imname in imnames:
        name = imname.split('/')[-1]
        label_dir = imname.replace('.jpg', '.txt').replace('/images/', '/labels/default/')
        dice = random.random()
        if dice > ratio_train + ratio_val:
            test.append(f'{dst}images/test/{name}')
            shutil.copy(imname, f'{dst}images/test/{name}')
            shutil.copy(label_dir, f'{dst}labels/test/{name.replace(".jpg", ".txt")}')
        elif dice > ratio_train:
            val.append(f'{dst}images/val/{name}')
            shutil.copy(imname, f'{dst}images/val/{name}')
            shutil.copy(label_dir, f'{dst}labels/val/{name.replace(".jpg", ".txt")}')
        else:
            train.append(f'{dst}/images/train/{name}')
            shutil.copy(imname, f'{dst}images/train/{name}')
            shutil.copy(label_dir, f'{dst}labels/train/{name.replace(".jpg", ".txt")}')
        
    print('train:', len(train))
    print('val:', len(val))
    print('test:', len(test))


def time_split():
    # find order of all images by creation date in metadata
    im_dates = []
    for imname in imnames:
        image_exif = Image.open(imname)._getexif()
        if image_exif:
            # Make a map with tag names
            exif = { ExifTags.TAGS[k]: v for k, v in image_exif.items() if k in ExifTags.TAGS and type(v) is not bytes }
            # Grab the date
            exif_date = exif['DateTimeOriginal']
            date_obj = datetime.strptime(exif_date, '%Y:%m:%d %H:%M:%S')
            #print(date_obj)
        else:
            print('ERROR: Unable to get date from exif for %s' % imname)
        im_dates.append((date_obj, imname))
    im_dates = sorted(im_dates)
    imnames_sorted = list(map(list, zip(*im_dates)))[1]
    
    # split dataset for train, val and test
    train = []
    val = []
    test = []
    nb_imgs = len(imnames_sorted)
    for i, imname in enumerate(imnames_sorted):
        name = imname.split('/')[-1]
        label_dir = imname.replace('.jpg', '.txt').replace('/images/', '/labels/default/')
        if i < nb_imgs * ratio_train: # the 80% (if ratio_train=0.8) earliest pictures for train
            train.append(f'{dst}images/train/{name}')
            shutil.copy(imname, f'{dst}images/train/{name}')
            shutil.copy(label_dir, f'{dst}labels/train/{name.replace(".jpg", ".txt")}')
        elif i < nb_imgs * (ratio_train + ratio_val):
            val.append(f'{dst}images/val/{name}')
            shutil.copy(imname, f'{dst}images/val/{name}')
            shutil.copy(label_dir, f'{dst}labels/val/{name.replace(".jpg", ".txt")}')
        else:
            test.append(f'{dst}images/test/{name}')
            shutil.copy(imname, f'{dst}images/test/{name}')
            shutil.copy(label_dir, f'{dst}labels/test/{name.replace(".jpg", ".txt")}')
    
    print('train:', len(train))
    print('val:', len(val))
    print('test:', len(test))



#random_split()
time_split()



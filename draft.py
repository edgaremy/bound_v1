import os
import piexif
from datetime import datetime
from PIL import Image, ExifTags

path = './c77ec1edeaf25c725ca0693b7568ee0b.jpg'
path2 = '(copie).jpg'

time = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%d-%m-%Y")
print(time)


im = Image.open(path)
exif = im.getexif()
creation_time = exif.get(36867)
print(creation_time)


def getCreationDate(filename):
    image_exif = Image.open(filename)._getexif()
    if image_exif:
        # Make a map with tag names
        exif = { ExifTags.TAGS[k]: v for k, v in image_exif.items() if k in ExifTags.TAGS and type(v) is not bytes }
        #print(json.dumps(exif, indent=4))
        # Grab the date
        exif_date = exif['DateTimeOriginal']
        date_obj = datetime.strptime(exif_date, '%Y:%m:%d %H:%M:%S')
        print(date_obj)
    else:
        print('Unable to get date from exif for %s' % filename)
    return exif_date


def editCreationDate(filename_src, filename_dst):

    img = Image.open(filename_src)
    img2 = Image.open(filename_dst)
    # exif_dict = piexif.load(img.info['exif'])
    exif_dict = img.info['exif']
    # exif_dict['DateTimeOriginal'] = exif_date
    # exif_bytes = piexif.dump(exif_dict)
    img2.save(filename_dst, exif=exif_dict)


creation_date = getCreationDate(path)
editCreationDate(path2, path2)
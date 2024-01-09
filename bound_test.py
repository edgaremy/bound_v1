import os

from collections import defaultdict
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class COCOParser:
    def __init__(self, anns_file, imgs_dir):
        with open(anns_file, 'r') as f:
            coco = json.load(f)
            
        self.annIm_dict = defaultdict(list)        
        self.cat_dict = {} 
        self.annId_dict = {}
        self.im_dict = {}
        self.licenses_dict = {}
        for ann in coco['annotations']:           
            self.annIm_dict[ann['image_id']].append(ann) 
            self.annId_dict[ann['id']]=ann
        for img in coco['images']:
            self.im_dict[img['id']] = img
        for cat in coco['categories']:
            self.cat_dict[cat['id']] = cat
        for license in coco['licenses']:
            self.licenses_dict[license['id']] = license
    def get_imgIds(self):
        return list(self.im_dict.keys())
    def get_annIds(self, im_ids):
        im_ids=im_ids if isinstance(im_ids, list) else [im_ids]
        return [ann['id'] for im_id in im_ids for ann in self.annIm_dict[im_id]]
    def load_anns(self, ann_ids):
        im_ids=ann_ids if isinstance(ann_ids, list) else [ann_ids]
        return [self.annId_dict[ann_id] for ann_id in ann_ids]        
    def load_cats(self, class_ids):
        class_ids=class_ids if isinstance(class_ids, list) else [class_ids]
        return [self.cat_dict[class_id] for class_id in class_ids]
    def get_imgLicenses(self,im_ids):
        im_ids=im_ids if isinstance(im_ids, list) else [im_ids]
        lic_ids = [self.im_dict[im_id]["license"] for im_id in im_ids]
        return [self.licenses_dict[lic_id] for lic_id in lic_ids]


coco_annotations_file="/home/edgarremy/Documents/task_lépinoc 2022 - ordre-2023_07_18_09_04_45-coco 1.0/annotations/instances_default.json"

coco_images_dir="/home/edgarremy/Documents/task_lépinoc 2022 - ordre-2023_07_18_09_04_45-coco 1.0/images"
coco = COCOParser(coco_annotations_file, coco_images_dir)


def get_bounding_boxes(x, y, w, h, cell_size):
    cell_idx_x, cell_idx_y = x // cell_size, y // cell_size
    x_rel, y_rel = x % cell_size, y % cell_size

    case = 'normal'

    # determine case:
    if x_rel+w > cell_size:
        if y_rel+h > cell_size: # spills out to cells under, right and under+right
            case = 'right_and_under'
        else: # spills out to the cell on the right
            case = 'right'
    elif y_rel+h > cell_size: # spills out to the cell under
        case = 'under'
        

    # applying case:
    if case == 'normal':
        cells2D_idx = [(cell_idx_x, cell_idx_y)]
        X = [x_rel]
        Y = [y_rel]
        W = [w]
        H = [h]
    elif case == 'right':
        w_cell1 = cell_size - x_rel
        cells2D_idx = [(cell_idx_x, cell_idx_y), (cell_idx_x+1, cell_idx_y)]
        X = [x_rel, 0]
        Y = [y_rel, y_rel]
        W = [w_cell1, w - w_cell1]
        H = [h, h]
    elif case == 'under':
        h_cell1 = cell_size - y_rel
        cells2D_idx = [(cell_idx_x, cell_idx_y), (cell_idx_x, cell_idx_y+1)]
        X = [x_rel, x_rel]
        Y = [y_rel, 0]
        W = [w, w]
        H = [h_cell1, h - h_cell1]
    elif case == 'right_and_under':
        w_cell1 = cell_size - x_rel
        h_cell1 = cell_size - y_rel
        cells2D_idx = [(cell_idx_x, cell_idx_y), (cell_idx_x+1, cell_idx_y), (cell_idx_x, cell_idx_y+1), (cell_idx_x+1, cell_idx_y+1)]
        X = [x_rel, 0, x_rel, 0]
        Y = [y_rel, y_rel, 0, 0]
        W = [w_cell1, w - w_cell1, w_cell1, w - w_cell1]
        H = [h_cell1, h_cell1, h - h_cell1, h - h_cell1]

    # removing too small bounding boxes:
    for i in range(len(W) - 1, -1, -1):
        if W[i] < 15 or H[i] < 15 or w/W[i] > 5 or h/H[i] > 5:
            del cells2D_idx[i]
            del X[i]
            del Y[i]
            del W[i]
            del H[i]

    return X, Y, W, H, cells2D_idx

def coco_to_yolo_bbox(x, y, w, h, cell_size):
    # all coordinates are normalize with the size of the cell
    center_x = (x + w) / 2 / cell_size
    center_y = (y + h) / 2 / cell_size
    norm_w = w / cell_size
    norm_h = h / cell_size
    return [center_x, center_y, norm_w, norm_h]

def add_annotation_txt(filename, class_id, yolo_bbox):
    line = str(class_id) + " " + str(yolo_bbox[0]) + " " + str(yolo_bbox[1]) + " " + str(yolo_bbox[2]) + " " + str(yolo_bbox[3])
    if os.path.exists(filename):
        with open(filename, 'a') as file:
            file.write("\n" + line)
    else:
        with open(filename, 'w') as file:
            file.write(line)



# format input data
color_list = ["pink", "red", "teal", "blue", "orange", "yellow", "black", "magenta","green","aqua"]*10
num_imgs_to_disp = 1
cell_size = 640
nb_cell_h = 4096 // cell_size
nb_cell_w = 3072 // cell_size
total_images = len(coco.get_imgIds()) # total number of images
sel_im_idxs = np.random.permutation(total_images)[:num_imgs_to_disp]
img_ids = coco.get_imgIds()
# selected_img_ids = [img_ids[i] for i in sel_im_idxs] # load random img
selected_img_ids = [img_ids[461]] # load specific img
ann_ids = coco.get_annIds(selected_img_ids)
im_licenses = coco.get_imgLicenses(selected_img_ids)
fig, ax = plt.subplots(nrows=nb_cell_h, ncols=nb_cell_w, figsize=(9,12.4),)
ax = ax.ravel()

for i, im in enumerate(selected_img_ids):
    print(coco.im_dict[im]['file_name']) # get the file_name from image id im
    image = Image.open(f"{coco_images_dir}/{coco.im_dict[im]['file_name']}")
    np_img = np.array(image)
    ann_ids = coco.get_annIds(im)
    annotations = coco.load_anns(ann_ids)

    for ann in annotations:
        # print(ann['attributes']['sous-embranchement']) # usually = 'hexapoda'
        bbox = ann['bbox']
        x, y, w, h = [int(b) for b in bbox]

        X, Y, W, H, cells2D_idx = get_bounding_boxes(x, y, w, h, cell_size) # get edge case bboxes as well

        for j in range(len(cells2D_idx)):

            grid_index = cells2D_idx[j][1] * nb_cell_w + cells2D_idx[j][0] # absolute 1D idx for subplot

            if grid_index < nb_cell_h * nb_cell_w: # add annotation to correct cell only if its inside the squared grid
                
                # save bbox annotations:
                # annot_dir="/home/edgarremy/Documents/Yaml_task_lépinoc 2022 - ordre-2023_07_18_09_04_45-coco 1.0/images"
                # add_annotation_txt("/home/edgarremy/Documents/task_lépinoc 2022 - ordre-2023_07_18_09_04_45-coco 1.0/images")

                # display bbox:
                class_id = ann["category_id"]
                class_name = coco.load_cats(class_id)[0]["name"]
                color_ = color_list[class_id]
                rect = plt.Rectangle((X[j], Y[j]), W[j], H[j], linewidth=2, edgecolor=color_, facecolor='none')
                t_box=ax[grid_index].text(X[j], Y[j], class_name,  color='red', fontsize=10)
                t_box.set_bbox(dict(boxstyle='square, pad=0',facecolor='white', alpha=0.6, edgecolor='blue'))
                ax[grid_index].add_patch(rect)
        
    # display each squared image in dedicated cell:
    for ix in range(nb_cell_h):
        for iy in range(nb_cell_w):
            ax[ix * nb_cell_w + iy].axis('off')
            ax[ix * nb_cell_w + iy].imshow(np_img[(ix*cell_size):(ix*cell_size+cell_size),(iy*cell_size):(iy*cell_size+cell_size),:])
plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.show()


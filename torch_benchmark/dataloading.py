import os
import glob
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2
import torch
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, class_folders_dir, transform=None, target_transform=None):
        self.img_dir = glob.glob(os.path.join(class_folders_dir, "*/*"))
        labels = [img.split("/")[-2] for img in self.img_dir]
        self.class_names = sorted(set(labels))
        self.nb_classes = len(self.class_names)
        self.img_labels = [self.class_names.index(l) for l in labels]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_dir[idx]
        # image = read_image(img_path)
        # open image as PIL image
        image = Image.open(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # make label hot one vector
        label = torch.nn.functional.one_hot(torch.as_tensor(label), num_classes=self.nb_classes)
        return image, label.float()


def create_augmentations(IMG_SIZE):
    augmentations = v2.Compose([
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.RandomRotation(degrees=10),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomAffine(degrees=(-10,10), translate=(0, 0), scale=(0.9, 1.1)),
        v2.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[0], std=[255]),
        # v2.ToTensor()
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    normalize = v2.Compose([
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[0], std=[255]),
    ])
    return augmentations, normalize
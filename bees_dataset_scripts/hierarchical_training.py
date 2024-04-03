import torch
import os
import math
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torchvision.io import read_image
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu
import keras
keras.backend.set_image_data_format("channels_first")
torch.cuda.empty_cache()

from keras import layers
from keras import regularizers
from keras import optimizers
from numpy.ma.core import transpose
from keras import backend as K

if torch.cuda.is_available():
    # Proceed with CUDA operations
    print("available CUDA devices: ", torch.cuda.device_count())
else:
    print("CUDA not available. Using CPU...")


#### PARAMETRES ####

IMG_SIZE = 320 # pour utiliser ResNet
train_dir = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/train/"
val_dir = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/val/"


#### DATASET ####

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
        image = read_image(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # make label hot one vector
        label = torch.nn.functional.one_hot(torch.as_tensor(label), num_classes=self.nb_classes)
        return image, label

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

training_data = CustomImageDataset(train_dir, augmentations)#, augmentations)
train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
class_names = training_data.class_names
nb_classes = len(class_names)

validation_data = CustomImageDataset(val_dir, normalize)
val_dataloader = DataLoader(validation_data, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

# Display image and label:
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0].numpy().argmax()
# # plt.imshow(img, cmap="gray")
# plt.imshow(img.permute(1, 2, 0), cmap="gray")
# plt.show()
# print(f"Label: {training_data.class_names[label]}")


#### MODEL ####

conv_base = keras.applications.resnet50.ResNet50(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(3, IMG_SIZE, IMG_SIZE),
    pooling=None,
    classes=nb_classes
)

# conv_base = K.keras.applications.resnet.ResNet101(
#     include_top=False,
#     weights='imagenet',
#     input_tensor=None,
#     input_shape=(IMG_SIZE, IMG_SIZE, 3),
#     pooling=None,
#     classes=nb_classes,
# )

# conv_base = K.keras.applications.EfficientNetB0(
#     include_top=False,
#     weights="imagenet",
#     input_tensor=None,
#     input_shape=(IMG_SIZE, IMG_SIZE, 3),
#     pooling=None,
#     classes=nb_classes
# )

# conv_base = tf.keras.applications.mobilenet.MobileNet(
#     input_shape=(IMG_SIZE, IMG_SIZE, 3),
#     include_top=False,
#     weights='imagenet',
#     input_tensor=None,
#     pooling=None,
#     classes=nb_classes
# )

inputs = keras.Input(shape=(3, IMG_SIZE, IMG_SIZE))
x = conv_base(inputs)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(nb_classes, kernel_regularizer=regularizers.L2(1e-4), activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs, name="bee_classifier")

model.summary()


#### HIERARCHY STUFF ####

hierarchy_csv = "/home/eremy/Documents/CODE/bound_v1/dataset_analysis/306_hierarchy.csv"
hierarchy = pd.read_csv(hierarchy_csv)

specie = list(hierarchy["specie"].unique())
nb_specie = len(specie)

genus = list(hierarchy["genus"].unique())
nb_genus = len(genus)

family = list(hierarchy["family"].unique())
nb_family = len(family)

order = list(hierarchy["order"].unique())
nb_order = len(order)

class_ = list(hierarchy["class"].unique())
nb_class = len(class_)

species_to_genus = np.zeros((nb_genus, nb_specie))
genus_to_family = np.zeros((nb_family, nb_genus))
family_to_order = np.zeros((nb_order, nb_family))
order_to_class = np.zeros((nb_class, nb_order))

for i in range(nb_specie):
  # specie -> genus
  genus_species = hierarchy.at[i, "genus"]
  ind_genus = genus.index(genus_species)
  species_to_genus[ind_genus, i] = 1

  # genus -> family
  family_species = hierarchy.at[i, "family"]
  ind_family = family.index(family_species)
  genus_to_family[ind_family, ind_genus] = 1

  # family -> order
  order_species = hierarchy.at[i, "order"]
  ind_order = order.index(order_species)
  family_to_order[ind_order, ind_family] = 1

  # order -> class
  class_species = hierarchy.at[i, "class"]
  ind_class = class_.index(class_species)
  order_to_class[ind_class, ind_order] = 1

categorical_crossentropy = keras.losses.CategoricalCrossentropy()
# categorical_crossentropy = torch.nn.CrossEntropyLoss(reduction='mean') # FIXME
alpha = 0.5
weight0 = math.exp(-alpha * 0)
weight1 = math.exp(-alpha * 1)
weight2 = math.exp(-alpha * 2)
weight3 = math.exp(-alpha * 3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
species_to_genus = torch.tensor(species_to_genus, dtype=torch.float32).to(device)
genus_to_family = torch.tensor(genus_to_family, dtype=torch.float32).to(device)
family_to_order = torch.tensor(family_to_order, dtype=torch.float32).to(device)

# Définition de la fonction de perte
def Hierarchicaloss(specie_to_genus, genus_to_family, family_to_order, batch_size, alpha=0.1):

    # def weight(height=1):
    #   return math.exp(-alpha * height)
    
    def specie_loss(y_true, y_pred):
      # height = 0
      return weight0 * categorical_crossentropy(y_true, y_pred)
  
    def specie_to_genus_loss(y_true, y_pred):
      # height = 1
      y_true_genus = (specie_to_genus.unsqueeze(0) @ y_true.unsqueeze(-1)).mT
      y_pred_genus = (specie_to_genus.unsqueeze(0) @ y_pred.unsqueeze(-1)).mT
      return weight1 * categorical_crossentropy(y_true_genus, y_pred_genus), y_true_genus, y_pred_genus
    
    def genus_to_family_loss(y_true, y_pred):
      # height = 2
      y_true_family = (genus_to_family.unsqueeze(0) @ y_true.unsqueeze(-1)).mT
      y_pred_family = (genus_to_family.unsqueeze(0) @ y_pred.unsqueeze(-1)).mT
      return weight2 * categorical_crossentropy(y_true_family, y_pred_family), y_true_family, y_pred_family
    
    def family_to_order_loss(y_true, y_pred):
      # height = 3
      y_true_order = (family_to_order.unsqueeze(0) @ y_true.unsqueeze(-1)).mT
      y_pred_order = (family_to_order.unsqueeze(0) @ y_pred.unsqueeze(-1)).mT
      return weight3 * categorical_crossentropy(y_true_order, y_pred_order)#, y_true_order, y_pred_order

    def HIERARCHICAL_loss(y_true, y_pred):
      loss_specie = specie_loss(y_true, y_pred)
      loss_genus, y_true_genus, y_pred_genus = specie_to_genus_loss(y_true, y_pred)
      loss_family, y_true_family, y_pred_family = genus_to_family_loss(y_true_genus, y_pred_genus)
      # loss_order, y_true_order, y_pred_order = family_to_order_loss(y_true_family, y_pred_family)
      loss_order = family_to_order_loss(y_true_family, y_pred_family)

      return (loss_specie + loss_genus + loss_family + loss_order)/batch_size
   
    # Return a function
    return HIERARCHICAL_loss

hierarchy_loss=[Hierarchicaloss(species_to_genus, genus_to_family, family_to_order, batch_size=32, alpha=0.5)]


#### TRAINING ####

# Ajout de l'optimiseur, de la fonction coût et des métriques
lr = 1e-3
# model.compile(optimizers.SGD(learning_rate=lr, momentum=0.9), loss=hierarchy_loss, metrics=['categorical_accuracy'])#, keras.metrics.Precision(), keras.metrics.Recall()])
model.compile(optimizers.SGD(learning_rate=lr, momentum=0.9), loss=categorical_crossentropy, metrics=['categorical_accuracy'])

# Les callbacks
model_checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath='./ResNet.weights.h5',
    save_weights_only=True,
    monitor='val_categorical_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1)
early_stopping_cb = keras.callbacks.EarlyStopping(
   monitor="val_categorical_accuracy",
   min_delta=0.01,
   patience=10,
   verbose=1,
   mode="auto")
reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.1,
                              patience=5, min_lr=0.00001, verbose=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# train_dataloader = train_dataloader.to(device)
# val_dataloader = val_dataloader.to(device)
image_batch, label_batch = next(iter(train_dataloader))
print(image_batch.shape, label_batch.shape)
history = model.fit(train_dataloader, epochs=100, validation_data=val_dataloader, callbacks=[model_checkpoint_cb, early_stopping_cb, reduce_lr_cb])


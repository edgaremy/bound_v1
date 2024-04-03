import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers

import pandas as pd
import numpy as np
import os
import glob
import cv2
import math

from albumentations import (Compose, Rotate, HorizontalFlip, VerticalFlip, Affine, RandomBrightnessContrast, ChannelShuffle)
import albumentations as A
from functools import partial

# Paramètres
IMG_SIZE = 224 # pour utiliser ResNet

train_dir = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/train/"
val_dir = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/val/"

AUGMENTATIONS_TRAIN = Compose([
    Rotate(limit=[0,100], p=0.5),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    Affine(shear=[-45, 45], p=0.5),
    RandomBrightnessContrast(p=0.5)
])   

def get_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img / 255.0, tf.float32) # normalize image
    return img

def process_path(file_path, label):
    img = get_image(file_path)
    # label = tf.one_hot(label, nb_classes)
    return img, label

def augment_image(image, img_size):
    img = cv2.imread(image.decode('utf-8'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    aug_data = AUGMENTATIONS_TRAIN(image=img)
    aug_img = aug_data["image"]
    aug_img = tf.cast(aug_img/255.0, tf.float32) # normalize image
    return tf.image.resize(aug_img, size=[img_size, img_size])

def process_data(image, label, img_size):
    aug_img = tf.numpy_function(func=augment_image, inp=[image, img_size], Tout=tf.float32)
    return aug_img, label

def set_shapes(img, label, img_shape=(IMG_SIZE,IMG_SIZE,3)):
    img.set_shape(img_shape)
    label = tf.one_hot(label, nb_classes)
    return img, label

train_images = glob.glob(os.path.join(train_dir, "*/*"))
train_labels = [img.split("/")[-2] for img in train_images]
class_names = sorted(set(train_labels))
nb_classes = len(class_names)
train_labels = [class_names.index(l) for l in train_labels]
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_image_count = len(train_ds)
train_ds = train_ds.shuffle(train_image_count, reshuffle_each_iteration=False)
# train_ds = train_ds.map(aug_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# create dataset
train_ds = train_ds.map(partial(process_data, img_size=IMG_SIZE),
                  num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
train_ds = train_ds.map(set_shapes, num_parallel_calls=tf.data.AUTOTUNE)

print("Class Names: ", class_names)
print("Number of classes: ", nb_classes)
print("Number of train images: ", train_image_count)


val_images = glob.glob(os.path.join(val_dir, "*/*"))
val_labels = [img.split("/")[-2] for img in val_images]
val_labels = [class_names.index(l) for l in val_labels]
val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
val_image_count = len(val_ds)
val_ds = val_ds.shuffle(val_image_count, reshuffle_each_iteration=False)
val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

print("Number of val images: ", val_image_count)


def configure_for_performance(ds):
  ds = ds.cache()
#   ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size=16)
  ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
  return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
print("\nData loaded & Optimized for performance.")

# conv_base = tf.keras.applications.resnet.ResNet101(
#     include_top=False,
#     weights='imagenet',
#     input_tensor=None,
#     input_shape=(IMG_SIZE, IMG_SIZE, 3),
#     pooling=None,
#     classes=nb_classes,
# )

# conv_base = tf.keras.applications.EfficientNetB0(
#     include_top=False,
#     weights="imagenet",
#     input_tensor=None,
#     input_shape=(IMG_SIZE, IMG_SIZE, 3),
#     pooling=None,
#     classes=nb_classes
# )

conv_base = tf.keras.applications.mobilenet.MobileNet(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    pooling=None,
    classes=nb_classes
)


inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = conv_base(inputs)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(nb_classes, kernel_regularizer=regularizers.L2(1e-4), activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs, name="bee_classifier")

model.summary()

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

categorical_crossentropy = tf.keras.losses.CategoricalCrossentropy()
alpha = 0.5
weight0 = math.exp(-alpha * 0)
weight1 = math.exp(-alpha * 1)
weight2 = math.exp(-alpha * 2)
weight3 = math.exp(-alpha * 3)


# Définition de la fonction de perte
def Hierarchicaloss(specie_to_genus, genus_to_family, family_to_order, batch_size, alpha=0.1):

    # def weight(height=1):
    #   return math.exp(-alpha * height)
    
    def specie_loss(y_true, y_pred):
      # height = 0
      return weight0 * categorical_crossentropy(y_true, y_pred)
  
    def specie_to_genus_loss(y_true, y_pred):
      # height = 1
      y_true_genus = tf.transpose(tf.raw_ops.MatMul(a=specie_to_genus, b=tf.cast(y_true, tf.float64), transpose_b=True))
      y_pred_genus = tf.transpose(tf.raw_ops.MatMul(a=specie_to_genus, b=tf.cast(y_pred, tf.float64), transpose_b=True))
      return weight1 * categorical_crossentropy(y_true_genus, y_pred_genus), y_true_genus, y_pred_genus
    
    def genus_to_family_loss(y_true, y_pred):
      # height = 2
      y_true_family = tf.transpose(tf.raw_ops.MatMul(a=genus_to_family, b=y_true, transpose_b=True))
      y_pred_family = tf.transpose(tf.raw_ops.MatMul(a=genus_to_family, b=y_pred, transpose_b=True))
      return weight2 * categorical_crossentropy(y_true_family, y_pred_family), y_true_family, y_pred_family
    
    def family_to_order_loss(y_true, y_pred):
      # height = 3
      y_true_order = tf.transpose(tf.raw_ops.MatMul(a=family_to_order, b=y_true, transpose_b=True))
      y_pred_order = tf.transpose(tf.raw_ops.MatMul(a=family_to_order, b=y_pred, transpose_b=True))
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

hierarchy_loss=[Hierarchicaloss(species_to_genus, genus_to_family, family_to_order, batch_size=16, alpha=0.5)]

# Ajout de l'optimiseur, de la fonction coût et des métriques
lr = 1e-3
model.compile(optimizers.SGD(learning_rate=lr, momentum=0.9), loss=hierarchy_loss, metrics=['categorical_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Les callbacks
model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath='./ResNet.weights.h5',
    save_weights_only=True,
    monitor='val_categorical_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1)

#early_stopping_cb = tf.keras.callbacks.EarlyStopping(
#    monitor="val_categorical_accuracy",
#    min_delta=0.01,
#    patience=8,
#    verbose=1,
#    mode="auto")
reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.1,
                              patience=5, min_lr=0.00001, verbose=1)

history = model.fit(train_ds, epochs=3, validation_data = val_ds, callbacks=[model_checkpoint_cb, reduce_lr_cb])
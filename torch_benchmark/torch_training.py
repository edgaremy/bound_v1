import dataloading
from model_and_trainer import *

import timm
from functools import partial
import torch
from pytorch_accelerated.trainer import DEFAULT_CALLBACKS
from pytorch_accelerated.callbacks import SaveBestModelCallback, ProgressBarCallback

torch.manual_seed(0) # TODO

IMG_SIZE = 224 # pour utiliser ResNet
data_path = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/"
train_dir = data_path + "train/"
val_dir = data_path + "val/"

training_epochs = 100
cooldown_epochs = 10
num_epochs = training_epochs + cooldown_epochs

model, optimizer, train_dl_kwargs, eval_dl_kwargs = create_model(load_state=True, state_path='./best_model.pt')

print(model.default_cfg)
print(model.get_classifier())
print()
print(model(torch.randn(1, 3, IMG_SIZE, IMG_SIZE)).shape)


# augmentations, normalize = dataloading.create_augmentations(IMG_SIZE)
dataset_train = dataloading.CustomImageDataset(train_dir)
dataset_val = dataloading.CustomImageDataset(val_dir)

# dataset_train = timm.data.create_dataset(
#         "Bee306",
#         root=data_path,
#         split=train_dir,
#         is_training=True,
#         batch_size=32,
#     )

# dataset_val = timm.data.create_dataset(
#         "Bee306",
#         root=data_path,
#         split=val_dir,
#         is_training=False,
#         batch_size=32,
#     )

loss_func = torch.nn.CrossEntropyLoss()
trainer = TimmTrainer(
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        eval_loss_func=loss_func,
        callbacks=[
            *DEFAULT_CALLBACKS,
            SaveBestModelCallback(save_path='ResNet50Baseline.pt',watch_metric="accuracy", greater_is_better=True),
            AccuracyCallback(num_classes=306)
        ]
    )

trainer.train(
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    num_epochs=num_epochs,
    per_device_batch_size=32,

    train_dataloader_kwargs=train_dl_kwargs,
    eval_dataloader_kwargs=eval_dl_kwargs,
    create_scheduler_fn=partial(timm.scheduler.CosineLRScheduler, t_initial=num_epochs)
)
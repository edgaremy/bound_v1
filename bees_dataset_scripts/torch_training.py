import os
import glob
import matplotlib.pyplot as plt

import timm
import timm.loss
import timm.optim
import timm.scheduler
from functools import partial
from timm.data import ImageDataset
from timm.data.mixup import Mixup
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
from torchvision.io import read_image
from torchvision.transforms import v2

import torchmetrics

from pytorch_accelerated.trainer import DEFAULT_CALLBACKS, Trainer
from pytorch_accelerated.callbacks import SaveBestModelCallback, TrainerCallback


IMG_SIZE = 224 # pour utiliser ResNet
data_path = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/"
train_dir = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/train/"
val_dir = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/val/"

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

# TODO USE MIXUP ?
# TODO USE STANDARD DEVIATION OF THE DATASET TO NORMALIZE

normalize = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToDtype(torch.float32),
    v2.Normalize(mean=[0], std=[255]),
])

training_data = CustomImageDataset(train_dir, normalize)#, augmentations)#, augmentations)
train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
class_names = training_data.class_names
nb_classes = len(class_names)

validation_data = CustomImageDataset(val_dir, normalize)
val_dataloader = DataLoader(validation_data, batch_size=32, shuffle=True, num_workers=8)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0].numpy().argmax()
# plt.imshow(img, cmap="gray")
plt.imshow(img.permute(1, 2, 0), cmap="gray")
plt.title(f"Label: {training_data.class_names[label]}")
plt.show()

# MODEL AND OPTIMIZER

# print(timm.list_models('resnet50*', pretrained=True))
model = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=306)
# print(model)
print(model.default_cfg)
print(model.get_classifier())
print()
print(model(torch.randn(1, 3, IMG_SIZE, IMG_SIZE)).shape)

optimizer = timm.optim.create_optimizer_v2(model, opt="lamb", lr=0.01, weight_decay=0.01)

training_epochs = 100
cooldown_epochs = 10
num_epochs = training_epochs + cooldown_epochs
# num_steps_per_epoch = len(train_dataloader)

lr_scheduler = timm.scheduler.CosineLRScheduler(optimizer,
                                                t_initial=training_epochs,
                                                cycle_decay=0.5,
                                                lr_min=1e-6,
                                                t_in_epochs=True,
                                                warmup_t=3,
                                                warmup_lr_init=1e-4,
                                                cycle_limit=1)


# TRAINING

# for epoch in range(num_epochs):

#     num_steps_per_epoch = len(train_dataloader)
#     num_updates = epoch * num_steps_per_epoch

#     for batch in train_dataloader:
#         inputs, targets = batch
#         outputs = model(inputs)
#         loss = loss_function(outputs, targets)

#         loss.backward()
#         optimizer.step()
#         lr_scheduler.step_update(num_updates=num_updates)

#         optimizer.zero_grad()

#     lr_scheduler.step(epoch + 1)

class TimmTrainer(Trainer):
    def __init__(self, eval_loss_func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_updates = None
        self.train_loss_func = kwargs["loss_func"]
        self.eval_loss_func = eval_loss_func

    def create_train_dataloader(self, batch_size: int, train_dl_kwargs: dict = None):
        return timm.data.create_loader(
            dataset=self.train_dataset,
            collate_fn=self.collate_fn,
            batch_size=batch_size,
            **train_dl_kwargs
        )

    def create_eval_dataloader(self, batch_size: int, eval_dl_kwargs: dict = None):
        return timm.data.create_loader(
            dataset=self.eval_dataset,
            collate_fn=self.collate_fn,
            batch_size=batch_size,
            **eval_dl_kwargs
        )

    def train_epoch_start(self):
        super().train_epoch_start()
        self.num_updates = self.run_history.current_epoch * len(self._train_dataloader)
        self.loss_func = self.train_loss_func

    def eval_epoch_start(self):
        super().eval_epoch_start()
        self.loss_func = self.eval_loss_func

    def eval_epoch_end(self):
        if self.scheduler is not None:
            self.scheduler.step(self.run_history.current_epoch + 1)

    def scheduler_step(self):
        self.num_updates += 1
        if self.scheduler is not None:
            self.scheduler.step_update(num_updates=self.num_updates)

class AccuracyCallback(TrainerCallback):
    def __init__(self, num_classes):
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def on_training_run_start(self, trainer, **kwargs):
        self.accuracy.to(trainer.device)

    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        preds = batch_output["model_outputs"].argmax(dim=-1)
        self.accuracy.update(preds, batch[1])

    def on_eval_epoch_end(self, trainer, **kwargs):
        trainer.run_history.update_metric("accuracy", self.accuracy.compute().item())
        self.accuracy.reset()


data_config = timm.data.resolve_data_config({}, model=model, verbose=True)

train_dl_kwargs = {
        "input_size": data_config["input_size"],
        "is_training": True,
        "use_prefetcher": False,
        "mean": data_config["mean"],
        "std": data_config["std"],
        "interpolation": data_config["interpolation"],
        "num_workers": 8,
        "distributed": False,
        "pin_memory": True,
        "persistent_workers": False,
    }

eval_dl_kwargs = {
    "input_size": data_config["input_size"],
    "is_training": False,
    "interpolation": data_config["interpolation"],
    "mean": data_config["mean"],
    "std": data_config["std"],
    "num_workers": 8,
    "distributed": False,
    "crop_pct": data_config["crop_pct"],
    "pin_memory": True,
    "use_prefetcher": False,
    "persistent_workers": False,
}


# train_loss_fn = timm.loss.LabelSmoothingCrossEntropy()
# validate_loss_fn = torch.nn.CrossEntropyLoss()
dataset_train = timm.data.create_dataset(
        "Bee306",
        root=data_path,
        split=train_dir,
        is_training=True,
        batch_size=32,
    )

dataset_val = timm.data.create_dataset(
        "Bee306",
        root=data_path,
        split=val_dir,
        is_training=False,
        batch_size=32,
    )

loss_func = torch.nn.CrossEntropyLoss()
trainer = TimmTrainer(
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        eval_loss_func=loss_func,
        callbacks=[
            *DEFAULT_CALLBACKS,
            SaveBestModelCallback(watch_metric="accuracy", greater_is_better=True),
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
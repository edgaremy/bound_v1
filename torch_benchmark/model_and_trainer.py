import torch
import timm
import timm.loss
import timm.optim
import timm.scheduler

import torchmetrics

from pytorch_accelerated.trainer import Trainer
from pytorch_accelerated.callbacks import TrainerCallback

# MODEL AND OPTIMIZER

def create_model(load_state=False, state_path=None):

    model = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=306)
    optimizer = timm.optim.create_optimizer_v2(model, opt="lamb", lr=0.01, weight_decay=0.01)

    if load_state:
        model.load_state_dict(torch.load(state_path)['model_state_dict'])
        optimizer.load_state_dict(torch.load(state_path)['optimizer_state_dict'])

    # training_epochs = 100
    # cooldown_epochs = 10
    # num_epochs = training_epochs + cooldown_epochs
    # num_steps_per_epoch = len(train_dataloader)

    # lr_scheduler = timm.scheduler.CosineLRScheduler(optimizer,
    #                                                 t_initial=training_epochs,
    #                                                 cycle_decay=0.5,
    #                                                 lr_min=1e-6,
    #                                                 t_in_epochs=True,
    #                                                 warmup_t=3,
    #                                                 warmup_lr_init=1e-4,
    #                                                 cycle_limit=1)

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

    return model, optimizer, train_dl_kwargs, eval_dl_kwargs

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
        batch_labels = torch.argmax(batch[1].squeeze(), dim=1)
        self.accuracy.update(preds, batch_labels)
        # self.accuracy.update(preds, batch[1])

    def on_eval_epoch_end(self, trainer, **kwargs):
        trainer.run_history.update_metric("accuracy", self.accuracy.compute().item())
        self.accuracy.reset()



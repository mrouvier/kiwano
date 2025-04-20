import math

import torch


class SpkScheduler:
    def __init__(
        self,
        optimizer,
        num_epochs=150,
        initial_lr=0.1,
        final_lr=0.00005,
        warm_up_epoch=6,
    ):
        self.current_iter = 0
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.warm_up_epoch = warm_up_epoch

    def set_lr(self):
        current_lr = self.get_current_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

    def get_current_lr(self):
        if self.current_iter <= self.warm_up_epoch:
            l = 0.00000001
            step = (self.initial_lr - l) / self.warm_up_epoch
            return l + step * self.current_iter
        else:
            step = (self.initial_lr - self.final_lr) / (
                self.num_epochs - self.warm_up_epoch
            )
            return self.final_lr + step * (self.num_epochs - self.current_iter)

    def step(self):
        self.set_lr()
        self.current_iter += 1


class IDRDScheduler:
    def __init__(
        self,
        optimizer,
        num_epochs=150,
        initial_lr=0.2,
        warm_up_epoch=5,
        plateau_epoch=25,
        patience=10,
        factor=2,
        amsmloss=0.4,
    ):
        self.current_iter = 0
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.initial_lr = initial_lr
        self.warm_up_epoch = warm_up_epoch
        self.plateau_epoch = plateau_epoch
        self.patience = patience
        self.amsmloss = amsmloss
        self.factor = factor

    def set_epoch(self, epoch):
        self.current_iter = epoch
        self.set_lr()

    def set_lr(self):
        current_lr = self.get_current_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

    def get_amsmloss(self):
        if (self.plateau_epoch + self.warm_up_epoch) <= self.current_iter:
            return self.amsmloss

        if self.current_iter <= self.warm_up_epoch:
            return 0.0

        step = self.amsmloss / (self.plateau_epoch)
        return (self.current_iter - self.warm_up_epoch) * step

    def get_current_lr(self):
        if self.current_iter <= self.warm_up_epoch:
            l = 0.00001
            step = (self.initial_lr - l) / self.warm_up_epoch
            return l + step * self.current_iter
        if (
            self.warm_up_epoch < self.current_iter
            and self.current_iter <= self.plateau_epoch
        ):
            return self.initial_lr
        else:
            n = math.floor(
                1
                + (
                    (self.current_iter - self.plateau_epoch - self.warm_up_epoch)
                    / self.patience
                )
            )
            return self.initial_lr / (self.factor**n)

    def step(self):
        self.set_lr()
        self.current_iter += 1

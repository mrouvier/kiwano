import torch

class SpkScheduler:

    def __init__(self, optimizer, num_epochs = 150, initial_lr = 0.1, final_lr = 0.00005, warm_up_epoch = 6):
        self.current_iter = 0
        self.num_epochs = num_epochs
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.warm_up_epoch = warm_up_epoch
        self.current_iter = 0.0

    def set_lr(self):
        current_lr = self.get_current_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

    def get_current_lr(self):
        if self.current_iter <= warm_up_epoch:
            step = initial_lr / warm_up_epoch
            return step*self.current_iter
        else:
            step = (initial_lr - final_lr) / (num_epochs - warm_up_epoch)
            return final_lr + step * (num_epochs - self.current_iter)


    def step(self):
        self.set_lr()
        self.current_iter += 1

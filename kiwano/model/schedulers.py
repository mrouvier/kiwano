import math

import torch


class WarmupPlateauScheduler:
    """
    Learning rate and margin scheduler combining warm-up, plateau, and
    step-decay strategies.

    This scheduler controls two training hyperparameters simultaneously:

    1. **Learning Rate (LR)** — evolves in three phases:
       - **Warm-up phase:** LR increases linearly from 1e-5 to `initial_lr`
         over `warm_up_epoch` epochs.
       - **Plateau phase:** LR remains constant at `initial_lr` for
         `plateau_epoch` epochs.
       - **Step-decay phase:** LR decays by a factor of `factor` every
         `patience` epochs.

    2. **Margin Loss (typically AM-Softmax margin or similar)** — evolves in two phases:
       - **Warm-up:** margin is 0 during the LR warm-up.
       - **Ramp-up:** increases linearly until reaching `margin_loss` at the
         end of the plateau.
       - **Stable:** remains fixed at `margin_loss` afterwards.

    This scheduler is designed for speaker verification or face recognition
    models where both LR and margin must evolve together for stable training.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer whose learning rate will be updated.
    max_epochs : int, default=150
        Total number of epochs used to bound the schedule.
    initial_lr : float, default=0.2
        Learning rate at the end of warm-up and during the plateau phase.
    warm_up_epoch : int, default=5
        Number of warm-up epochs during which LR grows linearly.
    plateau_epoch : int, default=25
        Number of epochs for which LR remains fixed at `initial_lr`
        and margin continues to increase linearly.
    patience : int, default=10
        Number of epochs between each step decay in the final decay phase.
    factor : float, default=2
        Multiplicative factor applied during LR decay (LR /= factor**n).
    margin_loss : float, default=0.4
        Final value of the margin used after the ramp-up phase.

    Attributes
    ----------
    current_iter : int
        Current epoch index used by the scheduler.

    Methods
    -------
    set_epoch(epoch)
        Manually set the internal epoch counter and update LR accordingly.
    set_lr()
        Update the optimizer’s LR based on the current epoch.
    get_margin_loss()
        Compute the current margin value according to warm-up and plateau rules.
    get_current_lr()
        Compute the current learning rate according to warm-up, plateau,
        and step-decay phases.
    step()
        Advance the scheduler by one epoch and update the LR.
    """

    def __init__(
        self,
        optimizer,
        max_epochs=150,
        initial_lr=0.2,
        warm_up_epoch=5,
        plateau_epoch=25,
        patience=10,
        factor=2,
        margin_loss=0.4,
    ):
        self.current_iter = 0
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.initial_lr = initial_lr
        self.warm_up_epoch = warm_up_epoch
        self.plateau_epoch = plateau_epoch
        self.patience = patience
        self.margin_loss = margin_loss
        self.factor = factor

    def set_epoch(self, epoch):
        self.current_iter = epoch
        self.set_lr()

    def set_lr(self):
        current_lr = self.get_current_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

    def get_margin_loss(self):
        if (self.plateau_epoch + self.warm_up_epoch) <= self.current_iter:
            return self.margin_loss

        if self.current_iter <= self.warm_up_epoch:
            return 0.0

        step = self.margin_loss / (self.plateau_epoch)
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

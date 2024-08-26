from torch import optim

scheduler_linear = optim.lr_scheduler.StepLR
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR
scheduler_rop = optim.lr_scheduler.ReduceLROnPlateau
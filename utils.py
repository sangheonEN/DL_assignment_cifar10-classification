import torch.optim as optim

class WarmupConstantSchedule(optim.lr_scheduler.LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, initial_lr):

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step)*50. / float(max(1.0, warmup_steps)*2000.)
            return initial_lr

        super(WarmupConstantSchedule, self).__init__(optimizer, lr_lambda, last_epoch=-1)


def step_lr(optimizer, step_size, gamma):

    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size= step_size, gamma= gamma)

    return scheduler

def lambda_lr(optimizer):

    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                    lr_lambda=lambda epoch: 0.95 ** epoch,
                                                    last_epoch=-1,
                                                    verbose=False)

    return scheduler






# class Step_lr(optim.lr_scheduler.LambdaLR):
#     def __init__(self, optimizer, initial_lr):
#
#         def lr_lambda(step):
#             if step < 20:
#                 return initial_lr
#             elif step < 40:
#                 return initial_lr * 0.1
#             else:
#                 return initial_lr * 0.01
#
#         super(Step_lr, self).__init__(optimizer, lr_lambda, last_epoch=-1)


#
# class Learning_rate():
#     def __init__(self, args, optimizer):
#         self.lr_type = args.lr_type
#         self.optimizer = optimizer
#         self.warm_step = args.warm_step
#         self.initial_lr = args.initial_lr
#
#         if self.lr_type == 'step':
#             self.step_lr()
#
#         elif self.lr_type == 'warmup':
#             self.warm_lr()
#
#         else:
#             self.lambda_lr()
#
#
#     def step_lr(self):
#
#         scheduler = Step_lr(optimizer=self.optimizer, initial_lr=self.initial_lr)
#
#         return scheduler
#
#
#     def lambda_lr(self):
#
#         scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
#                                                 lr_lambda=lambda epoch: 0.95 ** epoch,
#                                                 last_epoch=-1,
#                                                 verbose=False)
#
#         return scheduler
#
#     def warm_lr(self):
#
#         scheduler = WarmupConstantSchedule(self.optimizer, warmup_steps=self.warm_step, initial_lr=self.initial_lr)
#
#         return scheduler
import argparse
from tensorboardX import SummaryWriter
from dataset import *
import trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-max_epoch', default=80, type=int)
parser.add_argument('-initial_lr', default=0.0001, type=float)
parser.add_argument('-warm_step', default=5, type=int)
parser.add_argument('-lr_type', default='step', type=str, help='origin, step, warmup, lambda')
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-save_dir', default='./save_ckpt', type=str, help='save model parameters path')
parser.add_argument('-mode_flag', default='train', type=str, help='select the mode: [train], [inference]')
parser.add_argument('-root_dir', default='./cifar10', type=str, help='select the mode: [train], [inference]')
parser.add_argument('-backbone', default='resnet', type=str, help='[resnet, efficientnet, mobilenet]')
args = parser.parse_args()

if __name__ == "__main__":

    # data_loader
    dataloader = Cifar10(args)
    train_set, valid_set, test_set = dataloader.data_load()

    summary = SummaryWriter()

    if args.mode_flag == 'train':
        trainer.train(train_set, valid_set, args, device, summary)
    else:
        trainer.inference(test_set, args, device, summary)

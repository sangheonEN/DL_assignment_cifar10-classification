import torch
import torch.nn as nn
from torch.optim import Adam
import os
from early_stopping import EarlyStopping
import model
import utils



def train(train_set, valid_set, args, device, summary):
    # model 결과 log dir
    log_dir = "./log_saver"
    log_heads = ['epoch', 'val_loss']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, "train_log.csv"), 'w') as f:
        f.write(','.join(log_heads)+ '\n')

    # Seq2Seq -> nn.Sequential 클래스를 이용해 Multi Layer 구성을 만듬.
    if args.backbone == 'resnet':
        endtoendmodel = model.ResNet(class_num=10).to(device)
    elif args.backbone == 'efficientnet':
        endtoendmodel = model.Efficientnet(class_num=10).to(device)
    else:
        endtoendmodel = model.MobileNetV3(class_num=10).to(device)

    early_stopping = EarlyStopping(patience=20, improved_valid=True)


    if os.path.exists(os.path.join(args.save_dir, f'{args.backbone}_checkpoint.pth.tar')):
        # load existing model
        print('==> loading existing model')
        model_info = torch.load(os.path.join(args.save_dir, f'{args.backbone}_checkpoint.pth.tar'))
        endtoendmodel.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(endtoendmodel.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        cur_epoch = 0


    optimizer = Adam(endtoendmodel.parameters(), lr=args.initial_lr)


    # learning rate scheduling
    if args.lr_type == 'origin':
        pass
    elif args.lr_type == 'step':
        lr_scheduler = utils.Step_lr(optimizer, initial_lr=args.initial_lr)

    elif args.lr_type == 'warmup':
        lr_scheduler = utils.WarmupConstantSchedule(optimizer, warmup_steps=args.warm_step, initial_lr=args.initial_lr)
    else:
        lr_scheduler = utils.lambda_lr(optimizer)


    loss_function = nn.CrossEntropyLoss()


    for epoch in range(cur_epoch, args.max_epoch+1):
        train_loss = 0
        for batch_num, (input, target) in enumerate(train_set):
            optimizer.zero_grad()
            endtoendmodel.train()
            input, target = input.float().to(device), target.to(device)
            output = endtoendmodel(input)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if epoch % 10 == 0:
            summary.add_scalar('Train_Loss', train_loss, epoch)
            summary.add_scalar(f'{args.lr_type}_Learning_Rate', optimizer.param_groups[0]['lr'], epoch)


        train_loss /= len(train_set.dataset)

        if args.lr_type != 'origin':
            lr_scheduler.step()

        val_loss = 0
        val_total = 0
        val_correct = 0
        endtoendmodel.eval()
        with torch.no_grad():
            for batch_num, (input, target) in enumerate(valid_set):
                input, target = input.float().to(device), target.to(device)
                output = endtoendmodel(input)
                loss = loss_function(output, target)
                val_loss += loss

                val_total += target.size(0)
                val_correct += output.eq(target.data).cpu().sum()
                print(f'# TEST Acc: ({100. * val_correct / val_total :.2f}%) ({val_correct}/{val_total})')

                summary.add_scalar('Validation Acc', round(100. * val_correct / val_total, 2), batch_num)


        if epoch % 10 == 0:
            summary.add_scalar('Valid_Loss', val_loss, epoch)

        val_loss /= len(valid_set.dataset)

        print(f"Epoch: {epoch} Training Loss: {train_loss}, Validation Loss: {val_loss}")

        model_dict = {
            'epoch': epoch,
            'state_dict': endtoendmodel.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        early_stopping(val_loss.item(), model_dict, epoch, args.save_dir, log_dir, args.backbone)
        if early_stopping.early_stop:
            print("Early stopping")
            break

def inference(test_set, args, device, summary):
    # model 결과 log dir
    log_dir = "./saver"
    log_heads = ['batch', 'prediction', 'image_name']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, "test_log.csv"), 'w') as f:
        f.write(','.join(log_heads)+ '\n')

    if args.backbone == 'resnet':
        endtoendmodel = model.ResNet(class_num=10).to(device)
    elif args.backbone == 'efficientnet':
        endtoendmodel = model.Efficientnet(class_num=10).to(device)
    else:
        endtoendmodel = model.MobileNetV3(class_num=10).to(device)

    if os.path.exists(os.path.join(args.save_dir, f'{args.backbone}_checkpoint.pth.tar')):
        # load existing model
        print('==> loading existing model')
        model_info = torch.load(os.path.join(args.save_dir, f'{args.backbone}_checkpoint.pth.tar'))
        endtoendmodel.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(endtoendmodel.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        cur_epoch = 0

    total = 0
    correct = 0

    endtoendmodel.eval()
    with torch.no_grad():
        for batch_num, (input, target) in enumerate(test_set):
            input, target = input.to(device), target.to(device)
            output = endtoendmodel(input)

            # Decoder
            _, prediction = torch.max(output.data, 1)

            total += target.size(0)
            correct += prediction.eq(target.data).cpu().sum()
            print(f'# TEST Acc: ({100. * correct / total :.2f}%) ({correct}/{total})')

            summary.add_scalar('Test Acc', round(100. * correct / total, 2), batch_num)


            log = [batch_num, prediction, target]
            with open(os.path.join(log_dir, 'test_log.csv'), 'a') as f:
                log_1 = list(map(str, log))
                f.write(','.join(log_1) + '\n')




# import itertools
# import numpy as np
# import matplotlib.pyplot as plt
#
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
# @torch.no_grad()
#
# #def get_all_preds(modell, loader):
# #  all_preds = torch.tensor([])
# #  for batch in loader:
# #    images, labels = batch
# #    images, labels = images.to(device), labels.to(device)
#
# #    preds = modell(images)
# #    all_preds = torch.cat((all_preds.to(device), preds.to(device)) ,dim=0)
#
# #  return all_preds
#
#
# def get_all_preds(model, loader):
#   all_preds = torch.tensor([])
#   for batch in loader:
#     images, labels = batch
#     images, labels = images.to(device), labels.to(device)
#
#     preds = model(images)
#     all_preds = torch.cat((all_preds.to(device), preds.to(device)) ,dim=0)
#
#   return all_preds
#
# test_preds = get_all_preds(net, testloader)
#
# from sklearn.metrics import confusion_matrix
# #cm = confusion_matrix(testset.targets, test_preds.argmax(dim=1).cpu())
# cm = confusion_matrix(testset.targets, test_preds.argmax(dim=1).cpu())
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# plt.figure(figsize=(10,10))
# plot_confusion_matrix(cm, classes)
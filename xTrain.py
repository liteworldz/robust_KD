#from torchvision import transforms
import torch

from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
import warnings
import torchattacks
#from tvision import models
from resnet import *
#import preactresnet
#from models import *
from utils import progress_bar
import argparse
import torch.backends.cudnn as cudnn

import dataset

from loss import LossCalulcator

import time
import csv


from pytorchtools import EarlyStopping

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

DEVICES_IDS = [0]


'''
DEFAULT CONSTANTS
'''
TRAIN_END = -1
TEST_END = -1
EPS = 8/255
ALPHA = 2/255
STEPS = 10
STANDARD = 'nt'
ADVERSARIAL = 'at'
KDISTILLATION = 'kd'
NKDISTILLATION = 'nkd'
TEACHER = 'NT'
STUDENT = 'KD'
ADV = 'AT'
NKD = 'NKD'

def normalize(X):
    mu = torch.tensor(dataset.MEAN).view(3, 1, 1).cuda()
    std = torch.tensor(dataset.STD).view(3, 1, 1).cuda()
    return (X - mu)/std


def normalise(x, mean=dataset.MEAN, std=dataset.STD):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x


def get_model_by_name(name, n_classes):
    if name == "resnet34":
        model = ResNet18()  ## preactresnet.PreActResNet18() #
    elif name == "resnet18":
        model = ResNet18()  #models.resnet18() # preactresnet.PreActResNet18() # models.resnet18() #
    else:
        raise Exception('Unknown network name: {0}'.format(name))
    return model


class Small(nn.Module):
    """
    Small Network
    Reference: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html?highlight=cifar10#define-a-convolutional-neural-network
    """

    def __init__(self, num_class=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_class)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


'''
student_loss_fn: Loss function of difference between student
    predictions and ground-truth
distillation_loss_fn: Loss function of difference between soft
    student predictions and soft teacher predictions
alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
temperature: Temperature for softening probability distributions.
    Larger temperature gives softer distributions.
'''


def _hinton_criterion(alpha=0.1, T=3.0):
    def criterion(outputs, targets, labels, a=0.1):
        loss_criterion = nn.CrossEntropyLoss()
        # Ground Truth Loss
        student_loss = loss_criterion(outputs, labels)
        distillation_loss = (loss_criterion(
            F.softmax(targets / T, dim=1),
            F.softmax(outputs / T, dim=1),
        )
            * T**2
        ) 

        loss = a * student_loss + (1 - a) * distillation_loss

        return loss 
    return criterion


'''
The cross-entropy loss and the (negative) log-likelihood are
the same in the following sense:

If applied Pytorch's CrossEntropyLoss to the output layer, you get the same result as applying Pytorch's NLLLoss to a LogSoftmax layer added after your original output layer.

(Could it be! that using CrossEntropyLoss will be more efficient.)

We are trying to maximize the "likelihood" of the model parameters (weights) having the right values. Maximizing the likelihood is the same as maximizing the log-likelihood,
which is the same as minimizing the negative-log-likelihood. For the classification problem, the cross-entropy is the negative-log-likelihood. 
(The "math" definition of cross-entropy applies to your output layer being a (discrete) probability distribution. 

Pytorch's CrossEntropyLoss implicitly adds a soft-max that "normalizes" the output layer into such a probability distribution.)
'''


def train(logname, net, train_loader, val_loader,
          nb_epochs=10, learning_rate=0.1, patience = 200):

    net.train()
    start = time.time()

    optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=5e-4)  #1e-2 1e-4 5e-4
    #optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200 )

    loss_func = nn.CrossEntropyLoss()
    train_loss = []
    total = 0
    correct = 0
    step = 0
    acc = 0
    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience, verbose=True)
    
    print("Standard Training (Teacher) Started..")
    for _epoch in range(nb_epochs):
        optimizer, lr = adjust_learning_rate(0.1, optimizer, _epoch)
        print(f'Epoch: {_epoch + 1} ' + 
              f'acc: {acc:.3f} ' +
              f'Elapsed Time (Min): {int(np.floor((time.time() - start)/60))} ' + 
              f'lr: { lr:.4f}')
        net.train()
        for xs, ys in train_loader:
            xs, ys = Variable(xs), Variable(ys)
            if torch.cuda.is_available():
                xs, ys = xs.cuda(), ys.cuda()
           
            preds = net(xs)
            loss = loss_func(preds, ys)  
            train_losses.append(loss.data.item()) # record training loss
            preds_np = preds.cpu().detach().numpy()
            correct += (np.argmax(preds_np, axis=1) ==
                        ys.cpu().detach().numpy()).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += train_loader.batch_size
            step += 1
            
        
        acc = float(correct) / total
        #progress_bar(_epoch, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (loss.item()/(_epoch+1), 100.*correct/total, correct, total))
        print('[%s] Training accuracy: %.2f%%' % (step, acc * 100))
        total = 0
        correct = 0
        
        valid_losses, val_acc = evalClean(net, val_loader) #change
        #scheduler.step()
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        epoch_len = len(str(nb_epochs))
        
        print_msg = (f'[{_epoch + 1:>{epoch_len}}/{nb_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f} ' +
                     f'Train Acc: {acc:.5f} ' +
                     f'Val Acc: {val_acc:.5f}')
        
        print(print_msg)
        
        # clear lists to track next epoch
        
        train_losses = []
        valid_losses = []
        
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'val_acc': val_acc,
            'epoch': _epoch
                    }
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_acc, net, TEACHER, state)
        
        with open(logname, 'a', newline='') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([f'{_epoch + 1}', f'{train_loss:.6f}', f'{valid_loss:.6f}', f'{acc:.3f}', f'{val_acc:.3f}', f'{lr:.4f}', int(early_stopping.counter), f'{int(np.floor((time.time() - start)/60))}' ])
            
        if early_stopping.early_stop:
            print("Early stopping")
            break
        


def advTrain(logname, net, train_loader, val_loader,
             nb_epochs=10, learning_rate=0.1, patience=200, VERSION='_v1'):
    net.train()
    start = time.time()

    optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=5e-4)     #6e-4
    #optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    loss_func = nn.CrossEntropyLoss()
    train_loss = []
    total = 0
    correct = 0
    step = 0
    acc = 0
    best_acc = 0
    
    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience, verbose=True)
    '''
    The concept of adversarial training implemented into the code by feeding both the original and the perturbed training set into the architecture at the same time. 
    Note that both types of data should be used for adversarial training to prevent the loss in accuracy on the original set of data.
    '''
    # breakstep = 0
    print("Adversarial Training (Robust) Started..")
    attack = torchattacks.PGD(net, eps=EPS, alpha=ALPHA, steps=STEPS)
    for _epoch in range(nb_epochs):
        optimizer, lr = adjust_learning_rate(0.1, optimizer, _epoch)
        print(f'Epoch: {_epoch + 1} ' + 
              f'acc: {acc:.3f} ' +
              f'Elapsed Time (Min): {int(np.floor((time.time() - start)/60))} ' + 
              f'lr: { lr:.4f}')
        net.train()
        for xs, ys in train_loader:
            # Normal Training
            xs, ys = Variable(xs), Variable(ys)
            if torch.cuda.is_available():
                xs, ys = xs.cuda(), ys.cuda()
            adv = attack(xs, ys)
            preds = net(adv)
            loss = loss_func(preds, ys) 
            
            preds_np = preds.cpu().detach().numpy()
            correct += (np.argmax(preds_np, axis=1) ==
                        ys.cpu().detach().numpy()).sum()
            _, predicted = preds.max(1)
            #correct += predicted.eq(ys).sum().item()
        
            total += train_loader.batch_size
            step += 1
            optimizer.zero_grad()
            loss.backward()  # calc gradients
            optimizer.step()  # update gradients
            train_losses.append(loss.data.item()) # record training loss
            #if total % 1000 == 0:
        acc = float(correct) / total
        print('[%s] Adv Training accuracy: %.2f%%' %
                (step, acc * 100))
        total = 0
        correct = 0
        valid_losses, val_acc = evalAdvAttack(net, val_loader)
        #scheduler.step()
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        epoch_len = len(str(nb_epochs))
        
        print_msg = (f'[{_epoch + 1:>{epoch_len}}/{nb_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f} ' +
                     f'Train Acc: {acc:.5f} ' +
                     f'Val Acc: {val_acc:.5f}')
        
        print(print_msg)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'val_acc': val_acc,
            'epoch': _epoch
                    }
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_acc, net, ADV + VERSION, state)
        with open(logname, 'a', newline='') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([f'{_epoch + 1}', f'{train_loss:.6f}', f'{valid_loss:.6f}', f'{acc:.3f}', f'{val_acc:.3f}', f'{lr:.4f}', int(early_stopping.counter), f'{int(np.floor((time.time() - start)/60))}'])
            
        if early_stopping.early_stop:
            print("Early stopping")
            break


def advNKDTrain(logname, net, TEMP, train_loader, val_loader,
               nb_epochs=10, learning_rate=0.1, patience=200, VERSION='v1'):
    net.train()
    start = time.time()

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    #optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    train_loss = []
    total = 0
    correct = 0
    step = 0
    acc = 0
    best_acc = 0
    log = []
    loss_func = nn.CrossEntropyLoss()
    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience, verbose=True)
    
    print('==> Training will run: ', nb_epochs, ' epochs')
    print("Adversarial NKD Training (Robust-NKD) Started..")
    attack = torchattacks.PGD(net, eps=EPS, alpha=ALPHA, steps=STEPS)
    for _epoch in range(nb_epochs):
        optimizer, lr = adjust_learning_rate(0.1, optimizer, _epoch)
        lossCalculator = LossCalulcator(
            TEMP, 0.1).to(device, non_blocking=True)
        print(f'Epoch: {_epoch + 1} ' + 
              f'acc: {acc:.3f} ' +
              f'Elapsed Time (Min): {int(np.floor((time.time() - start)/60))} ' + 
              f'lr: { lr:.4f}')
        net.train()
        for xs, ys in train_loader:
            # NKD Training
            xs, ys = Variable(xs), Variable(ys)
            if torch.cuda.is_available():
                xs, ys = xs.cuda(), ys.cuda()
            adv = attack(xs, ys)
            preds_t = net(adv)
            preds = net(xs)
            loss = lossCalculator(outputs=preds, labels=ys, teacher_outputs=preds_t)
            train_losses.append(loss.data.item()) # record training loss
            
            preds_np = preds_t.cpu().detach().numpy()
            correct += (np.argmax(preds_np, axis=1) ==
                        ys.cpu().detach().numpy()).sum()
            total += train_loader.batch_size

            step += 1
            optimizer.zero_grad()
            loss.backward()  # calc gradients
            optimizer.step()  # update gradients
            #if total % 1000 == 0:
        acc = float(correct) / total
        print('[%s] Adv NKD Training accuracy: %.2f%%' %
                (step, acc * 100))
        total = 0
        correct = 0
        log = lossCalculator.get_log()
        valid_losses, val_acc = evalAdvAttack(net, val_loader)
        #scheduler.step()
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        epoch_len = len(str(nb_epochs))
        
        print_msg = (f'[{_epoch + 1:>{epoch_len}}/{nb_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f} ' +
                     f'Train Acc: {acc:.5f} ' +
                     f'Val Acc: {val_acc:.5f}')
        
        print(print_msg)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'val_acc': val_acc,
            'epoch': _epoch
                    }
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_acc, net, NKD + VERSION, state)
        with open(logname, 'a', newline='') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([f'{_epoch + 1}', f'{train_loss:.6f}', f'{valid_loss:.6f}', f'{acc:.3f}', f'{val_acc:.3f}', f'{lr:.4f}', int(early_stopping.counter), f'{int(np.floor((time.time() - start)/60))}' ])
            
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
def advKDTrain(logname, net, net_t, lossCalculator, train_loader, val_loader,
               nb_epochs=10, learning_rate=0.1, patience=200, VERSION='v1'):
    net.train()
    net_t.eval()
    start = time.time()

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    #optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    train_loss = []
    total = 0
    correct = 0
    step = 0
    acc = 0
    best_acc = 0
    log = []
    
    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience, verbose=True)
    
    print('==> Training will run: ', nb_epochs, ' epochs')
    print("Adversarial KD Training (Robust-KD) Started..")
    attack = torchattacks.PGD(net, eps=EPS, alpha=ALPHA, steps=STEPS)
    for _epoch in range(nb_epochs):
        optimizer, lr = adjust_learning_rate(0.1, optimizer, _epoch)
        print(f'Epoch: {_epoch + 1} ' + 
              f'acc: {acc:.3f} ' +
              f'Elapsed Time (Min): {int(np.floor((time.time() - start)/60))} ' + 
              f'lr: { lr:.4f}')
        net.train()
        net_t.eval()
        for xs, ys in train_loader:
            # KD Training
            xs, ys = Variable(xs), Variable(ys)
            if torch.cuda.is_available():
                xs, ys = xs.cuda(), ys.cuda()
            adv = attack(xs, ys)
            preds_t = net_t(xs)
            preds = net(adv)
            loss = lossCalculator(outputs=preds, labels=ys, teacher_outputs=preds_t)
            train_losses.append(loss.data.item()) # record training loss
            preds_np = preds.cpu().detach().numpy()
            correct += (np.argmax(preds_np, axis=1) ==
                        ys.cpu().detach().numpy()).sum()
            total += train_loader.batch_size

            step += 1
            optimizer.zero_grad()
            loss.backward()  # calc gradients
            optimizer.step()  # update gradients
            #if total % 1000 == 0:
        acc = float(correct) / total
        print('[%s] Adv KD Training accuracy: %.2f%%' %
                (step, acc * 100))
        total = 0
        correct = 0
        log = lossCalculator.get_log()
        valid_losses, val_acc = evalAdvAttack(net, val_loader)
        #scheduler.step()
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        epoch_len = len(str(nb_epochs))
        
        print_msg = (f'[{_epoch + 1:>{epoch_len}}/{nb_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f} ' +
                     f'Train Acc: {acc:.5f} ' +
                     f'Val Acc: {val_acc:.5f}')
        
        print(print_msg)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'val_acc': val_acc,
            'epoch': _epoch
                    }
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_acc, net, STUDENT + VERSION, state)
        with open(logname, 'a', newline='') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([f'{_epoch + 1}', f'{train_loss:.6f}', f'{valid_loss:.6f}', f'{acc:.3f}', f'{val_acc:.3f}', f'{lr:.4f}', int(early_stopping.counter), f'{int(np.floor((time.time() - start)/60))}' ])
            
        if early_stopping.early_stop:
            print("Early stopping")
            break


# Evaluate results on clean data
def evalClean(net=None, val_loader=None):
    print("Evaluating model results on clean data")
    total = 0
    correct = 0
    net.eval()
    criterion = nn.CrossEntropyLoss()
    # to track the validation loss as the model trains
    valid_losses = []
    with torch.no_grad():
        for xs, ys in val_loader:
            xs, ys = Variable(xs), Variable(ys)
            if torch.cuda.is_available():
                xs, ys = xs.cuda(), ys.cuda()
            preds1 = net(xs)
            loss = criterion(preds1, ys)
            valid_losses.append(loss.data.item())
            preds_np1 = preds1.cpu().detach().numpy()
            finalPred = np.argmax(preds_np1, axis=1)
            correct += (finalPred == ys.cpu().detach().numpy()).sum()
            total += len(xs)
    acc = float(correct) / total
    #print('Clean accuracy: %.2f%%' % (acc * 100))
    return valid_losses, acc

# Evaluate results on adversarially perturbed
def evalAdvAttack(net=None, val_loader=None):
    print("Evaluating model results on adv data")
    total = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    # to track the validation loss as the model trains
    valid_losses = []
    net.eval()
    for xs, ys in val_loader:
        if torch.cuda.is_available():
            xs, ys = xs.cuda(), ys.cuda()
        # pytorch PGD
        attack = torchattacks.PGD(net, eps=EPS, alpha=ALPHA, steps=STEPS*2)
        xs, ys = Variable(xs), Variable(ys)
        adv = attack(xs, ys)
        preds1 = net(adv)
        loss = criterion(preds1, ys)
        valid_losses.append(loss.data.item())
        preds_np1 = preds1.cpu().detach().numpy()
        finalPred = np.argmax(preds_np1, axis=1)
        correct += (finalPred == ys.cpu().detach().numpy()).sum()
        total += val_loader.batch_size
    acc = float(correct) / total
    #print('Adv accuracy: {:.3f}ï¼…'.format(acc * 100))
    return valid_losses, acc


def main(args):
    
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    trainloader, valloader, testloader = dataset.get_loader(
        args.val_size, args.batch_size)

    
    print('==> Preparing Log-File')
    if not os.path.isdir('results'):
        os.mkdir('results')
    logname = ('./results/log_' + args.type  + '.csv')
    if not os.path.exists(logname):
        with open(logname, 'a', newline='') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['_epoch', 'train_loss', 'valid_loss', 'acc', 'val_acc', 'lr', 'early_stopping', 'elapsed_time'])
   
    # Model
    print('==> Building model..')
    if args.small:
        net = Small()
        print('==> Small model..')
    else:
        print('==> network:', args.netname)
        net = get_model_by_name(args.netname, 10)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, device_ids=DEVICES_IDS)
        cudnn.benchmark = True

    # load pre-trained student checkpoint (currently only available for student, easily adapted for others.)
    if args.resume:
        if args.type == STANDARD:
            filename = TEACHER
        elif args.type == ADVERSARIAL:
            filename = ADV
        elif args.type == KDISTILLATION:
            filename = STUDENT + '_' + args.version
        else:
            raise AssertionError(
                "please choose trainnig (--type) std:Standard Train, adv: Adversarial Training, kd: Adversarial Knowledge Distillation")

        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(
            'checkpoint'), 'Error: no checkpoint directory found!'
        load_path = "./checkpoint/"
        checkpoint = torch.load(load_path + args.netname + filename + '.pth',
                                map_location=lambda storage, loc: storage.cuda(0))['net']
        net.load_state_dict(checkpoint)
        acc = torch.load(load_path + args.netname + filename + '.pth',
                         map_location=lambda storage, loc: storage.cuda(0))['acc']
        start_epoch = torch.load(load_path + args.netname + filename + '.pth',
                                 map_location=lambda storage, loc: storage.cuda(0))['epoch']

    if args.type == STANDARD:
        train(logname, net, trainloader, valloader, args.epochs, args.lr, args.patience)
    elif args.type == ADVERSARIAL:
        advTrain(logname, net, trainloader, valloader, args.epochs,
                  args.lr, args.patience, args.version)
    elif args.type == NKDISTILLATION:
        advNKDTrain(logname, net, args.temperature, trainloader, valloader, args.epochs, args.lr, args.patience, args.version)
    elif args.type == KDISTILLATION:
        loss_calculator = LossCalulcator(
            args.temperature, args.distillation_weight).to(device, non_blocking=True)

        net_t = get_model_by_name(args.netname, 10)
        net_t = net_t.to(device)
        if device == 'cuda':
            net_t = torch.nn.DataParallel(net_t, device_ids=DEVICES_IDS)
            cudnn.benchmark = True

        # load teacher
        load_path = "./checkpoint/"
        checkpoint = torch.load(load_path + TEACHER + '_.pth',
                                map_location=lambda storage, loc: storage.cuda(0))['net']
        net_t.load_state_dict(checkpoint)
        net_t.eval()
        print('==> loaded Teacher')
        advKDTrain(logname, net, net_t, loss_calculator, trainloader, valloader, args.epochs, args.lr, args.patience, args.version)
    else:
        raise AssertionError(
            "please choose trainnig (--type) std:Standard Train, adv: Adversarial Training, kd: Adversarial Knowledge Distillation")

    #evalClean(net, valloader)
    #evalAdvAttack(net, valloader)

def adjust_learning_rate(learning_rate,optimizer, epoch):
    lr = learning_rate
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer, lr
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--type', type=str, default="nt")
    parser.add_argument('--val_size', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--netname', type=str, default="resnet18")
    parser.add_argument('--version', type=str, default="v1")
    parser.add_argument('--temperature', default=3.0,
                        type=float, help='KD Loss Temperature')
    parser.add_argument('--distillation_weight', default=0.5,
                        type=float, help=' KD distillation weight / ALPHA: 0-1')
    parser.add_argument('--lr', default=0.1,
                        type=float, help='learning rate')
    parser.add_argument('--small', '-s', action='store_true',
                        help='small network for Student')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()
    main(args)

import os
import sys

import torch.nn as nn
import torch.nn.parallel
import random
import argparse
from network.net import weight_init, CNN9LAYER
from network.resnet import resnet18, resnet34
from network.preact_resnet import preact_resnet18, preact_resnet34, preact_resnet101, initialize_weights, conv_init
from network.pointnet import PointNetCls
from data.cifar_prepare import CIFAR10, CIFAR100
from data.mnist_prepare import MNIST
from data.pc_prepare import ModelNet40
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from Utils.noise import noisify_with_P, noisify_cifar10_asymmetric, \
    noisify_cifar100_asymmetric, noisify_mnist_asymmetric, noisify_pairflip, noisify_modelnet40_asymmetric
from Utils.RAdam.radam import RAdam
import numpy as np
import copy
from tqdm import tqdm
import pickle as pkl
from termcolor import cprint
import datetime
import math

def check_folder(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def _init_fn(worker_id):
    np.random.seed(77 + worker_id)


def updateA(s, h, rho=0.9):
    '''
    Used to calculate retroactive loss

    Input
    s : output after softlayer of NN for a specific former epoch
    h : logrithm of output after softlayer of NN at current epoch

    Output
    result : retroactive loss L_retro
    A : correction matrix
    '''
    eps = 1e-4
    h = torch.tensor(h, dtype=torch.float32).reshape(-1, 1)
    s = torch.tensor(s, dtype=torch.float32).reshape(-1, 1)
    A = torch.ones(len(s), len(s))*eps
    A[s.argmax(0)] = rho - eps/(len(s)-1)
    result = -((A.matmul(s)).t()).matmul(h)

    return result, A

def lrt_flip_scheme(pred_softlabels_bar, y_tilde, delta):
    '''
    The LRT correction scheme.
    pred_softlabels_bar is the prediction of the network which is compared with noisy label y_tilde.
    If the LR is smaller than the given threshhold delta, we reject LRT and flip y_tilde to prediction of pred_softlabels_bar

    Input
    pred_softlabels_bar: rolling average of output after softlayers for past 10 epochs. Could use other rolling windows.
    y_tilde: noisy labels at current epoch
    delta: LRT threshholding

    Output
    y_tilde : new noisy labels after cleanning
    clean_softlabels : softversion of y_tilde
    '''
    ntrain = pred_softlabels_bar.shape[0]
    num_class = pred_softlabels_bar.shape[1]
    for i in range(ntrain):
        cond_1 = (not pred_softlabels_bar[i].argmax()==y_tilde[i])
        cond_2 = (pred_softlabels_bar[i].max()/pred_softlabels_bar[i][y_tilde[i]] > delta)
        if cond_1 and cond_2:
            y_tilde[i] = pred_softlabels_bar[i].argmax()
    eps = 1e-2
    clean_softlabels = torch.ones(ntrain, num_class)*eps/(num_class - 1)
    clean_softlabels.scatter_(1, torch.tensor(np.array(y_tilde)).reshape(-1, 1), 1 - eps)
    return y_tilde, clean_softlabels


def learning_rate(init, epoch):
    optim_factor = 0
    if (epoch > 200):
        optim_factor = 4
    elif(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1

    return init*math.pow(0.5, optim_factor)

def main(args):

    random_seed = int(np.random.choice(range(1000), 1))
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True

    arg_which_net = args.network
    arg_dataset = args.dataset
    arg_epoch_start = args.epoch_start
    lr = args.lr
    arg_gpu = args.gpu
    arg_num_gpu = args.n_gpus
    arg_every_n_epoch = args.every_n_epoch   # interval to perform the correction
    arg_epoch_update = args.epoch_update     # the epoch to start correction (warm-up period)
    arg_epoch_interval = args.epoch_interval # interval between two update of A
    noise_level = args.noise_level
    noise_type = args.noise_type             # "uniform", "asymmetric", "none"
    train_val_ratio = 0.9
    which_net = arg_which_net                # "cnn" "resnet18" "resnet34" "preact_resnet18" "preact_resnet34" "preact_resnet101" "pc"
    num_epoch = args.n_epochs                # Total training epochs


    print('Using {}\nTest on {}\nRandom Seed {}\nevery n epoch {}\nStart at epoch {}'.
          format(arg_which_net, arg_dataset, random_seed, arg_every_n_epoch, arg_epoch_start))

    # -- training parameters
    if arg_dataset == 'mnist':
        milestone = [30, 60]
        batch_size = 64
        in_channels = 1
    elif arg_dataset == 'cifar10':
        milestone = [60, 180]
        batch_size = 128
        in_channels = 1
    elif arg_dataset == 'cifar100':
        milestone = [60, 180]
        batch_size = 128
        in_channels = 1
    elif arg_dataset == 'pc':
        milestone = [30, 60]
        batch_size = 128

    start_epoch = 0
    num_workers = 1

    #gamma = 0.5

    # -- specify dataset
    # data augmentation
    if arg_dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif arg_dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.507, 0.487, 0.441)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.507, 0.487, 0.441)),
        ])
    elif arg_dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    else:
        transform_train = None
        transform_test = None

    if arg_dataset == 'cifar10':
        trainset = CIFAR10(root='./data', split='train', train_ratio=train_val_ratio, trust_ratio=0, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=_init_fn)
        valset = CIFAR10(root='./data', split='val', train_ratio=train_val_ratio, trust_ratio=0, download=True, transform=transform_test)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        testset = CIFAR10(root='./data', split='test', download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        images_size = [3, 32, 32]
        num_class = 10
        in_channel = 3

    elif arg_dataset == 'cifar100':
        trainset = CIFAR100(root='./data', split='train', train_ratio=train_val_ratio, trust_ratio=0, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,worker_init_fn=_init_fn)
        valset = CIFAR100(root='./data', split='val', train_ratio=train_val_ratio, trust_ratio=0, download=True, transform=transform_test)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        testset = CIFAR100(root='./data', split='test', download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        num_class = 100
        in_channel = 3

    elif arg_dataset == 'mnist':
        trainset = MNIST(root='./data', split='train', train_ratio=train_val_ratio, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,worker_init_fn=_init_fn)
        valset = MNIST(root='./data', split='val', train_ratio=train_val_ratio, download=True, transform=transform_test)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        testset = MNIST(root='./data', split='test', download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        num_class = 10
        in_channel = 1

    elif arg_dataset == 'pc':
        trainset = ModelNet40(split='train', train_ratio=train_val_ratio, num_ptrs=1024, random_jitter=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=_init_fn, drop_last=True)
        valset = ModelNet40(split='val', train_ratio=train_val_ratio, num_ptrs=1024)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        testset = ModelNet40(split='test', num_ptrs=1024)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        num_class = 40

    print('train data size:', len(trainset))
    print('validation data size:', len(valset))
    print('test data size:', len(testset))

    eps = 1e-6               # This is the epsilon used to soft the label (not the epsilon in the paper)
    ntrain = len(trainset)

    # -- generate noise --
    # y_train is ground truth labels, we should not have any access to  this after the noisy labels are generated
    # algorithm after y_tilde is generated has nothing to do with y_train
    y_train = trainset.get_data_labels()
    y_train = np.array(y_train)

    noise_y_train = None
    keep_indices = None
    p = None

    if(noise_type == 'none'):
        pass
    else:
        if noise_type == "uniform":
            noise_y_train, p, keep_indices = noisify_with_P(y_train, nb_classes=num_class, noise=noise_level, random_state=random_seed)
            trainset.update_corrupted_label(noise_y_train)
            noise_softlabel = torch.ones(ntrain, num_class)*eps/(num_class-1)
            noise_softlabel.scatter_(1, torch.tensor(noise_y_train.reshape(-1, 1)), 1-eps)
            trainset.update_corrupted_softlabel(noise_softlabel)

            print("apply uniform noise")
        else:
            if arg_dataset == 'cifar10':
                noise_y_train, p, keep_indices = noisify_cifar10_asymmetric(y_train, noise=noise_level, random_state=random_seed)
            elif arg_dataset == 'cifar100':
                noise_y_train, p, keep_indices = noisify_cifar100_asymmetric(y_train, noise=noise_level, random_state=random_seed)
            elif arg_dataset == 'mnist':
                noise_y_train, p, keep_indices = noisify_mnist_asymmetric(y_train, noise=noise_level, random_state=random_seed)
            elif arg_dataset == 'pc':
                noise_y_train, p, keep_indices = noisify_modelnet40_asymmetric(y_train, noise=noise_level,
                                                                               random_state=random_seed)
            trainset.update_corrupted_label(noise_y_train)
            noise_softlabel = torch.ones(ntrain, num_class) * eps / (num_class - 1)
            noise_softlabel.scatter_(1, torch.tensor(noise_y_train.reshape(-1, 1)), 1 - eps)
            trainset.update_corrupted_softlabel(noise_softlabel)

            print("apply asymmetric noise")
        print("clean data num:", len(keep_indices))
        print("probability transition matrix:\n{}".format(p))

    # -- create log file
    file_name = '[' + arg_dataset + '_' + which_net + ']' \
                + 'type:' + noise_type + '_' + 'noise:' + str(noise_level) + '_' \
                + '_' + 'start:' + str(arg_epoch_start) + '_' \
                + 'every:' + str(arg_every_n_epoch) + '_'\
                + 'time:' + str(datetime.datetime.now()) + '.txt'
    log_dir = check_folder('new_logs/logs_txt_' + str(random_seed))
    file_name = os.path.join(log_dir, file_name)
    saver = open(file_name, "w")

    saver.write('noise type: {}\nnoise level: {}\nwhen_to_apply_epoch: {}\n'.format(
        noise_type, noise_level, arg_epoch_start))

    if noise_type != 'none':
        saver.write('total clean data num: {}\n'.format(len(keep_indices)))
        saver.write('probability transition matrix:\n{}\n'.format(p))
    saver.flush()

    # -- set network, optimizer, scheduler, etc
    if which_net == "cnn":
        net_trust = CNN9LAYER(input_channel=in_channel, n_outputs=num_class)
        net = CNN9LAYER(input_channel=in_channel, n_outputs=num_class)
        net.apply(weight_init)
        feature_size = 128
    elif which_net == 'resnet18':
        net_trust = resnet18(in_channel=in_channel, num_classes=num_class)
        net = resnet18(in_channel=in_channel, num_classes=num_class)
        feature_size = 512
    elif which_net == 'resnet34':
        net_trust = resnet34(in_channel=in_channel, num_classes=num_class)
        net = resnet34(in_channel=in_channel, num_classes=num_class)
        feature_size = 512
    elif which_net == 'preact_resnet18':
        net_trust = preact_resnet18(num_classes=num_class, num_input_channels=in_channel)
        net = preact_resnet18(num_classes=num_class, num_input_channels=in_channel)
        feature_size = 256
    elif which_net == 'preact_resnet34':
        net_trust = preact_resnet34(num_classes=num_class, num_input_channels=in_channel)
        net = preact_resnet34(num_classes=num_class, num_input_channels=in_channel)
        feature_size = 256
    elif which_net == 'preact_resnet101':
        net_trust = preact_resnet101()
        net = preact_resnet101()
        feature_size = 256
    elif which_net == 'pc':
        net_trust = PointNetCls(k=num_class)
        net = PointNetCls(k=num_class)
        feature_size = 256
    else:
        ValueError('Invalid network!')

    opt_gpus = [i for i in range(arg_gpu, arg_gpu+int(arg_num_gpu))]
    if len(opt_gpus) > 1:
        print("Using ", len(opt_gpus), " GPUs")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in opt_gpus)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if len(opt_gpus) > 1:
        net_trust = torch.nn.DataParallel(net_trust)
        net = torch.nn.DataParallel(net)
    net_trust.to(device)
    net.to(device)
    # net.apply(conv_init)

    # ------------------------- Initialize Setting ------------------------------
    best_acc = 0
    best_epoch = 0
    patience = 50
    no_improve_counter = 0

    A = 1/num_class*torch.ones(ntrain, num_class, num_class, requires_grad=False).float().to(device)
    h = np.zeros([ntrain, num_class])

    criterion_1 = nn.NLLLoss()
    pred_softlabels = np.zeros([ntrain, arg_every_n_epoch, num_class], dtype=np.float)

    train_acc_record = []
    clean_train_acc_record = []
    noise_train_acc_record = []
    val_acc_record = []
    recovery_record = []
    noise_ytrain = copy.copy(noise_y_train)
    #noise_ytrain = torch.tensor(noise_ytrain).to(device)

    cprint("================  Clean Label...  ================", "yellow")
    for epoch in range(num_epoch):  # Add some modification here

        train_correct = 0
        train_loss = 0
        train_total = 0
        delta = 1.2 + 0.02*max(epoch - arg_epoch_update + 1, 0)

        clean_train_correct = 0
        noise_train_correct = 0

        optimizer_trust = RAdam(net_trust.parameters(),
                                     lr=learning_rate(lr, epoch),
                                     weight_decay=5e-4)
        #optimizer_trust = optim.SGD(net_trust.parameters(), lr=learning_rate(lr, epoch), weight_decay=5e-4,
        #                            nesterov=True, momentum=0.9)

        net_trust.train()

        # Train with noisy data
        for i, (images, labels, softlabels, indices) in enumerate(tqdm(trainloader, ncols=100, ascii=True)):
            if images.size(0) == 1:  # when batch size equals 1, skip, due to batch normalization
                continue

            images, labels, softlabels = images.to(device), labels.to(device), softlabels.to(device)
            outputs, features = net_trust(images)
            log_outputs = torch.log_softmax(outputs, 1).float()

            # arg_epoch_start : epoch start to introduce loss retro
            # arg_epoch_interval : epochs between two updating of A
            if epoch in [arg_epoch_start-1, arg_epoch_start+arg_epoch_interval-1]:
                h[indices] = log_outputs.detach().cpu()
            normal_outputs = torch.softmax(outputs, 1)

            if epoch >= arg_epoch_start: # use loss_retro + loss_ce
                A_batch = A[indices].to(device)
                loss = sum([-A_batch[i].matmul(softlabels[i].reshape(-1, 1).float()).t().matmul(log_outputs[i])
                            for i in range(len(indices))]) / len(indices) + \
                       criterion_1(log_outputs, labels)
            else: # use loss_ce
                loss = criterion_1(log_outputs, labels)

            optimizer_trust.zero_grad()
            loss.backward()
            optimizer_trust.step()

            #arg_every_n_epoch : rolling windows to get eta_tilde
            if epoch >= (arg_epoch_update - arg_every_n_epoch):
                pred_softlabels[indices, epoch % arg_every_n_epoch, :] = normal_outputs.detach().cpu().numpy()

            train_loss += loss.item()
            train_total += images.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()

            # For monitoring purpose, comment the following line out if you have your own dataset that doesn't have ground truth label
            train_label_clean = torch.tensor(y_train)[indices].to(device)
            train_label_noise = torch.tensor(noise_ytrain[indices]).to(device)
            clean_train_correct += predicted.eq(train_label_clean).sum().item()
            noise_train_correct += predicted.eq(train_label_noise).sum().item() # acc wrt the original noisy labels

        train_acc = train_correct / train_total * 100
        clean_train_acc = clean_train_correct/train_total*100
        noise_train_acc = noise_train_correct/train_total*100
        print(" Train Epoch: [{}/{}] \t Training Acc wrt Corrected {:.3f} \t Train Acc wrt True {:.3f} \t Train Acc wrt Noise {:.3f}".
              format(epoch, num_epoch, train_acc, clean_train_acc, noise_train_acc))

        # updating A
        if epoch in [arg_epoch_start - 1, arg_epoch_start + 40 - 1]:
            cprint("+++++++++++++++++ Updating A +++++++++++++++++++", "magenta")
            unsolved = 0
            infeasible = 0
            y_soft = trainset.get_data_softlabel()

            with torch.no_grad():
                for i in tqdm(range(ntrain), ncols=100, ascii=True):
                    try:
                        result, A_opt = updateA(y_soft[i], h[i], rho=0.9)
                    except:
                        A[i] = A[i]
                        unsolved += 1
                        continue

                    if (result == np.inf):
                        A[i] = A[i]
                        infeasible += 1
                    else:
                        A[i] = torch.tensor(A_opt)
            print(A[0])
            print("Unsolved points: {} | Infeasible points: {}".format(unsolved, infeasible))

        # applying lRT scheme
        # args_epoch_update : epoch to update labels
        if epoch >= arg_epoch_update:
            y_tilde = trainset.get_data_labels()
            pred_softlabels_bar = pred_softlabels.mean(1)
            clean_labels, clean_softlabels = lrt_flip_scheme(pred_softlabels_bar, y_tilde, delta)
            trainset.update_corrupted_softlabel(clean_softlabels)
            trainset.update_corrupted_label(clean_softlabels.argmax(1))

        # validation
        if not (epoch % 5):
            val_total = 0
            val_correct = 0
            net_trust.eval()

            with torch.no_grad():
                for i, (images, labels, _, _) in enumerate(valloader):
                    images, labels = images.to(device), labels.to(device)

                    outputs, _ = net_trust(images)

                    val_total += images.size(0)
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(labels).sum().item()

            val_acc = val_correct / val_total * 100

            train_acc_record.append(train_acc)
            val_acc_record.append(val_acc)
            clean_train_acc_record.append(clean_train_acc)
            noise_train_acc_record.append(noise_train_acc)

            recovery_acc = np.sum(trainset.get_data_labels() == y_train) / ntrain
            recovery_record.append(recovery_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                no_improve_counter = 0
            else:
                no_improve_counter += 1
                if no_improve_counter >= patience:
                    print('>> No improvement for {} epochs. Stop at epoch {}'.format(patience, epoch))
                    saver.write('>> No improvement for {} epochs. Stop at epoch {}'.format(patience, epoch))
                    saver.write('>> val epoch: {}\n>> current accuracy: {}%\n'.format(epoch, val_acc))
                    saver.write('>> best accuracy: {}\tbest epoch: {}\n\n'.format(best_acc, best_epoch))
                    break

            cprint('val accuracy: {}'.format(val_acc), 'cyan')
            cprint('>> best accuracy: {}\n>> best epoch: {}\n'.format(best_acc, best_epoch), 'green')
            cprint('>> final recovery rate: {}\n'.format(recovery_acc),
                   'green')
            saver.write('>> val epoch: {}\n>> current accuracy: {}%\n'.format(epoch, val_acc))
            saver.write("outputs: {}\n".format(normal_outputs))
            saver.write('>> best accuracy: {}\tbest epoch: {}\n\n'.format(best_acc, best_epoch))
            saver.write(
                '>> final recovery rate: {}%\n'.format(np.sum(trainset.get_data_labels() == y_train) / ntrain * 100))
            saver.flush()

    # If want to train the neural network again with corrected labels and original loss_ce again
    # set args.two_stage to True
    print("Use Two-Stage Model {}".format(args.two_stage))
    if args.two_stage==True:
       criterion_2 = nn.NLLLoss()
       best_acc = 0
       best_epoch = 0
       patience = 50
       no_improve_counter = 0

       cprint("================ Normal Training  ================", "yellow")
       for epoch in range(num_epoch):  # Add some modification here

           train_correct = 0
           train_loss = 0
           train_total = 0

           optimizer_trust = optim.SGD(net.parameters(), momentum=0.9, nesterov=True,
                                       lr=learning_rate(lr, epoch),
                                       weight_decay=5e-4)

           net.train()

           for i, (images, labels, softlabels, indices) in enumerate(tqdm(trainloader, ncols=100, ascii=True)):
               if images.size(0) == 1:  # when batch size equals 1, skip, due to batch normalization
                   continue

               images, labels, softlabels = images.to(device), labels.to(device), softlabels.to(device)
               outputs, features = net(images)
               log_outputs = torch.log_softmax(outputs, 1).float()

               loss = criterion_2(log_outputs, labels)

               optimizer_trust.zero_grad()
               loss.backward()
               optimizer_trust.step()

               train_loss += loss.item()
               train_total += images.size(0)
               _, predicted = outputs.max(1)
               train_correct += predicted.eq(labels).sum().item()

           train_acc = train_correct / train_total * 100
           print(" Train Epoch: [{}/{}] \t Training Accuracy {}%".format(epoch, num_epoch, train_acc))

           if not (epoch % 5):

               val_total = 0
               val_correct = 0
               net_trust.eval()

               with torch.no_grad():
                   for i, (images, labels, _, _) in enumerate(valloader):
                       images, labels = images.to(device), labels.to(device)
                       outputs, _ = net(images)

                   val_total += images.size(0)
                   _, predicted = outputs.max(1)
                   val_correct += predicted.eq(labels).sum().item()

               val_acc = val_correct / val_total * 100

               if val_acc > best_acc:
                   best_acc = val_acc
                   best_epoch = epoch
                   no_improve_counter = 0
               else:
                   no_improve_counter += 1
                   if no_improve_counter >= patience:
                       print('>> No improvement for {} epochs. Stop at epoch {}'.format(patience, epoch))
                       saver.write('>> No improvement for {} epochs. Stop at epoch {}'.format(patience, epoch))
                       saver.write('>> val epoch: {}\n>> current accuracy: {}%\n'.format(epoch, val_acc))
                       saver.write('>> best accuracy: {}\tbest epoch: {}\n\n'.format(best_acc, best_epoch))
                       break

               cprint('val accuracy: {}'.format(val_acc), 'cyan')
               cprint('>> best accuracy: {}\n>> best epoch: {}\n'.format(best_acc, best_epoch), 'green')
               cprint('>> final recovery rate: {}\n'.format(np.sum(trainset.get_data_labels() == y_train) / ntrain), 'green')

    cprint("================  Start Testing  ================", "yellow")
    test_total = 0
    test_correct = 0

    net_trust.eval()
    for i, (images, labels, softlabels, indices) in enumerate(testloader):
        if images.shape[0] == 1:
            continue

        images, labels = images.to(device), labels.to(device)
        outputs, _ = net_trust(images)

        test_total += images.shape[0]
        test_correct += outputs.argmax(1).eq(labels).sum().item()

    test_acc = test_correct/test_total*100
    print("Final test accuracy {} %".format(test_correct/test_total*100))

    return test_acc


if __name__ == "__main__":

    '''
    will return test accuracy
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, help='delimited list input of GPUs', type=int)
    parser.add_argument('--n_gpus', default=1, help="num of GPUS to use", type=int)
    parser.add_argument("--dataset", default='mnist', help='choose dataset', type=str)
    parser.add_argument('--network', default='preact_resnet34', help="network architecture", type=str)
    parser.add_argument('--noise_type', default='uniform', help='noisy type', type=str)
    parser.add_argument('--noise_level', default=0.8, help='noisy level', type=float)
    parser.add_argument('--lr', default=1e-3, help="learning rate", type=float)
    parser.add_argument('--n_epochs', default=180, help="training epoch", type=int)
    parser.add_argument('--epoch_start', default=25, help='epoch start to introduce l_r', type=int)
    parser.add_argument('--epoch_update', default=30, help='epoch start to update labels', type=int)
    parser.add_argument('--epoch_interval', default=40, help="interval for updating A", type=int)
    parser.add_argument('--every_n_epoch', default=10, help='rolling window for estimating f(x)_t', type=int)
    parser.add_argument('--two_stage', default=False, help='train NN again with clean data using original cross entropy', type=bool)
    parser.add_argument('--seed', default=123, help='set random seed', type=int)
    args = parser.parse_args()

    opt_gpus = [i for i in range(args.gpu, (args.gpu + args.n_gpus))]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in opt_gpus)

    test_temp = main(args)


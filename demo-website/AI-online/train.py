import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import visdom
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from net import Net
import utils

def train_net(model_name,learning_rate, batch_size, optimizer,epoch,platform,activate):
    '''
    train_net(model_name,learning_rate, batch_size, optimizer,epoch,platform,activate)
    '''
    dataset_dir = './static/dataset/MNIST'
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor()])
    batch_size = batch_size

    train_dataset = torchvision.datasets.MNIST(root=dataset_dir, train=True, transform=transform)
    val_dataset = torchvision.datasets.MNIST(root=dataset_dir, train=False, transform=transform)

    print('train dataset: {} \nval dataset: {}'.format(len(train_dataset), len(val_dataset)))

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # 显示一个batch
    viz = visdom.Visdom(env='train-mnist')
    viz.image(
        torchvision.utils.make_grid(next(iter(train_dataloader))[0], nrow=8), 
        win='train-image',
        opts=dict(title='1st batch-train-image'))

    # plt.figure()
    # utils.imshow(next(iter(train_dataloader)))
    # plt.show()

    # ------------------模型，优化方法------------------------------
    # device = torch.device(platform)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = Net()
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fc = nn.CrossEntropyLoss()

    # -----------------训练---------------------------------------
    loss_win = viz.line(np.arange(10), opts=dict(title='Loss'))
    acc_win = viz.line(X=np.column_stack((np.array(0), np.array(0))),
                    Y=np.column_stack((np.array(0), np.array(0))), opts=dict(title='Acc',legned=['Train_acc', 'Val_acc']))
    iter_count = 0
    for epoch in range(epoch):

        running_loss = 0.0
        tr_loss = 0.0
        tr_acc = 0.0
        ts_acc = 0.0
        tr_total = 0
        tr_correct = 0
        ts_total = 0
        ts_correct = 0


        scheduler.step()
        for i, sample_batch in enumerate(train_dataloader):
            inputs = sample_batch[0].to(device)
            labels = sample_batch[1].to(device)

            net.train()
            optimizer.zero_grad()

            outputs = net(inputs)

            loss = loss_fc(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            tr_total += labels.size(0)
            tr_correct += (torch.max(outputs, 1)[1] == labels).sum().item()

            if i % 200 == 199:
                # test
                for sample_batch in val_dataloader:
                    inputs = sample_batch[0].to(device)
                    labels = sample_batch[1].to(device)

                    net.eval()
                    outputs = net(inputs)

                    _, prediction = torch.max(outputs, 1)
                    ts_correct += (prediction == labels).sum().item()
                    ts_total += labels.size(0)

                tr_loss = running_loss / 200
                tr_acc = tr_correct / tr_total
                ts_acc = ts_correct / ts_total
                iter_count += 200
                if iter_count == 200:
                    viz.line(Y=np.array([tr_loss]), X=np.array([iter_count]), update='replace', win=loss_win)
                    viz.line(Y=np.column_stack((np.array([tr_acc]), np.array([ts_acc]))),
                            X=np.column_stack((np.array([iter_count]), np.array([iter_count]))),
                            win=acc_win, update='replace',
                             opts=dict(legned=['Train_acc', 'Val_acc']))

                else:
                    viz.line(Y=np.array([tr_loss]), X=np.array([iter_count]), update='append', win=loss_win)
                    viz.line(Y=np.column_stack((np.array([tr_acc]), np.array([ts_acc]))),
                            X=np.column_stack((np.array([iter_count]), np.array([iter_count]))),
                            win=acc_win, update='append')

                running_loss = 0
                tr_total = 0
                tr_correct = 0
                ts_total = 0
                ts_correct = 0

    print('Train finish!')
    # torch.save(net.state_dict(),'./static/model/'+model_name+'.pth')
    torch.save(net.state_dict(), './static/model/model_10_2_epoch.pth')

if __name__=='__main__':

    dataset_dir = './static/dataset/MNIST'
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor()])
    batch_size = 64

    train_dataset = torchvision.datasets.MNIST(root=dataset_dir, train=True, transform=transform)
    val_dataset = torchvision.datasets.MNIST(root=dataset_dir, train=False, transform=transform)

    print('train dataset: {} \nval dataset: {}'.format(len(train_dataset), len(val_dataset)))

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # 显示一个batch
    viz = visdom.Visdom(env='train-mnist')
    viz.image(torchvision.utils.make_grid(next(iter(train_dataloader))[0], nrow=8), win='train-image', opts=dict(title='1st batch-train-image'))

    # plt.figure()
    # utils.imshow(next(iter(train_dataloader)))
    # plt.show()

    # ------------------模型，优化方法------------------------------

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = Net()
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fc = nn.CrossEntropyLoss()

    # -----------------训练---------------------------------------
    loss_win = viz.line(np.arange(10),opts=dict(title='Loss'))
    acc_win = viz.line(X=np.column_stack((np.array(0), np.array(0))),
                    Y=np.column_stack((np.array(0), np.array(0))),opts=dict(title='Accuracy'))
    iter_count = 0
    for epoch in range(5):

        running_loss = 0.0
        tr_loss = 0.0
        tr_acc = 0.0
        ts_acc = 0.0
        tr_total = 0
        tr_correct = 0
        ts_total = 0
        ts_correct = 0


        scheduler.step()
        for i, sample_batch in enumerate(train_dataloader):
            inputs = sample_batch[0].to(device)
            labels = sample_batch[1].to(device)

            net.train()
            optimizer.zero_grad()

            outputs = net(inputs)

            loss = loss_fc(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            tr_total += labels.size(0)
            tr_correct += (torch.max(outputs, 1)[1] == labels).sum().item()

            if i % 20 == 19:
                # test
                for sample_batch in val_dataloader:
                    inputs = sample_batch[0].to(device)
                    labels = sample_batch[1].to(device)

                    net.eval()
                    outputs = net(inputs)

                    _, prediction = torch.max(outputs, 1)
                    ts_correct += (prediction == labels).sum().item()
                    ts_total += labels.size(0)

                tr_loss = running_loss / 20
                tr_acc = tr_correct / tr_total
                ts_acc = ts_correct / ts_total
                iter_count += 20
                if iter_count == 20:
                    viz.line(Y=np.array([tr_loss]), X=np.array([iter_count]), update='replace', win=loss_win)
                    viz.line(Y=np.column_stack((np.array([tr_acc]), np.array([ts_acc]))),
                            X=np.column_stack((np.array([iter_count]), np.array([iter_count]))),
                            win=acc_win, update='replace',
                            opts=dict(legned=['Train_acc', 'Val_acc']))

                else:
                    viz.line(Y=np.array([tr_loss]), X=np.array([iter_count]), update='append', win=loss_win)
                    viz.line(Y=np.column_stack((np.array([tr_acc]), np.array([ts_acc]))),
                            X=np.column_stack((np.array([iter_count]), np.array([iter_count]))),
                            win=acc_win, update='append')

                running_loss = 0
                tr_total = 0
                tr_correct = 0
                ts_total = 0
                ts_correct = 0

    print('Train finish!')
    torch.save(net.state_dict(), './model/model_10_2_epoch.pth')
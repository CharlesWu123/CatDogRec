# -*- coding: utf-8 -*-
'''
@Version : 0.1
@Author : Charles
@Time : 2022/10/7 14:46 
@File : train.py 
@Desc : 
'''
import os
import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


from utils.utils import setup_logger, WarmupPolyLR
from utils.io_utils import write_yaml
from dataset import CatDogsDataset
import vgg
# from vgg import vgg16


def init_args():
    params = argparse.ArgumentParser()
    params.add_argument('--model_name', type=str, default='vgg16_bn', help='model_name')
    params.add_argument('--data_root', type=str, default='./data/cat_vs_dog', help='data root')
    params.add_argument('--epochs', type=int, default=50, help='epochs')
    params.add_argument('--batch_size', type=int, default=32, help='batch size')
    params.add_argument('--lr', type=float, default=1e-3, help='lr')
    params.add_argument('--save_dir', type=str, default='./output', help='save_dir')
    params.add_argument('--log_iter', type=int, default=20, help='log iter')
    params.add_argument('--warmup', type=bool, default=True, help='warmup')
    params.add_argument('--warmup_epoch', type=int, default=2, help='warmup_epoch')
    params.add_argument('--save_latest', type=bool, default=True, help='save latest')
    args = params.parse_args()
    return args


def train(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    save_dir = os.path.join(args.save_dir, f'{args.model_name}-{time.strftime("%Y-%m-%d %H:%M:%S")}')
    # 保存配置
    model_save_dir = os.path.join(save_dir, 'model')
    logs_save_dir = os.path.join(save_dir, 'logs')
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(logs_save_dir, exist_ok=True)
    write_yaml(vars(args), os.path.join(save_dir, 'config.yaml'))
    logger_save_path = os.path.join(logs_save_dir, 'train.log')
    logger = setup_logger(logger_save_path)
    logger.info(args)
    writer = SummaryWriter(logs_save_dir)
    # 数据
    logger.info('Prepare Data...')
    # 数据预处理
    train_transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.4875, 0.4544, 0.4164], [0.2521, 0.2453, 0.2481])
    ])
    test_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.4875, 0.4544, 0.4164], [0.2521, 0.2453, 0.2481])
    ])
    train_dataset = CatDogsDataset(os.path.join(args.data_root, 'train'), train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = CatDogsDataset(os.path.join(args.data_root, 'test'), test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    train_loader_len = len(train_dataloader)
    logger.info('train: {} dataloader, test: {} dataloader'.format(len(train_dataloader), len(test_dataloader)))
    # 模型
    logger.info('Prepare Model...')
    # 自己写的
    # model = vgg16(num_classes=2, is_dropout=args.is_dropout, is_bn=args.is_bn)
    # torchvision
    model = getattr(vgg, args.model_name)(num_classes=2, pretrained=True)
    model.to(device)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=args.step_size,gamma=0.5,last_epoch=-1)
    # WarmupPolyLR
    if args.warmup:
        warmup_iters = args.warmup_epoch * train_loader_len
        scheduler = WarmupPolyLR(optimizer, max_iters=args.epochs * train_loader_len, warmup_iters=warmup_iters, warmup_epoch=args.warmup_epoch)
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    logger.info('Train Begin ...')
    best_acc = 0
    best_epoch = 0
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        acc = 0
        train_loss = 0
        for idx, (data, targets) in enumerate(train_dataloader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            if args.warmup:
                scheduler.step()
            train_loss += loss.item()
            train_loss = float(train_loss) / (idx + 1)
            lr = optimizer.param_groups[0]["lr"]
            # 计算准确率
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(probs, 1)
            acc += (preds == targets).sum()
            train_acc = int(acc) / (targets.size(0) * (idx + 1))
            writer.add_scalar('train/loss', train_loss, global_step)
            writer.add_scalar('train/acc', train_acc, global_step)
            writer.add_scalar('train/lr', lr, global_step)
            global_step += 1
            if (idx + 1) % args.log_iter == 0:
                logger.info(f'[{epoch}/{args.epochs}] [{idx}/{len(train_dataloader)}] global_step: {global_step}, '
                            f'lr: {lr:.6f}, acc: {train_acc:.4f}, loss: {train_loss:.6f}')
        test_acc, test_loss = val(model, test_dataloader, device, criterion)
        writer.add_scalar('test/acc', test_acc, global_step)
        writer.add_scalar('test/loss', test_loss, global_step)
        logger.info(f'[{epoch}/{args.epochs}] lr: {optimizer.param_groups[0]["lr"]:.6f} '
                    f'test acc: {test_acc:.4f}, test loss: {test_loss:.6f}')
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            ckpt = {
                'best_acc': best_acc,
                'best_epoch': best_epoch,
                'state_dict': model.state_dict()
            }
            torch.save(ckpt, os.path.join(model_save_dir, f'{args.model_name}-best.pth'))
        if args.save_latest:
            ckpt = {
                'best_acc': best_acc,
                'best_epoch': best_epoch,
                'state_dict': model.state_dict()
            }
            torch.save(ckpt, os.path.join(model_save_dir, f'{args.model_name}-latest.pth'))
        logger.info(f'[{epoch}/{args.epochs}] current best: acc: {best_acc:.4f}, epoch: {best_epoch}')
    writer.close()
    logger.info('Train Finish.')


@torch.no_grad()
def val(model, dataloader, device, criterion):
    model.eval()
    loss, acc, num = 0, 0, 0
    for idx, (data, targets) in enumerate(dataloader):
        data, targets = data.to(device), targets.to(device)
        logits = model(data)
        loss += criterion(logits, targets).item()
        probs = torch.softmax(logits, dim=1)
        _, preds = torch.max(probs, 1)
        num += targets.size(0)
        acc += (preds == targets).sum()
    return acc.float() / num, loss / num


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    args = init_args()
    train(args)
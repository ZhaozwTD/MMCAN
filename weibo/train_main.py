import random
import torch
import time
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report

from utils_weibo import parse_args
from utils_weibo import AverageMeter
from utils_weibo import Optimizer
from utils_weibo import get_logger
from utils_weibo import device
from model import MutualModel
from datasets_weibo import get_data


def cal_performance(text_pred, visual_pred, label, theta, beta):
    # Get loss
    criterion_cla_text = torch.nn.CrossEntropyLoss()
    criterion_kl_text = torch.nn.KLDivLoss(size_average=True)

    criterion_cla_visual = torch.nn.CrossEntropyLoss()
    criterion_kl_visual = torch.nn.KLDivLoss(size_average=True)

    loss_class = beta * criterion_cla_text(text_pred, label) + (1 - beta) * criterion_cla_visual(visual_pred, label)

    text_pred, visual_pred = F.softmax(text_pred, dim=1), F.softmax(visual_pred, dim=1)
    loss_mutual = criterion_kl_text(text_pred.log(), visual_pred) + criterion_kl_visual(visual_pred.log(), text_pred)

    loss = loss_mutual * theta + (1 - theta) * loss_class

    # Get accuracy
    text_argmax = torch.argmax(text_pred, dim=1).cpu().numpy()
    visual_argmax = torch.argmax(visual_pred, dim=1).cpu().numpy()
    total_argmax = torch.argmax(beta * text_pred + (1 - beta) * visual_pred, dim=1).cpu().numpy()

    label = label.cpu().numpy()

    accuracy_text = accuracy_score(label, text_argmax)
    accuracy_visual = accuracy_score(label, visual_argmax)
    accuracy_total = accuracy_score(label, total_argmax)
    accuracy = [accuracy_text, accuracy_visual, accuracy_total]

    return loss, accuracy


def cal_performance_valid(text_pred, visual_pred, y, label, predict, beta):
    text_pred, visual_pred = F.softmax(text_pred, dim=1), F.softmax(visual_pred, dim=1)
    total_argmax = torch.argmax(beta * text_pred + (1 - beta) * visual_pred, dim=1).cpu().numpy()

    y = y.cpu().numpy()
    label = np.append(label, y)
    predict = np.append(predict, total_argmax)

    accuracy_total = accuracy_score(y, total_argmax)

    return label, predict, accuracy_total


def train_every_epoch(train_iter, model, optimizer, epoch, logger, writer, args):
    model.train()

    losses = AverageMeter()
    times = AverageMeter()

    start = time.time()

    for i, ((x_texts, x_texts_mask, x_images, x_images_mask, matching_embed), y) in enumerate(train_iter):
        text_outp, visual_outp = model(x_texts, x_texts_mask, x_images, x_images_mask, matching_embed)
        loss, accuracy = cal_performance(text_outp, visual_outp, y, args.theta, args.beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        elapsed = time.time() - start
        start = time.time()

        losses.update(loss.item())
        times.update(elapsed)

        if i % args.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'text acc: {3}, visual acc {4}, total acc: {5}\t'
                        'Batch time {time.val:.5f} ({time.avg:.5f})\t'
                        'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(epoch, i, len(train_iter),
                                                                        accuracy[0], accuracy[1], accuracy[2],
                                                                        time=times, loss=losses))
            writer.add_scalar('step_num/train_loss', losses.avg, optimizer.step_num)
            writer.add_scalar('step_num/learning_rate', optimizer.lr, optimizer.step_num)

    return losses.avg


def valid_every_epoch(valid_loader, model, logger, args):
    model.eval()

    label = []
    predict = []

    target_names = ['fake news', 'true news']

    for (x_texts, x_texts_mask, x_images, x_images_mask, matching_embed), y in valid_loader:
        with torch.no_grad():
            text_outp, visual_outp = model(x_texts, x_texts_mask, x_images, x_images_mask, matching_embed)
            label, predict, _ = cal_performance_valid(text_outp, visual_outp, y, label, predict, args.beta)

    accuracy = accuracy_score(label, predict)
    logger.info('dataset accuracy: {}'.format(accuracy))
    logger.info(classification_report(label, predict, target_names=target_names, digits=5))
    return accuracy


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, accuracy, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'accuracy': accuracy,
             'model': model.state_dict(),
             'optimizer': optimizer}

    filename = 'checkpoint.tar'
    torch.save(state, filename)
    if is_best:
        torch.save(state, 'BEST_checkpoint.tar')


def train_model(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    start_epoch = 0
    best_acc = 0
    writer = SummaryWriter()
    epochs_since_improvement = 0
    checkpoint = args.checkpoint
    logger = get_logger()
    logger.info(args)
    logger.info('Begin training...')

    # Initialize / load checkpoint
    if args.checkpoint is None:
        model = MutualModel(args)
        optimizer = Optimizer(args,
                              torch.optim.AdamW(model.parameters(), lr=args.lr,
                                                betas=(0.9, 0.98), eps=1e-09))

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    device_train = device(args)
    model = model.to(device=device_train)
    logger.info('Complete the model loading...')

    # get dataloader
    train_iter, val_iter = get_data(args.data_file, args)
    logger.info('Complete data acquisition...')

    # train epochs
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_every_epoch(train_iter=train_iter,
                                       model=model,
                                       optimizer=optimizer,
                                       epoch=epoch,
                                       logger=logger,
                                       writer=writer,
                                       args=args)
        writer.add_scalar('epoch/train_loss', train_loss, epoch)
        writer.add_scalar('epoch/learning_rate', optimizer.lr, epoch)

        logger.info('\nLearning rate: {}'.format(optimizer.lr))
        logger.info('Step num: {}\n'.format(optimizer.step_num))

        # One epoch's validation
        valid_acc = valid_every_epoch(valid_loader=val_iter,
                                      model=model,
                                      logger=logger,
                                      args=args)
        writer.add_scalar('epoch/valid_acc', valid_acc, epoch)

        # Check if there was an improvement
        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        if not is_best:
            epochs_since_improvement += 1
            logger.info("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_acc, is_best)


def main():
    args = parse_args()
    if args.mode == 'train':
        train_model(args)


if __name__ == '__main__':
    main()

import pdb
import os
import time
import datetime
import argparse
import numpy as np
import utils

import torch
import torch.optim as optim

from yolo import Yolo
from datasets import VOCDetection
from torch.utils.data import DataLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--gradient_accumulations', type=int, default=2, help='number of gradient accums before step')
    parser.add_argument('--data_config', type=str, default='config/voc.data', help='path to data config file')
    parser.add_argument('--pretrained_weights', type=str, help='if specified starts from checkpoint model')
    parser.add_argument('--num_workers', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='interval between saving model weights')
    parser.add_argument('--evaluation_interval', type=int, default=5, help='interval evaluations on validation set')
    parser.add_argument('--output_path', type=str, default='output/', help=' result path')
    parser.add_argument('--use_gpu', default=True, help='whether use gpu or not')
    parser.add_argument('--train_path', type=str, default='2012_train.txt', help='training set path')
    parser.add_argument('--val_path', type=str, default='2012_val.txt', help='validation set path')

    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (default: 5e-4)')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
    parser.add_argument('--map_thresh', type=float, default=0.5, help='iou thresshold for mAP computation')

    args = parser.parse_args()
    print(args)

    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Initiate model
    model = Yolo(num_classes=20).to(device)

    # If specified we start from checkpoint
    if args.pretrained_weights:
        if args.pretrained_weights.endswith('.pth'):
            model.load_state_dict(torch.load(args.pretrained_weights))
        else:
            model.load_darknet_weights(args.pretrained_weights)

    # Get dataloader
    train_dataset = VOCDetection(args.train_path, args.img_size)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                               collate_fn=train_dataset.collate_fn)
    val_dataset = VOCDetection(args.val_path, args.img_size)
    val_loader  = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, 
                             collate_fn=val_dataset.collate_fn)


    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, 
                                              weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.98 ** epoch)

    for epoch in range(args.epochs):

        # model.train()
        # start_time = time.time()

        # for ind, (imgs, targets) in enumerate(train_loader):
        #     print(ind, imgs.shape)
        #     batches_done = len(train_loader) * epoch + ind

        #     imgs = imgs.to(device)
        #     targets = targets.to(device)

        #     outputs, loss = model(imgs, targets)
        #     loss.backward()

        #     if batches_done % args.gradient_accumulations:
        #         # Accumulates gradient before each step
        #         optimizer.step()
        #         optimizer.zero_grad()

        #     log_str = '\n---- [Epoch %d/%d, Batch %d/%d] ----\n' % (epoch+1, args.epochs, ind+1, len(train_loader))
        #     log_str += '\nTotal loss %.3f' % (loss.item())

        #     # Determine approximate time left for epoch
        #     # epoch_batches_left = len(train_loader) - (ind + 1)
        #     # time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (ind + 1))
        #     # log_str += '\n---- ETA {time_left}'
        # print(log_str)
        # epoch_time = time.time() - start_time
        # print('Training time of epoch %d is %.2f' % (epoch, epoch_time))

        if epoch % args.evaluation_interval == 0 and epoch > -1:
            print('\n---- Evaluating Model ----')
            # Evaluate the model on the validation set
            model.eval()

            labels = []
            sample_metrics = []  # List of tuples (TP, confs, pred)

            for ind, (imgs, targets) in enumerate(val_loader):

                imgs = imgs.to(device)
                targets = targets.to(device)

                # Extract labels
                labels += targets[:, 1].tolist()
                # Rescale target
                targets[:, 2:] = utils.xywh2xyxy(targets[:, 2:])
                targets[:, 2:] *= args.img_size

                with torch.no_grad():
                    outputs, _ = model(imgs)
                    outputs = utils.non_max_suppression(outputs, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh)

                sample_metrics += utils.get_batch_statistics(outputs, targets, iou_thresh=args.map_thresh)

            if len(sample_metrics) == 0:
                print('---- mAP is NULL')
            else:
                # Concatenate sample statistics
                true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
                precision, recall, AP, f1, ap_class = utils.ap_per_class(true_positives, pred_scores, pred_labels, labels)
                print('---- mAP %.3f' % (AP.mean()))

        if epoch % args.checkpoint_interval == 0 and epoch > 20:
            torch.save(model.state_dict(), os.path.join(args.output_path, 'yolov3_tiny_ckpt_%d.pth' % epoch))

        scheduler.step()

import os
import cv2
import time
import datetime
import argparse
import utils

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from yolo import Yolo


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='data/samples', help='path to dataset')
    parser.add_argument('--weights_path', type=str, default='weights/yolov3-tiny.weights', help='path to weights file')
    parser.add_argument('--output_path', type=str, default='output/', help=' result path')
    parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
    parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--checkpoint_model', type=str, help='path to checkpoint model')
    parser.add_argument('--use_gpu', type=bool, default=False, help='whether to use gpu')
    args = parser.parse_args()

    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    FloatTensor = torch.cuda.FloatTensor if args.use_gpu and torch.cuda.is_available() else torch.FloatTensor

    classes = utils.load_classes(args.class_path)  # Extracts class labels from file

    # Set up model
    model = Yolo().to(device)

    if args.weights_path is not None:
        # Load darknet weights
        model.load_weights(args.weights_path)
        
    model.eval()  # Set in evaluation mode

    # dataloader = DataLoader(
    #     ImageFolder(args.image_folder, img_size=args.img_size),
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.n_cpu,
    # )

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)


    if not os.path.exists(args.image_folder):
        print ('No file or directory with the name {}'.format(args.image_folder))
        exit()
    else:
        imlist = [os.path.join(os.path.realpath('.'), args.image_folder, img) for img in os.listdir(args.image_folder)]

    loaded_ims = [cv2.imread(x) for x in imlist]
    im_batches = list(map(utils.prep_image, loaded_ims, [args.img_size for x in range(len(imlist))]))

    im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
    im_dim_list = FloatTensor(im_dim_list).repeat(1,2)

    if (len(im_dim_list) % args.batch_size):
        num_batches = len(imlist) // args.batch_size + 1            
        im_batches = [torch.cat((im_batches[i*args.batch_size : min((i+1)*args.batch_size, len(im_batches))]))
                     for i in range(num_batches)]
        im_dim_batches = [torch.cat((im_dim_list[i*args.batch_size : min((i+1)*args.batch_size, len(im_batches))]))
                     for i in range(num_batches)]

    output = []
    for i, batch in enumerate(im_batches):
        start = time.time()
        with torch.no_grad():
            prediction, _ = model(batch)

        prediction = utils.non_max_suppression(prediction, args.conf_thresh, args.nms_thresh)
        end = time.time()
        print("The inference time of batch %d is %.3f" % (i, end - start))
        output.extend(prediction)

    colors = utils.get_cmap()

    for i in range(len(output)):
        if output[i] is not None:
            res = utils.recover_img_size(output[i], im_dim_list[i], args.img_size)
            list(map(lambda x: utils.draw_bounding_box(x, loaded_ims[i], colors, classes), res))
            name = os.path.join(args.output_path, 'det_' + os.path.basename(imlist[i]))
            cv2.imwrite(name, loaded_ims[i])
import os
from PIL import Image
import numpy as np 

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


def letterbox(img, boxes, out_dim, pad_value=128):

    C, H, W = img.shape
    dim_diff = np.abs(H - W)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if H <= W else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)
    C, padded_H, padded_W = img.shape
    # Resize to (C, out_dim, out_dim)
    img = F.interpolate(img.unsqueeze(0), size=out_dim, mode="nearest").squeeze(0)

    x1 = W * (boxes[:, 1] - boxes[:, 3] / 2)
    y1 = H * (boxes[:, 2] - boxes[:, 4] / 2)
    x2 = W * (boxes[:, 1] + boxes[:, 3] / 2)
    y2 = H * (boxes[:, 2] + boxes[:, 4] / 2)

    # Adjust for added padding
    x1 += pad[0]
    y1 += pad[2]
    x2 += pad[1]
    y2 += pad[3]  

    # Returns (x, y, w, h)
    boxes[:, 1] = ((x1 + x2) / 2) / padded_W
    boxes[:, 2] = ((y1 + y2) / 2) / padded_H
    boxes[:, 3] = abs(x1 - x2) / padded_W
    boxes[:, 4] = abs(y1 - y2) / padded_H

    targets = torch.zeros((len(boxes), 6))
    targets[:, 1:] = boxes

    return img, targets

# def ToTensor(img):
#     img = np.array(img).astype(np.float32)
#     # Handle images with less than three channels
#     img = torch.from_numpy(img)
#     if len(img.shape) != 3:
#         img = img.unsqueeze(0)
#         img = img.expand((3, -1, -1))
#     return img.transpose((2, 0, 1)).float().div(255.0)

class VOCDetection(Dataset):
    def __init__(self, list_path, img_size=416):
        
        self.img_size = img_size

        with open(list_path, "r") as file:  
            self.img_files = file.readlines()
        self.label_files = [path.replace("JPEGImages", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
                            for path in self.img_files]

    def __getitem__(self, index):

        # load image
        img_path = self.img_files[index % len(self.img_files)].strip()
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        # Handle images with less than three channels
        if len(img.shape) != 3:
            print('Seems %s does not have 3 dimension' % img_path)
            img = img.unsqueeze(0)
            img = img.expand((3, -1, -1))

        # load label
        label_path = self.label_files[index % len(self.img_files)].strip()
        assert np.loadtxt(label_path) is not None
        boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))

        img, boxes = letterbox(img, boxes, self.img_size)

        return img, boxes

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))

        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)

        imgs = torch.stack([img for img in imgs])
        return imgs, targets

    def __len__(self):
        return len(self.img_files)


if __name__ == '__main__':
    
    train_dataset = VOCDetection('2012_val.txt')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=train_dataset.collate_fn)

    for n in range(10):
        for i, (ims, targets) in enumerate(train_loader):
            print(i)
            # print(ims.shape)
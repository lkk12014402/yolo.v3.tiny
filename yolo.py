import torch
import torch.nn as nn
import numpy as np

from loss import YoloLoss
from yolov3_tiny_base import TinyYolov3Base, Conv2dBatchReLU


class Yolo(nn.Module):
    """
    
    """
    def __init__(self, num_classes=80, input_channels=3, input_size=416,
                 anchors=[(10,14), (23,27), (37,58), (81,82), (135,169), (344,319)], 
                 anchors_mask=[(3,4,5), (0,1,2)], backbone_name='yolov3_tiny'):
        super().__init__()

        self.num_classes = num_classes
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.num_anchors = len(anchors_mask[0])
        self.input_size = input_size
        self.loss = YoloLoss()

        if backbone_name == 'yolov3_tiny':
            self.backbone = TinyYolov3Base(input_channels, output_channels=len(anchors_mask[0])*(5+num_classes))
        else:
            raise NotImplementedError
    
    def forward(self, x, target=None):

        features = self.backbone(x)

        total_loss = []
        output = []
        for idx, x in enumerate(features):

            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor

            batch_size = x.size(0)
            grid_size  = x.size(2)
            current_anchors = [self.anchors[i] for i in self.anchors_mask[idx]]
            stride = self.input_size // grid_size

            prediction = x.view(batch_size, self.num_anchors, self.num_classes+5, grid_size*grid_size)
            prediction = prediction.permute(0, 3, 1, 2).contiguous()
            prediction = prediction.view(batch_size, self.num_anchors*grid_size*grid_size, self.num_classes+5)

            prediction[..., 0] = torch.sigmoid(prediction[..., 0])  # Center x
            prediction[..., 1] = torch.sigmoid(prediction[..., 1])  # Center y
            prediction[..., 4] = torch.sigmoid(prediction[..., 4])   # object coofidence
            prediction[..., 5:] = torch.sigmoid(prediction[..., 5:])  # classs prediction

            # Add offset and scale with anchors
            grid = np.arange(grid_size)
            m,n = np.meshgrid(grid, grid)
            x_offset = FloatTensor(m).view(-1,1)
            y_offset = FloatTensor(n).view(-1,1)
            x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,self.num_anchors).view(-1, 2).unsqueeze(0)
            x_y_offset = x_y_offset.repeat(batch_size, 1, 1)

            _scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in current_anchors])
            scaled_anchors = _scaled_anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
            scaled_anchors = scaled_anchors.repeat(batch_size, 1, 1)

            prediction[..., 0:2] = prediction[..., 0:2] + x_y_offset
            prediction[..., 2:4] = torch.exp(prediction[..., 2:4]) * scaled_anchors
            prediction[..., 0:4] = prediction[..., 0:4] * stride

            output.append(prediction)

            if target is not None:
                total_loss.append(self.loss(prediction.view(batch_size, self.num_anchors, grid_size, grid_size, self.num_classes+5), 
                                            target[idx], _scaled_anchors, stride))

        return torch.cat(output, 1), np.sum(total_loss)


    def load_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            weights = np.fromfile(f, dtype=np.float32)        # The rest are weights

        ptr = 0
        for block in self.backbone.layers:
            for module in block:

                if isinstance(module, Conv2dBatchReLU):
                    conv_layer = module.layers[0]
                    bn_layer =  module.layers[1]

                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                    # Load conv. weights
                    num_w = conv_layer.weight.numel()
                    conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                    conv_layer.weight.data.copy_(conv_w)
                    ptr += num_w

                elif isinstance(module, nn.Conv2d):
                    # Load conv. bias
                    conv_layer = module
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                    # Load conv. weights
                    num_w = conv_layer.weight.numel()
                    conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                    conv_layer.weight.data.copy_(conv_w)
                    ptr += num_w

        # print(self.state_dict())
        print("successfully load weights")
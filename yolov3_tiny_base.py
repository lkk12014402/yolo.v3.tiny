import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchsummary import summary

class Conv2dBatchReLU(nn.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a ReLU.
    They are executed in a sequential manner.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
        momentum (int, optional): momentum of the moving averages of the normalization; Default **0.01**
        relu (class, optional): Which ReLU to use; Default :class:`torch.nn.LeakyReLU`

    Note:
        If you require the `relu` class to get extra parameters, you can use a `lambda` or `functools.partial`:

        >>> conv = ln.layer.Conv2dBatchReLU(
        ...     in_c, out_c, kernel, stride, padding,
        ...     relu=functools.partial(torch.nn.LeakyReLU, 0.1, inplace=True)
        ... )   # doctest: +SKIP
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 momentum=0.01, relu=lambda: nn.LeakyReLU(0.1, inplace = True)):
        super().__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.momentum = momentum

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels, momentum=self.momentum),
            relu()
        )

    def __repr__(self):
        s = '{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, {relu})'
        return s.format(name=self.__class__.__name__, relu=self.layers[2], **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x


class PaddedMaxPool2d(nn.Module):
    """ Maxpool layer with a replicating padding.

    Args:
        kernel_size (int or tuple): Kernel size for maxpooling
        stride (int or tuple, optional): The stride of the window; Default ``kernel_size``
        padding (tuple, optional): (left, right, top, bottom) padding; Default **None**
        dilation (int or tuple, optional): A parameter that controls the stride of elements in the window
    """
    def __init__(self, kernel_size, stride=None, padding=(0, 0, 0, 0), dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.layers = nn.Sequential(
            nn.ZeroPad2d(self.padding),
            nn.MaxPool2d(self.kernel_size, self.stride)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class TinyYolov3Base(nn.Module):
    """
    
    """
    def __init__(self, input_channels=3, output_channels=85):
        super().__init__()

        block_list = [
            OrderedDict([
                ('conv_0',   Conv2dBatchReLU(input_channels, 16, 3, stride=1, padding=1)),
                ('max_1',    nn.MaxPool2d(2, 2)),
                ('conv_2',   Conv2dBatchReLU(16, 32, 3, stride=1, padding=1)),
                ('max_3',    nn.MaxPool2d(2, 2)),
                ('conv_4',   Conv2dBatchReLU(32, 64, 3, stride=1, padding=1)),
                ('max_5',    nn.MaxPool2d(2, 2)),
                ('conv_6',   Conv2dBatchReLU(64, 128, 3, stride=1, padding=1)),
                ('max_7',    nn.MaxPool2d(2, 2)),
                ('conv_8',   Conv2dBatchReLU(128, 256, 3, stride=1, padding=1))
                ]),

            OrderedDict([
                ('max_9',    nn.MaxPool2d(2, 2)),
                ('conv_10',  Conv2dBatchReLU(256, 512, 3, stride=1, padding=1)),
                ('max_11',   PaddedMaxPool2d(2, 1, padding=(0, 1, 0, 1))),
                ('conv_12',  Conv2dBatchReLU(512, 1024, 3, stride=1, padding=1)),
                ('conv_13',  Conv2dBatchReLU(1024, 256, 1, stride=1, padding=0))
                ]),

            OrderedDict([
                ('conv_14',   Conv2dBatchReLU(256,  512, 3, stride=1, padding=1)),
                ('conv_15',   nn.Conv2d(512, output_channels, 1, stride=1, padding=0)),
                ]),

                # yolo_16  
                # route_17 13

            OrderedDict([
                ('conv_18',  Conv2dBatchReLU(256, 128, 1, stride=1, padding=0)),
                ]),

                # upsample_19
                # route_20 19 8

            OrderedDict([
                ('conv_21',  Conv2dBatchReLU(128+256, 256, 3, stride=1, padding=1)),
                ('conv_22',  nn.Conv2d(256, output_channels, 1, stride=1, padding=0)),
                ]),
                # yolo_23
        ] 

        self.layers = nn.ModuleList([nn.Sequential(block) for block in block_list])

    def forward(self, x):
        conv8  = self.layers[0](x)
        conv13 = self.layers[1](conv8)
        conv15 = self.layers[2](conv13)
        conv18 = self.layers[3](conv13)
        upsample19 = F.interpolate(conv18, scale_factor = 2)
        conv22 = self.layers[4]((torch.cat((upsample19, conv8), 1)))

        return [conv15, conv22]


if __name__ == '__main__':
    net = TinyYolov3Base()
    for m in net.state_dict():
        print(m)

    # device = torch.device('cuda')
    # net = TinyYolov3Base().to(device)
    # summary(net, (3, 416, 416))
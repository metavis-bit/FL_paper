import torch
from torch import nn as tnn
import math

def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size, stride=(1, 1)),
        tnn.BatchNorm2d(chann_out),
        tnn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    layers += [tnn.MaxPool2d(kernel_size=(pooling_k, pooling_k), stride=(pooling_s, pooling_s), ceil_mode=False)]
    return tnn.Sequential(*layers)


def vgg_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Dropout(),
        tnn.Linear(size_in, size_out),
        tnn.BatchNorm1d(size_out),
        tnn.ReLU(True)
    )
    return layer


class VGG16(tnn.Module):
    # def __init__(self, n_classes=1000):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        # Cifar100
        self.layers = tnn.ModuleDict({
            "vgg_block1": vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2),
            "vgg_block2": vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2),
            "vgg_block3": vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2),
            "vgg_block4": vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2),
            "vgg_block5": vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2),
            # "fc1": vgg_fc_layer(7 * 7 * 512, 4096),
            "fc1": vgg_fc_layer(1 * 1 * 512, 4096),
            "fc2": vgg_fc_layer(4096, 4096),
            "fc3": tnn.Linear(4096, num_classes)
            # "fc2": vgg_fc_layer(4096, 4096),
            # "fc3": tnn.Linear(4096, n_classes)
        })

        # self.layers = tnn.ModuleDict({
        #     "vgg_block1": vgg_conv_block([1, 64], [64, 64], [3, 3], [1, 1], 2, 2),
        #     "vgg_block2": vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2),
        #     "vgg_block3": vgg_conv_block([128, 256, 256, 256], [256, 256, 256, 256], [3, 3, 3, 3], [1, 1, 1, 1], 2, 2),
        #     "vgg_block4": vgg_conv_block([256, 512, 512, 512], [512, 512, 512, 512], [3, 3, 3, 3], [1, 1, 1, 1], 2, 2),
        #     "vgg_block5": vgg_conv_block([512, 512, 512, 512], [512, 512, 512, 512], [3, 3, 3, 3], [1, 1, 1, 1], 2, 2),
        #     "fc1": vgg_fc_layer(512 * 2 * 2, 4096),
        #     "fc2": vgg_fc_layer(4096, 4096),
        #     "fc3": tnn.Linear(4096, num_classes)
        # })

        self._weight_initialization()

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, tnn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, tnn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        count = 0
        for layer_name, layer in self.layers.items():
            # if count == 0:
            #     print(x.size())
            # count += 1
            # print(layer_name)
            # print(count)
            x = layer(x)
            if layer_name == "vgg_block5":
                # x = (-1, 1 * 1 * 512)
                x = x.view(-1, 512 * 1 * 1)
            # print(type(x))
        # print(x)
        return x


if __name__ == "__main__":
    model = VGG16(num_classes=10)
    print(model)
    x = torch.randn(1, 1, 64, 64)
    out = model(x)
    print(out)
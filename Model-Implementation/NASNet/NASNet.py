import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# 기본 `log_dir` 은 "runs"이며, 여기서는 더 구체적으로 지정하였습니다
writer = SummaryWriter('runs/logs')
#위 행(line)은 runs/logs 폴더를 생성합니다.

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

batch_size = 64
num_epochs = 100

# 헬퍼 함수

def images_to_probs(net, images):
    '''
    학습된 신경망과 이미지 목록으로부터 예측 결과 및 확률을 생성합니다
    '''
    output = net(images, num_epochs)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)

    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()

    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def plot_classes_preds(net, images, labels):
    '''
    학습된 신경망과 배치로부터 가져온 이미지 / 라벨을 사용하여 matplotlib
    Figure를 생성합니다. 이는 신경망의 예측 결과 / 확률과 함께 정답을 보여주며,
    예측 결과가 맞았는지 여부에 따라 색을 다르게 표시합니다. "images_to_probs"
    함수를 사용합니다.
    '''
    preds, probs = images_to_probs(net, images)
    # 배치에서 이미지를 가져와 예측 결과 / 정답과 함께 표시(plot)합니다
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx])
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

class Factorized_reduction(nn.Module):
    def __init__(self, in_channels, num_of_filters):
        super(Factorized_reduction, self).__init__()

        self.op_stride1 = nn.Sequential(
            nn.Conv2d(in_channels, num_of_filters, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_of_filters),
        )

        self.relu = nn.ReLU(inplace=False)
        self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        self.conv1 = nn.Conv2d(in_channels, num_of_filters // 2, 1, stride=2)
        self.conv2 = nn.Conv2d(in_channels, num_of_filters // 2, 1, stride=2)
        self.bn = nn.BatchNorm2d(num_of_filters, eps=1e-3)
        self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)

    def forward(self, x, stride):
        if stride == 1:
            return self.op_stride1(x)

        x = self.relu(x)
        y = self.pad(x)
        out = torch.cat([self.conv1(x), self.conv2(y[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)

        return out


class Reduce_prev_layer(nn.Module):
    def __init__(self, x_width, x_1_width, x_1_channels, num_of_filters):
        super(Reduce_prev_layer, self).__init__()

        self.x_width = x_width
        self.x_1_width = x_1_width
        self.x_1_channels = x_1_channels
        self.num_of_filters = num_of_filters
        self.relu = nn.ReLU()
        self.factorized_reduction = Factorized_reduction(x_1_channels, num_of_filters)
        self.conv = nn.Conv2d(x_1_channels, num_of_filters, 1, stride=1)
        self.bn = nn.BatchNorm2d(num_of_filters, eps=1e-3)

        # 对第一个x_1进行处理
        self.conv_ = nn.Conv2d(96, num_of_filters, 1, stride=1)
        self.bn_ = nn.BatchNorm2d(num_of_filters, eps=1e-3)

    def forward(self, x, x_1):
        if x_1 is None:
            x_1 = x

        if self.x_width != self.x_1_width:
            x_1 = self.relu(x_1)
            x_1 = self.factorized_reduction(x_1, stride=2)

        elif self.x_1_channels != self.num_of_filters:
            x_1 = self.relu(x_1)
            x_1 = self.conv(x_1)
            x_1 = self.bn(x_1)

        return x_1


def calc_drop_keep_prob(keep_prob, cell_idx, total_cells, epoch, max_epochs):
    if keep_prob == 1:
        return 1

    prob = keep_prob
    layer_ratio = cell_idx / total_cells
    prob = 1 - layer_ratio * (1 - prob)
    current_ratio = epoch / max_epochs
    prob = (1 - current_ratio * (1 - prob))

    return prob


class SepConv(nn.Module):
    def __init__(self, C_in, num_of_filters, kernel_size, stride=1):
        super(SepConv, self).__init__()
        C_out = num_of_filters

        # kernel_size = 3,5,7 => padding = 1,2,3
        padding = int((kernel_size - 1) / 2)

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0),
            nn.BatchNorm2d(C_out, eps=1e-3),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_out, C_out, kernel_size=kernel_size, stride=1, padding=padding, groups=C_out),
            nn.Conv2d(C_out, C_out, kernel_size=1, padding=0),
            nn.BatchNorm2d(C_out, eps=1e-3),
        )

    def forward(self, x, keep_prob=1):
        dropout = nn.Dropout2d(keep_prob)
        return dropout(self.op(x))


class identity(nn.Module):
    def __init__(self, C_in, num_of_filters, stride=1):
        super(identity, self).__init__()
        self.C_in = C_in
        self.C_out = num_of_filters

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(self.C_in, self.C_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.C_out, eps=1e-3),
        )

    def forward(self, x):
        if self.C_in != self.C_out:
            return self.op(x)

        return x


class avg_layer(nn.Module):
    def __init__(self, C_in, num_of_filters, stride=1):
        super(avg_layer, self).__init__()

        self.C_in = C_in
        self.C_out = num_of_filters

        self.avg_pool = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)

        self.op = nn.Sequential(
            nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
            nn.Conv2d(self.C_in, self.C_out, 1, stride=1, padding=0),
            nn.BatchNorm2d(self.C_out, eps=1e-3)
        )

    def forward(self, x, keep_prob=1):
        dropout = nn.Dropout2d(keep_prob)

        if self.C_in != self.C_out:
            return dropout(self.op(x))

        return dropout(self.avg_pool(x))


class max_layer(nn.Module):
    def __init__(self, C_in, num_of_filters, stride=1):
        super(max_layer, self).__init__()
        self.C_in = C_in
        self.C_out = num_of_filters

        self.max_pool = nn.MaxPool2d(3, stride=stride, padding=1)

        self.op = nn.Sequential(
            nn.MaxPool2d(3, stride=stride, padding=1),
            nn.Conv2d(self.C_in, self.C_out, 1, stride=1, padding=0),
            nn.BatchNorm2d(self.C_out, eps=1e-3)
        )

    def forward(self, x, keep_prob=1):
        dropout = nn.Dropout2d(keep_prob)

        if self.C_in != self.C_out:
            return dropout(self.op(x))

        return dropout(self.max_pool(x))


class Normal_cell(nn.Module):

    def __init__(self, x_width, x_1_width, x_channels, x_1_channels,
                 filters, keep_prob, cell_idx, total_cells, max_epochs):
        super(Normal_cell, self).__init__()

        # set parameters
        self.filters = filters
        self.keep_prob = keep_prob
        self.cell_idx = cell_idx
        self.total_cells = total_cells
        self.max_epochs = max_epochs

        # set base layer
        self.cell_base_relu = nn.ReLU()
        self.cell_base_conv = nn.Conv2d(x_channels, self.filters, 1, stride=1)
        self.cell_base_bn = nn.BatchNorm2d(self.filters, eps=1e-3)

        self.reduce_prev_layer = Reduce_prev_layer(x_width, x_1_width, x_1_channels, filters)

        self.avg_layer_y3_a = avg_layer(filters, filters)
        self.avg_layer_y4_a = avg_layer(filters, filters)
        self.avg_layer_y4_b = avg_layer(filters, filters)
        self.identity_y1_b = identity(filters, filters)
        self.identity_y3_b = identity(filters, filters)
        self.sepConv_y1_a = SepConv(filters, filters, kernel_size=3)
        self.sepConv_y2_a = SepConv(filters, filters, kernel_size=3)
        self.sepConv_y2_b = SepConv(filters, filters, kernel_size=5)
        self.sepConv_y5_a = SepConv(filters, filters, kernel_size=5)
        self.sepConv_y5_b = SepConv(filters, filters, kernel_size=3)

#        print("Build normal_cell %d, input filters=%d, output filters(one branch)=%d"
#              % (cell_idx, x_channels, filters))

    def forward(self, x, x_1, epoch):
        x_1 = self.reduce_prev_layer(x, x_1)

        x = self.cell_base_bn(self.cell_base_conv(self.cell_base_relu(x)))

        dp_prob = calc_drop_keep_prob(self.keep_prob, self.cell_idx,
                                      self.total_cells, epoch, self.max_epochs)

        y1_a = self.sepConv_y1_a(x, keep_prob=dp_prob)
        y1_b = self.identity_y1_b(x)
        y1 = y1_a + y1_b

        y2_a = self.sepConv_y2_a(x_1, keep_prob=dp_prob)
        y2_b = self.sepConv_y2_b(x, keep_prob=dp_prob)
        y2 = y2_a + y2_b

        y3_a = self.avg_layer_y3_a(x, keep_prob=dp_prob)
        y3_b = self.identity_y3_b(x_1)
        y3 = y3_a + y3_b

        y4_a = self.avg_layer_y4_a(x_1, keep_prob=dp_prob)
        y4_b = self.avg_layer_y4_b(x_1, keep_prob=dp_prob)
        y4 = y4_a + y4_b

        y5_a = self.sepConv_y5_a(x_1, keep_prob=dp_prob)
        y5_b = self.sepConv_y5_b(x_1, keep_prob=dp_prob)
        y5 = y5_a + y5_b

        return torch.cat([y1, y2, y3, y4, y5], dim=1)


class Reduction_cell(nn.Module):
    def __init__(self, x_width, x_1_width, x_channels, x_1_channels,
                 filters, stride,
                 keep_prob, cell_idx, total_cells, max_epochs):
        super(Reduction_cell, self).__init__()
        # set parameters
        self.filters = filters
        self.stride = stride
        self.keep_prob = keep_prob
        self.cell_idx = cell_idx
        self.total_cells = total_cells
        self.max_epochs = max_epochs

        # set blocks
        self.cell_base_relu = nn.ReLU()
        self.cell_base_conv = nn.Conv2d(x_channels, self.filters, 1, stride=1)
        self.cell_base_bn = nn.BatchNorm2d(self.filters, eps=1e-3)

        self.reduce_prev_layer = Reduce_prev_layer(x_width, x_1_width, x_1_channels, filters)

        self.max_layer_y2_a = max_layer(filters, filters, stride)
        self.max_layer_z1_a = max_layer(filters, filters, stride)
        self.avg_layer_y3_a = avg_layer(filters, filters, stride)
        self.avg_layer_z2_a = avg_layer(filters, filters)  # stride=1
        self.identity_z2_b = identity(filters, filters)
        self.sepConv_y1_a = SepConv(filters, filters, 7, stride)
        self.sepConv_y1_b = SepConv(filters, filters, 5, stride)
        self.sepConv_y2_b = SepConv(filters, filters, 7, stride)
        self.sepConv_y3_b = SepConv(filters, filters, 5, stride)
        self.sepConv_z1_b = SepConv(filters, filters, 3)

#        print("Build reduction_cell %d, input filters=%d, output filters(one branch)=%d"
#              % (cell_idx, x_channels, filters))

    def forward(self, x, x_1, epoch):
        x_1 = self.reduce_prev_layer(x, x_1)

        x = self.cell_base_bn(self.cell_base_conv(self.cell_base_relu(x)))

        dp_prob = calc_drop_keep_prob(self.keep_prob, self.cell_idx,
                                      self.total_cells, epoch, self.max_epochs)

        y1_a = self.sepConv_y1_a(x_1, keep_prob=dp_prob)
        y1_b = self.sepConv_y1_b(x, keep_prob=dp_prob)
        y1 = y1_a + y1_b

        y2_a = self.max_layer_y2_a(x, keep_prob=dp_prob)
        y2_b = self.sepConv_y2_b(x_1, keep_prob=dp_prob)
        y2 = y2_a + y2_b

        y3_a = self.avg_layer_y3_a(x, keep_prob=dp_prob)
        y3_b = self.sepConv_y3_b(x_1, keep_prob=dp_prob)
        y3 = y3_a + y3_b

        z1_a = self.max_layer_z1_a(x, keep_prob=dp_prob)
        z1_b = self.sepConv_z1_b(y1, keep_prob=dp_prob)
        z1 = z1_a + z1_b

        z2_a = self.avg_layer_z2_a(y1, keep_prob=dp_prob)
        z2_b = self.identity_z2_b(y2)
        z2 = z2_a + z2_b

        return torch.cat([z1, z2, y3], dim=1)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Auxhead(nn.Module):
    def __init__(self, C_in, x_width, num_classes, final_filters):
        super(Auxhead, self).__init__()

        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1 = nn.Conv2d(C_in, 128, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(128, eps=1e-3)
        self.conv2 = nn.Conv2d(128, final_filters, kernel_size=x_width, stride=1)
        self.flatten = Flatten()
        self.fc = nn.Linear(final_filters, num_classes)

    def forward(self, x):
        x = self.relu(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)

        return nn.Softmax(dim=1)(x)


class Head(nn.Module):
    def __init__(self, x_channels, num_classes):
        super(Head, self).__init__()

        self.relu = nn.ReLU()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.flatten = Flatten()
        self.fc = nn.Linear(x_channels, num_classes)

    def forward(self, x):
        x = self.relu(x)
        x = self.global_pooling(x)
        x = self.flatten(x)
        x = self.fc(x)

        return nn.Softmax(dim=1)(x)


class NASnet(nn.Module):
    def __init__(self, num_normal_cells=6, num_blocks=3,
                 num_classes=10, num_filters=32, stem_multiplier=3, filter_multiplier=2,
                 dimension_reduction=2, final_filters=768,
                 dropout_prob=0.0, drop_path_keep=0.6, max_epochs=300):
        super(NASnet, self).__init__()

        # set parameters
        self.num_normal_cells = num_normal_cells
        self.num_classes = num_classes
        self.final_filters = final_filters

        # set layers
        filters = num_filters
        self.stem = self.create_stem(filters, stem_multiplier)

        # 对应norm1_1，x_1 = x
        self.layer_norm1_1 = Normal_cell(32, 32, filters * stem_multiplier, filters * stem_multiplier,
                                         filters, drop_path_keep,
                                         1, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm1_2 = Normal_cell(32, 32, filters * 5, filters * stem_multiplier,
                                         filters, drop_path_keep,
                                         2, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm1_3 = Normal_cell(32, 32, filters * 5, filters * 5,
                                         filters, drop_path_keep,
                                         3, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm1_4 = Normal_cell(32, 32, filters * 5, filters * 5,
                                         filters, drop_path_keep,
                                         4, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm1_5 = Normal_cell(32, 32, filters * 5, filters * 5,
                                         filters, drop_path_keep,
                                         5, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm1_6 = Normal_cell(32, 32, filters * 5, filters * 5,
                                         filters, drop_path_keep,
                                         6, num_normal_cells * num_blocks, max_epochs)

        old_filters = filters
        filters *= filter_multiplier

        self.layer_redu1 = Reduction_cell(32, 32, old_filters * 5, old_filters * 5,
                                          filters, dimension_reduction,
                                          drop_path_keep, 6, num_normal_cells * num_blocks, max_epochs)

        self.layer_norm2_1 = Normal_cell(16, 32, filters * 3, old_filters * 5,
                                         filters, drop_path_keep,
                                         7, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm2_2 = Normal_cell(16, 16, filters * 5, filters * 3,
                                         filters, drop_path_keep,
                                         8, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm2_3 = Normal_cell(16, 16, filters * 5, filters * 5,
                                         filters, drop_path_keep,
                                         9, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm2_4 = Normal_cell(16, 16, filters * 5, filters * 5,
                                         filters, drop_path_keep,
                                         10, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm2_5 = Normal_cell(16, 16, filters * 5, filters * 5,
                                         filters, drop_path_keep,
                                         11, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm2_6 = Normal_cell(16, 16, filters * 5, filters * 5,
                                         filters, drop_path_keep,
                                         12, num_normal_cells * num_blocks, max_epochs)

        old_filters = filters
        filters *= filter_multiplier

        self.layer_redu2 = Reduction_cell(16, 16, old_filters * 5, old_filters * 5,
                                          filters, dimension_reduction,
                                          drop_path_keep, 12, num_normal_cells * num_blocks, max_epochs)

        self.layer_norm3_1 = Normal_cell(8, 16, filters * 3, old_filters * 5,
                                         filters, drop_path_keep,
                                         13, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm3_2 = Normal_cell(8, 8, filters * 5, filters * 3,
                                         filters, drop_path_keep,
                                         14, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm3_3 = Normal_cell(8, 8, filters * 5, filters * 5,
                                         filters, drop_path_keep,
                                         15, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm3_4 = Normal_cell(8, 8, filters * 5, filters * 5,
                                         filters, drop_path_keep,
                                         16, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm3_5 = Normal_cell(8, 8, filters * 5, filters * 5,
                                         filters, drop_path_keep,
                                         17, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm3_6 = Normal_cell(8, 8, filters * 5, filters * 5,
                                         filters, drop_path_keep,
                                         18, num_normal_cells * num_blocks, max_epochs)

        self.head = Head(640, num_classes)
        self.auxhead = Auxhead(320, 4, num_classes, final_filters)

    def create_stem(self, filters, stem_multiplier):
        stem = nn.Sequential(
            nn.Conv2d(3, filters * stem_multiplier, kernel_size=3, stride=1, padding=1),  # padding=SAME
            nn.BatchNorm2d(filters * stem_multiplier)
        )
        return stem

    def forward(self, input, epoch):
        x = self.stem(input)
        x_1 = None

        y = self.layer_norm1_1(x, x_1, epoch)
        x_1 = x
        x = y

        y = self.layer_norm1_2(x, x_1, epoch)
        x_1 = x
        x = y

        y = self.layer_norm1_3(x, x_1, epoch)
        x_1 = x
        x = y

        y = self.layer_norm1_4(x, x_1, epoch)
        x_1 = x
        x = y

        y = self.layer_norm1_5(x, x_1, epoch)
        x_1 = x
        x = y

        y = self.layer_norm1_6(x, x_1, epoch)
        x_1 = x
        x = y

        y = self.layer_redu1(x, x_1, epoch)
        x_1 = x
        x = y

        y = self.layer_norm2_1(x, x_1, epoch)
        x_1 = x
        x = y

        y = self.layer_norm2_2(x, x_1, epoch)
        x_1 = x
        x = y

        y = self.layer_norm2_3(x, x_1, epoch)
        x_1 = x
        x = y

        y = self.layer_norm2_4(x, x_1, epoch)
        x_1 = x
        x = y

        y = self.layer_norm2_5(x, x_1, epoch)
        x_1 = x
        x = y

        y = self.layer_norm2_6(x, x_1, epoch)
        x_1 = x
        x = y

        aux_head = self.auxhead(x)

        y = self.layer_redu2(x, x_1, epoch)
        x_1 = x
        x = y

        y = self.layer_norm3_1(x, x_1, epoch)
        x_1 = x
        x = y

        y = self.layer_norm3_2(x, x_1, epoch)
        x_1 = x
        x = y

        y = self.layer_norm3_3(x, x_1, epoch)
        x_1 = x
        x = y

        y = self.layer_norm3_4(x, x_1, epoch)
        x_1 = x
        x = y

        y = self.layer_norm3_5(x, x_1, epoch)
        x_1 = x
        x = y

        y = self.layer_norm3_6(x, x_1, epoch)
        x_1 = x
        x = y

        y = self.head(x)

        return y  # , aux_head


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)


    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 모델을 CUDA에 맞게 작동하도록 변경
    print("---------network----------!")
    net = NASnet()
    net.to(device)

    '''
    x = torch.randn(1, 3, 32, 32)
    y, aux_head = net(x, 1)
    print(y.shape)
    print(aux_head.shape)
    '''
    ## loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    # save checkpoint
    ckpt_path = './checkpoint/checkpoint-epoch-{}-batch-{}-trial-{}-.ckpt'.format(num_epochs, batch_size,1)

    # train
    for epoch in range(num_epochs):  # 데이터셋을 수차례 반복

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
            inputs, labels = data[0].to(device), data[1].to(device)

            # 변화도(Gradient) 매개변수를 0으로 만들고
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화를 한 후
            outputs = net(inputs, num_epochs)
            loss = criterion(outputs, labels)
            torch.save({
                'epoch': epoch,
                'model_state_dict':net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss' : loss
            }, ckpt_path)

            loss.backward()
            optimizer.step()  # train!

            # 통계를 출력
            running_loss += loss.item()
            if i % 100 == 99:  # print every 10 epoch
                print('[%d/%d][%d/%d]  loss: %.4f' %
                      (epoch + 1, num_epochs, i + 1, len(trainloader), loss.item()))


                # ...학습 중 손실(running loss)을 기록하고
                writer.add_scalar('training loss',
                                  loss.item(),
                                  epoch * len(trainloader) + i)

                # ...무작위 미니배치(mini-batch)에 대한 모델의 예측 결과를 보여주도록
                # Matplotlib Figure를 기록합니다
                writer.add_figure('predictions vs. actuals',
                                  plot_classes_preds(net, inputs, labels),
                                  global_step=epoch * len(trainloader) + i)


                running_loss = 0.0
    writer.close()
    print('Finished Training')

    # save model
    PATH = './nasnet_cifar.pth'
    torch.save(net.state_dict(), PATH)

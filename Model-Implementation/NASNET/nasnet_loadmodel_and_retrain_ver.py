#https://github.com/yilak1/gap-res-tumor_delete_gap23/blob/64688003d14ad6060a570203ccdac4ccafd1ca16/models/nasnet.py


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# to fix only one gpu
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

# 기본 `log_dir` 은 "runs"
writer = SummaryWriter('runs/logs2')
#위 행(line)은 runs/logs 폴더를 생성

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

batch_size = 64
num_epochs = 40



def get_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    return rank


def get_size():
    """
    Gets size of communicator
    """
    if dist.is_available() and dist.is_initialized():
        size = dist.get_world_size()
    else:
        size = 1
    return size



#### LAYER

class SeperableConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Conv2d(
            input_channels,
            input_channels,
            kernel_size,
            groups=input_channels,
            **kwargs
        )

        self.pointwise = nn.Conv2d(
            input_channels,
            output_channels,
            1
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class SeperableBranch(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        """Adds 2 blocks of [relu-separable conv-batchnorm]."""
        super().__init__()
        self.block1 = nn.Sequential(
            nn.ReLU(),
            SeperableConv2d(input_channels, output_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(output_channels)
        )

        self.block2 = nn.Sequential(
            nn.ReLU(),
            SeperableConv2d(output_channels, output_channels, kernel_size, stride=1, padding=int(kernel_size / 2)),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        return x


class Fit(nn.Module):
    """Make the cell outputs compatible
    Args:
        prev_filters: filter number of tensor prev, needs to be modified
        filters: filter number of normal cell branch output filters
    """

    def __init__(self, prev_filters, filters):
        super().__init__()
        self.relu = nn.ReLU()

        self.p1 = nn.Sequential(
            nn.AvgPool2d(1, stride=2),
            nn.Conv2d(prev_filters, int(filters / 2), 1)
        )

        # make sure there is no information loss
        self.p2 = nn.Sequential(
            nn.ConstantPad2d((0, 1, 0, 1), 0),
            nn.ConstantPad2d((-1, 0, -1, 0), 0),  # cropping
            nn.AvgPool2d(1, stride=2),
            nn.Conv2d(prev_filters, int(filters / 2), 1)
        )

        self.bn = nn.BatchNorm2d(filters)

        self.dim_reduce = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(prev_filters, filters, 1),
            nn.BatchNorm2d(filters)
        )

        self.filters = filters

    def forward(self, inputs):
        x, prev = inputs
        if prev is None:
            return x

        # image size does not match
        elif x.size(2) != prev.size(2):
            prev = self.relu(prev)
            p1 = self.p1(prev)
            p2 = self.p2(prev)
            prev = torch.cat([p1, p2], 1)
            prev = self.bn(prev)

        elif prev.size(1) != self.filters:
            prev = self.dim_reduce(prev)

        return prev


class NormalCell(nn.Module):

    def __init__(self, x_in, prev_in, output_channels):
        super().__init__()

        self.dem_reduce = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(x_in, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels)
        )

        self.block1_left = SeperableBranch(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.block1_right = nn.Sequential()

        self.block2_left = SeperableBranch(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.block2_right = SeperableBranch(
            output_channels,
            output_channels,
            kernel_size=5,
            padding=2,
            bias=False
        )

        self.block3_left = nn.AvgPool2d(3, stride=1, padding=1)
        self.block3_right = nn.Sequential()

        self.block4_left = nn.AvgPool2d(3, stride=1, padding=1)
        self.block4_right = nn.AvgPool2d(3, stride=1, padding=1)

        self.block5_left = SeperableBranch(
            output_channels,
            output_channels,
            kernel_size=5,
            padding=2,
            bias=False
        )
        self.block5_right = SeperableBranch(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.fit = Fit(prev_in, output_channels)

    def forward(self, x):
        x, prev = x

        # return transformed x as new x, and original x as prev
        # only prev tensor needs to be modified
        prev = self.fit((x, prev))

        h = self.dem_reduce(x)

        x1 = self.block1_left(h) + self.block1_right(h)
        x2 = self.block2_left(prev) + self.block2_right(h)
        x3 = self.block3_left(h) + self.block3_right(h)
        x4 = self.block4_left(prev) + self.block4_right(prev)
        x5 = self.block5_left(prev) + self.block5_right(prev)

        return torch.cat([prev, x1, x2, x3, x4, x5], 1), x


class ReductionCell(nn.Module):

    def __init__(self, x_in, prev_in, output_channels):
        super().__init__()

        self.dim_reduce = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(x_in, output_channels, 1),
            nn.BatchNorm2d(output_channels)
        )

        # block1
        self.layer1block1_left = SeperableBranch(output_channels, output_channels, 7, stride=2, padding=3)
        self.layer1block1_right = SeperableBranch(output_channels, output_channels, 5, stride=2, padding=2)

        # block2
        self.layer1block2_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1block2_right = SeperableBranch(output_channels, output_channels, 7, stride=2, padding=3)

        # block3
        self.layer1block3_left = nn.AvgPool2d(3, 2, 1)
        self.layer1block3_right = SeperableBranch(output_channels, output_channels, 5, stride=2, padding=2)

        # block5
        self.layer2block1_left = nn.MaxPool2d(3, 2, 1)
        self.layer2block1_right = SeperableBranch(output_channels, output_channels, 3, stride=1, padding=1)

        # block4
        self.layer2block2_left = nn.AvgPool2d(3, 1, 1)
        self.layer2block2_right = nn.Sequential()

        self.fit = Fit(prev_in, output_channels)

    def forward(self, x):
        x, prev = x
        prev = self.fit((x, prev))

        h = self.dim_reduce(x)

        layer1block1 = self.layer1block1_left(prev) + self.layer1block1_right(h)
        layer1block2 = self.layer1block2_left(h) + self.layer1block2_right(prev)
        layer1block3 = self.layer1block3_left(h) + self.layer1block3_right(prev)
        layer2block1 = self.layer2block1_left(h) + self.layer2block1_right(layer1block1)
        layer2block2 = self.layer2block2_left(layer1block1) + self.layer2block2_right(layer1block2)

        return torch.cat([
            layer1block2,
            # https://github.com/keras-team/keras-applications/blob/master/keras_applications/nasnet.py line 739
            layer1block3,
            layer2block1,
            layer2block2
        ], 1), x


class NasNetA(nn.Module):

    def __init__(self, repeat_cell_num, reduction_num, filters, stemfilter, class_num=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, stemfilter, 3, padding=1, bias=False),
            nn.BatchNorm2d(stemfilter)
        )

        self.prev_filters = stemfilter
        self.x_filters = stemfilter
        self.filters = filters

        self.cell_layers = self._make_layers(repeat_cell_num, reduction_num)

        self.relu = nn.ReLU()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.filters * 6, class_num)

    def _make_normal(self, block, repeat, output):
        """make normal cell
        Args:
            block: cell type
            repeat: number of repeated normal cell
            output: output filters for each branch in normal cell
        Returns:
            stacked normal cells
        """

        layers = []
        for r in range(repeat):
            layers.append(block(self.x_filters, self.prev_filters, output))
            self.prev_filters = self.x_filters
            self.x_filters = output * 6  # concatenate 6 branches

        return layers

    def _make_reduction(self, block, output):
        """make normal cell
        Args:
            block: cell type
            output: output filters for each branch in reduction cell
        Returns:
            reduction cell
        """

        reduction = block(self.x_filters, self.prev_filters, output)
        self.prev_filters = self.x_filters
        self.x_filters = output * 4  # stack for 4 branches

        return reduction

    def _make_layers(self, repeat_cell_num, reduction_num):

        layers = []
        for i in range(reduction_num):
            layers.extend(self._make_normal(NormalCell, repeat_cell_num, self.filters))
            self.filters *= 2
            layers.append(self._make_reduction(ReductionCell, self.filters))

        layers.extend(self._make_normal(NormalCell, repeat_cell_num, self.filters))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.stem(x)
        prev = None
        x, prev = self.cell_layers((x, prev))
        x = self.relu(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def nasnet():
    # stem filters must be 44, it's a pytorch workaround, cant change to other number
    return NasNetA(4, 2, 44, 44)

        # repeat_cell_num, reduction_num, filters, stemfilter, class_num=10)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # init!
    torch.distributed.init_process_group(backend='nccl', init_method="env://", rank =0, world_size=1)  # rank should be 0 ~ world_size-1

    ## check initialization
    if torch.distributed.is_initialized() is True:
        print("initialized!")
    if not torch.distributed.is_initialized():
        print("fail initialized")

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    train_sampler = DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, sampler=train_sampler)


    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    test_sampler = DistributedSampler(testset)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, sampler=test_sampler)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 모델을 CUDA에 맞게 작동하도록 변경
    print("---------network----------! _ trial_2")
    '''
    net = nasnet()
    if torch.cuda.device_count() > 1:
       # net = nn.DataParallel(net)
        net = nn.DataParallel(net)
    net.to(device)
    '''
    model = nasnet()
    model = torch.nn.DataParallel(model)

    PATH = './Nasnet_cifar_module.pth'
    model.module.load_state_dict(torch.load(PATH))
#   net = nn.DataParallel(net)
    net = DDP(model, delay_allreduce=True) # device_ids=[0], find_unused_parameters=True, output_device=1) #find_unused_parameters=True -> for this error :  https://study-grow.tistory.com/entry/pytorch-%EC%97%90%EB%9F%AC-DistributedDataParallel-%EC%97%90%EB%9F%AC


    ## loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    # save checkpoint
    ckpt_path = './checkpoints/checkpoint-epoch-{}-batch-{}-trial-{}-.ckpt'.format(num_epochs, batch_size,1)

    # train
    for epoch in range(num_epochs):  # 데이터셋을 수차례 반복

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # [inputs, labels]의 목록인 data로부터 입력을 받은 후
            inputs, labels = data[0].to(device), data[1].to(device)

            # 변화도(Gradient) 매개변수를 0!
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화를 한 후
            outputs = net(inputs)
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


                # 학습 중 손실(running loss) 기록
                writer.add_scalar('training loss',
                                  loss.item(),
                                  epoch)


                running_loss = 0.0
    writer.close()
    print('Finished Training')

    # save model
    PATH = './Nasnet_cifar_module_epoch_40.pth'
    torch.save(net.module.state_dict(), PATH)

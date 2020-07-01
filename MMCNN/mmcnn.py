import torch
import math 
import torch.nn as nn
import torch.nn.functional as F 


class MaxPool1dPaddingSame(nn.Module):
    '''pytorch version of padding=='same'
    ============== ATTENTION ================
    Only work when dilation == 1, groups == 1
    =========================================
    '''

    def __init__(self, kernel_size, stride):
        super(MaxPool1dPaddingSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        batch_size, num_channels, length = x.shape
        if length % self.stride == 0:
            out_length = length // self.stride
        else:
            out_length = length // self.stride + 1

        pad = math.ceil((out_length * self.stride +
                         self.kernel_size - length - self.stride) / 2)
        out = F.max_pool1d(x, kernel_size=self.kernel_size,
                           stride=self.stride, padding=pad)
        return out


class Conv1dPaddingSame(nn.Module):
    '''pytorch version of padding=='same'
    ============== ATTENTION ================
    Only work when dilation == 1, groups == 1
    =========================================
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv1dPaddingSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight = nn.Parameter(torch.rand((out_channels,
                                               in_channels, kernel_size)))
        # nn.Conv1d default set bias=True，所以这里也开启 bias
        self.bias = nn.Parameter(torch.rand(out_channels))

    def forward(self, x):
        batch_size, num_channels, length = x.shape
        if length % self.stride == 0:
            out_length = length // self.stride
        else:
            out_length = length // self.stride + 1

        pad = math.ceil((out_length * self.stride +
                         self.kernel_size - length - self.stride) / 2)
        out = F.conv1d(input=x,
                       weight=self.weight,
                       stride=self.stride,
                       bias=self.bias,
                       padding=pad)
        return out

def activations(activation):
    '''
    根据需要选择激活函数
    '''
    act = nn.ModuleDict({
                'relu': nn.ReLU(),
                'elu': nn.ELU(),
                'sigmoid': nn.Sigmoid(),
                'softmax': nn.Softmax()
            })
    return act[activation]


class InceptionBlk(nn.Module):
    '''
    MMCNN的inception模块
    '''
    def __init__(self, length, in_channels, out_channels, kernel_size, stride, activation):
        super(InceptionBlk, self).__init__()
        self.branch1 = nn.Sequential(Conv1dPaddingSame(in_channels, out_channels[0],
                                                       kernel_size[0], stride=stride),
                                     nn.BatchNorm1d(out_channels[0]),
                                     activations(activation))

        self.branch2 = nn.Sequential(Conv1dPaddingSame(in_channels, out_channels[1],
                                                       kernel_size[1], stride=stride),
                                      nn.BatchNorm1d(out_channels[1]),
                                     activations(activation))

        self.branch3 = nn.Sequential(Conv1dPaddingSame(in_channels, out_channels[2],
                                                       kernel_size[2], stride=stride),
                                     nn.BatchNorm1d(out_channels[2]),
                                     activations(activation))

        self.branch4 = nn.Sequential(MaxPool1dPaddingSame(kernel_size[3], stride),
                                     Conv1dPaddingSame(in_channels, out_channels[3],
                                                       kernel_size=1, stride=1),
                                     nn.BatchNorm1d(out_channels[3]),
                                     activations(activation))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        out = torch.cat((branch1, branch2, branch3, branch4), axis=1)
        return out


class ResBlk(nn.Module):
    '''
    MMCNN的resnet模块
    '''
    def __init__(self, length, in_channels, out_channels, kernel_size, activation, dropout_p):
        super(ResBlk, self).__init__()
        stride = 1
        self.branch1 = nn.Sequential(Conv1dPaddingSame(in_channels[0], out_channels[0],
                                                       kernel_size, stride=stride),
                                     nn.BatchNorm1d(out_channels[0]),
                                     activations(activation),

                                     Conv1dPaddingSame(in_channels[1], out_channels[1],
                                                       kernel_size, stride=stride),
                                     nn.BatchNorm1d(out_channels[1]),
                                     activations(activation),

                                     Conv1dPaddingSame(in_channels[2], out_channels[2],
                                                       kernel_size, stride=stride),
                                     nn.BatchNorm1d(out_channels[2]))

        self.branch2 = nn.Sequential(Conv1dPaddingSame(in_channels[0], out_channels[2],
                                                       kernel_size=1, stride=stride),
                                     nn.BatchNorm1d(out_channels[2]))

        self.act = activations(activation)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        out = self.dropout(self.act(branch1 + branch2))
        return out


class SEnet(nn.Module):
    '''
    MMCNN的senet模块
    '''
    def __init__(self, num_channels, activation, ratio):
        super(SEnet, self).__init__()

        num_channels_reduced = num_channels // ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.act1 = activations(activation)
        self.sigmoid = activations('sigmoid')

    def forward(self, x):
        batch_size, num_channels, length = x.size()
        squeeze_tensor = torch.squeeze(F.avg_pool1d(x, length))

        fc_out_1 = self.act1(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        out = torch.mul(x, fc_out_2.view(a, b, 1))
        return out


class MMCNN(nn.Module):
    def __init__(self, channels):
        super(MMCNN, self).__init__()
        self.activation = 'elu'
        self.in_channels = channels
        self.length = 1000
        # the parameter of the first part :EEG Inception block
        self.inception_out_channels = [16, 16, 16, 16]
        self.inception_kernel_size = [[5, 10, 15, 10],
                                      [40, 45, 50, 100],
                                      [60, 65, 70, 100],
                                      [80, 85, 90, 100],
                                      [160, 180, 200, 180]]
        self.inception_stride = [2, 4, 4, 4, 16]
        self.first_maxpooling_size = 4
        self.first_maxpooling_stride = 4
        self.batch_norm_features = sum(self.inception_out_channels)
        self.dropout_p = 0.2
        # the parameter of the second part :Residual block
        self.res_block_in_channels = [64, 16, 16]
        self.res_block_out_channels = [16, 16, 16]
        self.res_block_kernel_size = [7, 7, 7, 7, 7]
        # the parameter of the third part :SE block
        self.se_block_kernel_size = 16
        self.se_ratio = 8
        self.second_maxpooling_size = [4, 3, 3, 3, 2]
        self.second_maxpooling_stride = [4, 3, 3, 3, 2]

        # EIN-A
        self.branch_a = nn.Sequential(InceptionBlk(self.length, self.in_channels,
                                                   self.inception_out_channels,
                                                   self.inception_kernel_size[0],
                                                   self.inception_stride[0],
                                                   self.activation),
                                      MaxPool1dPaddingSame(self.first_maxpooling_size,
                                                           self.first_maxpooling_stride),
                                      nn.BatchNorm1d(self.batch_norm_features),
                                      nn.Dropout(self.dropout_p),
                                      ResBlk(self.length,
                                             self.res_block_in_channels,
                                             self.res_block_out_channels,
                                             self.res_block_kernel_size[0],
                                             self.activation,
                                             self.dropout_p),
                                      SEnet(self.se_block_kernel_size,
                                            self.activation,
                                            self.se_ratio),
                                      MaxPool1dPaddingSame(self.second_maxpooling_size[0],
                                                           self.second_maxpooling_stride[0]))

        # EIN-B
        self.branch_b = nn.Sequential(InceptionBlk(self.length, self.in_channels,
                                                   self.inception_out_channels,
                                                   self.inception_kernel_size[1],
                                                   self.inception_stride[1],
                                                   self.activation),
                                      MaxPool1dPaddingSame(self.first_maxpooling_size,
                                                           self.first_maxpooling_stride),
                                      nn.BatchNorm1d(self.batch_norm_features),
                                      nn.Dropout(self.dropout_p),
                                      ResBlk(self.length,
                                             self.res_block_in_channels,
                                             self.res_block_out_channels,
                                             self.res_block_kernel_size[1],
                                             self.activation,
                                             self.dropout_p),
                                      SEnet(self.se_block_kernel_size,
                                            self.activation,
                                            self.se_ratio),
                                      MaxPool1dPaddingSame(self.second_maxpooling_size[1],
                                                           self.second_maxpooling_stride[1]))
        # EIN-C
        self.branch_c = nn.Sequential(InceptionBlk(self.length,
                                                   self.in_channels,
                                                   self.inception_out_channels,
                                                   self.inception_kernel_size[2],
                                                   self.inception_stride[2],
                                                   self.activation),
                                      MaxPool1dPaddingSame(self.first_maxpooling_size,
                                                           self.first_maxpooling_stride),
                                      nn.BatchNorm1d(self.batch_norm_features),
                                      nn.Dropout(self.dropout_p),
                                      ResBlk(self.length,
                                             self.res_block_in_channels,
                                             self.res_block_out_channels,
                                             self.res_block_kernel_size[2],
                                             self.activation,
                                             self.dropout_p),
                                      SEnet(self.se_block_kernel_size,
                                            self.activation,
                                            self.se_ratio),
                                      MaxPool1dPaddingSame(self.second_maxpooling_size[2],
                                                           self.second_maxpooling_stride[2]))
        # EIN-D
        self.branch_d = nn.Sequential(InceptionBlk(self.length,
                                                   self.in_channels,
                                                   self.inception_out_channels,
                                                   self.inception_kernel_size[3],
                                                   self.inception_stride[3],
                                                   self.activation),
                                      MaxPool1dPaddingSame(self.first_maxpooling_size,
                                                           self.first_maxpooling_stride),
                                      nn.BatchNorm1d(self.batch_norm_features),
                                      nn.Dropout(self.dropout_p),
                                      ResBlk(self.length,
                                             self.res_block_in_channels,
                                             self.res_block_out_channels,
                                             self.res_block_kernel_size[3],
                                             self.activation,
                                             self.dropout_p),
                                      SEnet(self.se_block_kernel_size,
                                            self.activation,
                                            self.se_ratio),
                                      MaxPool1dPaddingSame(self.second_maxpooling_size[3],
                                                           self.second_maxpooling_stride[3]))
        # EIN-E
        self.branch_e = nn.Sequential(InceptionBlk(self.length,
                                                   self.in_channels,
                                                   self.inception_out_channels,
                                                   self.inception_kernel_size[4],
                                                   self.inception_stride[4],
                                                   self.activation),
                                      MaxPool1dPaddingSame(self.first_maxpooling_size,
                                                           self.first_maxpooling_stride),
                                      nn.BatchNorm1d(self.batch_norm_features),
                                      nn.Dropout(self.dropout_p),
                                      ResBlk(self.length,
                                             self.res_block_in_channels,
                                             self.res_block_out_channels,
                                             self.res_block_kernel_size[4],
                                             self.activation,
                                             self.dropout_p),
                                      SEnet(self.se_block_kernel_size,
                                            self.activation,
                                            self.se_ratio),
                                      MaxPool1dPaddingSame(self.second_maxpooling_size[4],
                                                           self.second_maxpooling_stride[4]))

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(self.dropout_p)
        self.fc = nn.Linear(1648, 3)
        # self.act = activations('softmax')

    def forward(self, x):
        x1 = self.branch_a(x)
        x2 = self.branch_b(x)
        x3 = self.branch_c(x)
        x4 = self.branch_d(x)
        x5 = self.branch_e(x)
        out = torch.cat((x1, x2, x3, x4, x5), axis=2)
        out = self.flatten(out)
        out = self.fc(self.dropout(out))

        return out

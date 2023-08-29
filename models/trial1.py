import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x, x_dim):
        return self.lambd(x, x_dim)
                
           
class Trial1_Module(nn.Module):
    expansion = 1
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        rank: int = 2,
        sparsity: float = None,
        degree: float = None,
    ):
        super(Trial1_Module, self).__init__()
        
        self.lr_fc1 = LambdaLayer(lambda x, x_dim: 
                                    nn.Linear(in_features=x_dim[1]*x_dim[2]*x_dim[3],
                                    out_features=rank, 
                                    bias=False).to('cuda')(x))
        self.lr_fc2 = LambdaLayer(lambda x, x_dim:
                                    nn.Linear(in_features=rank,
                                    out_features=x_dim[1]*x_dim[2]*x_dim[3], 
                                    bias=False).to('cuda')(x))
        
        self.sao_conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  padding=1,
            
                                  bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        #self.bn2 = nn.BatchNorm2d(out_channels)
        #37.7
        #36.1
    def forward(self, x):
        #x_conv = self.bn1(self.sao_conv(x))
        x_conv = self.bn1(self.sao_conv(x))
        x_fc = x.view(x.size(0), -1)
        x_fc = self.lr_fc1(x_fc, x.size())
        x_fc = self.lr_fc2(x_fc, x_conv.size())
        x_fc = x_fc.view(x_conv.size())
        return F.relu(self.bn1(x_fc + x_conv))


class Trial1(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(Trial1, self).__init__()
        
        self.in_planes = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
def trial1():
    return Trial1(Trial1_Module, [3, 3, 3])

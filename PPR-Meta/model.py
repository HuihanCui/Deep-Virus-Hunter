import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3, padding=0)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

class Conv1DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv1DLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
    
    def forward(self, x):
        x = F.relu(self.conv(x))
        return x

class BiPathCNN(nn.Module):
    def __init__(self):
        super(BiPathCNN, self).__init__()
        
        # Base Path (BOH)
        self.base_conv1 = Conv1DBlock(in_channels=4, out_channels=64, kernel_size=6, dropout_rate=0.25)
        self.base_conv2 = Conv1DBlock(in_channels=64, out_channels=128, kernel_size=3, dropout_rate=0.25)
        self.base_conv3 = Conv1DLayer(in_channels=128, out_channels=256, kernel_size=3)
        self.base_global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Codon Path (COH)
        self.codon_conv1 = Conv1DBlock(in_channels=64, out_channels=64, kernel_size=6, dropout_rate=0.25)
        self.codon_conv2 = Conv1DBlock(in_channels=64, out_channels=128, kernel_size=3, dropout_rate=0.25)
        self.codon_conv3 = Conv1DLayer(in_channels=128, out_channels=256, kernel_size=3)
        self.codon_global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(2 * 256, 512)
        self.fc2 = nn.Linear(512, 3)
        
    def forward(self, base_input, codon_input):
        # Base Path Forward
        base_x = self.base_conv1(base_input)
        base_x = self.base_conv2(base_x)
        base_x = self.base_conv3(base_x)
        base_x = self.base_global_avg_pool(base_x)
        base_x = base_x.view(base_x.size(0), -1)  # Flatten
        
        # Codon Path Forward
        codon_x = self.codon_conv1(codon_input)
        codon_x = self.codon_conv2(codon_x)
        codon_x = self.codon_conv3(codon_x)
        codon_x = self.codon_global_avg_pool(codon_x)
        codon_x = codon_x.view(codon_x.size(0), -1)  # Flatten
        
        # Concatenate
        x = torch.cat((base_x, codon_x), dim=1)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Softmax Activation
        return F.softmax(x, dim=1)

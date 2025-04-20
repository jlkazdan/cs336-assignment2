import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

class ToyModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        print(x.dtype)
        x = self.ln(x)
        print(x.dtype)
        x = self.fc2(x)
        print(x.dtype)
        return x

def main():
    model = ToyModel(100, 100).to('cuda')
    with autocast():
        x = torch.randn(100, 100).to('cuda')
        model(x)
        

if __name__ == '__main__':
    main()

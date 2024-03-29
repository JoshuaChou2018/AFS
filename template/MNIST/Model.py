import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim_in*dim_in*3, dim_hidden)
        self.bn1 = nn.BatchNorm1d(dim_hidden)
        self.relu1 = nn.ReLU(dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_hidden)
        self.bn2 = nn.BatchNorm1d(dim_hidden)
        self.relu2 = nn.ReLU(dim_hidden)
        self.fc3 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, 28*28*3)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1)

def get_model():
    return MLP(28, 256, 10)

def get_teacher_model():
    return MLP(28, 256, 10)

def get_student_model():
    return MLP(28, 64, 10)

if __name__ == '__main__':
    teacher = get_teacher_model().to('cuda:0')
    student = get_student_model().to('cuda:0')
    summary(teacher, (28,28,3))
    summary(student, (28,28,3))
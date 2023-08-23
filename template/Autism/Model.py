import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.relu1 = nn.ReLU(dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, 25)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

def get_model():
    return MLP(25, 128, 2)

def get_teacher_model():
    return MLP(25, 128, 2)

def get_student_model():
    return MLP(25, 32, 2)

if __name__ == '__main__':
    teacher = get_teacher_model().to('cuda:0')
    student = get_student_model().to('cuda:0')
    summary(teacher, (1,25))
    summary(student, (1,25))
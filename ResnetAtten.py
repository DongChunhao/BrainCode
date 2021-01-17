import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
from torch import optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from imblearn.over_sampling import SMOTE
from torch.utils.data import Dataset, DataLoader
import torch
from torchkeras import Model
import pandas as pd
np.random.seed(1337)
seed=1337
class Attention(Model):
    def __init__(self, in_dim=128):
        super(Attention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64,in_dim),
            nn.Sigmoid()
        )
        self.dense = nn.Linear(in_dim, in_dim)
    def forward(self, x):
        #  b c d/
        score = self.mlp(x)#b c 1
        # score = F.softmax(score, dim=1)#b c 1
        x = x * score# b 1 d
        x = self.dense(x)
        return x
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class ResidualBlock(Model):
    def __init__(self, inchannel, outchannel,reduction=16,stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv1d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(outchannel)
        )
        self.attn=Attention(outchannel)
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outchannel)
            )
    def forward(self, x):
        out = self.left(x)
        # score = self.mlp(out)  # b c 1
        # score = F.softmax(score, dim=1)  # b c 1
        # out = (out * score).sum(1)  # b 1 d
        out = out.transpose(1, 2)
        out = self.attn(out)
        out = out.transpose(1, 2)
        # out = out.squeeze(1)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class ResNet(Model):
    def __init__(self, ResidualBlock, num_classes=2):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=1, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)
        self.pool=nn.AdaptiveAvgPool1d(1)
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    def forward(self, x):
        x = x.transpose(1,2)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
def ResNet18():
    return ResNet(ResidualBlock)
class DCDataset(Dataset):
    def __init__(self, data, label):
        super(DCDataset, self).__init__()
        self.data = data
        self.label = label
    def __getitem__(self, item):
        x = self.data[item]
        y = self.label[item]
        x = torch.tensor(x)
        y = torch.tensor(y).float()
        return [x, y]
    def __len__(self):
        return len(self.data)
def get_loader():
    #加载数据集

    df = pd.read_csv("XXX/XXX.csv")
    # df = pd.read_csv("D:\BrainMethod_dataset\BrainMethodtest.csv")
    x, y = df.iloc[:, 1:-1], df.iloc[:, -1]
    smo = SMOTE(sampling_strategy=1, random_state=42)  # ratio={1: 10000 },
    x, y = smo.fit_sample(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    x_train, x_valid, y_train, y_vaild = train_test_split(x_train, y_train, test_size=0.1, random_state=0)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_valid= x_valid.astype('float32')
    # x_train /= 255
    # x_test /= 255
    # x_valid/=255
    # X_train = x_train.astype('int')
    # X_test = x_test.astype('int')

    x_train = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1], 1))
    x_valid = x_valid.values.reshape((x_valid.shape[0], x_valid.shape[1], 1))
    # y_train = np_utils.to_categorical(y_train, num_classes=2)
    # y_test = np_utils.to_categorical(y_test, num_classes=2)
    # y_vaild = np_utils.to_categorical(y_vaild, num_classes=2)
    onehot = OneHotEncoder(sparse=False)
    y_train = onehot.fit_transform(y_train.values.reshape(len(y_train), 1))
    y_test = onehot.fit_transform(y_test.values.reshape(len(y_test), 1))
    y_vaild = onehot.fit_transform(y_vaild.values.reshape(len(y_vaild), 1))

    print(x_train.shape)
    print(y_train.shape)
    # train = TensorDataset(x_train, y_train)
    # test = TensorDataset(x_test, y_test)
    train = DCDataset(x_train, y_train)
    test = DCDataset(x_test, y_test)
    x_valid = DCDataset(x_valid, y_vaild)

    train = DataLoader(train, batch_size=100, shuffle=True, num_workers=0)
    test = DataLoader(test, batch_size=100, shuffle=True, num_workers=0)
    valid = DataLoader(x_valid, batch_size=100,shuffle=True,num_workers=0)
    return train,test,valid
def metric():
    def to_label(val_targ, val_predict):
        val_targ = val_targ.detach()
        _, val_targ = torch.max(val_targ, 1)
        val_predict = val_predict.detach()
        _, val_predict = torch.max(val_predict, 1)
        return val_targ, val_predict
    def prec(val_targ, val_predict):
        val_targ, val_predict = to_label(val_targ, val_predict)
        return precision_score(val_targ, val_predict, pos_label=1, average='weighted')
    def recall(val_targ, val_predict):
        val_targ, val_predict = to_label(val_targ, val_predict)
        return recall_score(val_targ, val_predict, pos_label=1, average='weighted')
    def f1(val_targ, val_predict):
        val_targ, val_predict = to_label(val_targ, val_predict)
        return f1_score(val_targ, val_predict, pos_label=1, average='weighted')
    def acc(val_targ, val_predict):
        val_targ, val_predict = to_label(val_targ, val_predict)
        return accuracy_score(val_targ, val_predict)
    return prec, recall, f1, acc
def main():
    model = ResNet18()
    # model = ResNet50(input_shape=(28, 28, 1), classes=10)
    model.summary(input_shape=(1,1))
    train,test,valid = get_loader()
    prec, recall, f1, acc = metric()
    model.compile(loss_func=nn.BCEWithLogitsLoss(), optimizer=optim.Adam(model.parameters(), lr=0.01),
                  metrics_dict={"prec":prec, "recall":recall, "f1":f1, "acc":acc})
    model.fit(epochs=50, dl_train=train, dl_val=valid, log_step_freq=200)
    torch.save(model.state_dict(), "model_1.pkl")
    # ckpt=torch.load("model_1.pkl")
    # model.load_state_dict(ckpt)
    print(model.evaluate(test))
    # model.predict(test)
    # loss, accuracy = model.evaluate(dl_val=test)
    # print('loss：'+loss,'accuracy:'+accuracy)
if __name__ == '__main__':
    main()















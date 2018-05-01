import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F

from a import load_data
from b import save_to_file
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

use_cuda = torch.cuda.is_available()

cfg = {
    'VGG11_modified': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGGhc': [128, 128, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGGhc2': [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256],
    'VGG13_modified': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG16_modified': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'VGG19_modified': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 20)
#         self.classifier = nn.Linear(256, 20)

    def forward(self, x):
        x = x.view(x.size(0), 1, 28, 28)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = F.log_softmax(out, dim=0)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                layers += [nn.Dropout(p=0.2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class NN(nn.Module):
    def __init__(self, input_size, num_units, label_size):
        super(NN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, num_units),
            nn.Sigmoid(),
            nn.Linear(num_units, label_size)
        )

    def forward(self, x):
        x = self.layer(x)
        x = F.log_softmax(x, dim=0)
        return x


class CNN(nn.Module):
    def __init__(self, out_channels, kernel_size, fcc):
        super(CNN, self).__init__()
        self.input_droput = nn.Dropout(p=0.2)
        self.conv1 = nn.Sequential(  # batch x 1 x 28 x 28
            nn.Conv2d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.ReLU(),  # batch x 10 x 28 x 28
            nn.MaxPool2d(kernel_size=2),  # batch x 10 x 14 x 14
            nn.Dropout(p=0.2),
        )
        self.fcc = nn.Linear(out_channels * 14 * 14, fcc)
        self.out = nn.Linear(fcc, 20)

    def forward(self, x):
        # print(x.size())
        x = x.view(x.size(0), 1, 28, 28)
        # print(x.size())
        x = self.input_droput(x)
        # print(x.size())
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.fcc(x)
        # print(x.size())
        x = self.out(x)
        x = F.log_softmax(x, dim=0)
        # print(x.size())
        return x


def gen_index_for_labels(labels):
    labels = list(set(labels))
    l2i = {}
    i2l = {}
    for label in labels:
        if label not in l2i:
            l2i[label] = len(l2i)
            i2l[l2i[label]] = label
    return l2i, i2l


def lables_2_index(labels, l2i):
    indices = []
    for label in labels:
        indices.append(l2i[label])
    return np.array(indices)


def index_2_labels(indices, i2l):
    labels = []
    for index in indices:
        labels.append(i2l[index])
    return np.array(labels)


def train(model, model_file, epochs=100, batch_size=1000, dev=True):
    if os.path.exists(model_file):
        print("Loading Model: %s" % (model_file))
        model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    if use_cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.NLLLoss()  # taking softmax and log likelihood
    if dev:
        dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    else:
        dataset = torch.utils.data.TensorDataset(org_train_data, org_train_labels)

    train_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    best_dev = 0
    for epoch in range(epochs):
        model.train()
        gold, pred, epoch_loss = [], [], []
        for x, y in tqdm(train_loader):
            b_x = Variable(x)
            b_y = Variable(y)

            if use_cuda:
                b_x = b_x.cuda()
                b_y = b_y.cuda()

            outputs = model(b_x)  # output of the cnn
            loss = loss_func(outputs, b_y)  # loss
            optimizer.zero_grad()  # clearing gradients
            loss.backward()  # backpropogation
            optimizer.step()  # applygradients

            cur_pred = torch.max(outputs, dim=1)[1].data.cpu().numpy().tolist()
            cur_gold = b_y.data.cpu().numpy().tolist()
            gold.extend(cur_gold)
            pred.extend(cur_pred)
            epoch_loss.extend(loss)

        accuracy = accuracy_score(gold, pred) * 100
        loss = np.mean(np.array(epoch_loss))
        if dev:
            dev_pred = predict(model, model_file, dev_data, batch_size=100, load=False)
            dev_gold = dev_labels.cpu().numpy().tolist()
            dev_accuracy = accuracy_score(dev_gold, dev_pred) * 100

            if dev_accuracy > best_dev:
                print("Saving Model")
                torch.save(model.state_dict(), model_file)
                best_dev = dev_accuracy
                improved = "*"
            else:
                improved = ""

            print('Epoch: ', epoch, '| train loss: %.4f' % loss, '| train_accuracy: %.2f' % accuracy, '| dev_accuracy: %.2f%s' % (dev_accuracy, improved), "\n")
        else:
            print("Saving Model")
            torch.save(model.state_dict(), model_file)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss, '| train_accuracy: %.2f' % accuracy, "\n")


def predict(model, model_file, data, batch_size=100, load=True):

    if load:
        if os.path.exists(model_file):
            print("Loading Model: %s" % (model_file))
            model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
        else:
            print("Model does not exist")
            return []

    if use_cuda:
        model = model.cuda()

    model.eval()

    pred = []
    for i in range(0, len(data), batch_size):
        x = data[i:i + batch_size]
        b_x = Variable(x, volatile=True)
        if use_cuda:
            b_x = b_x.cuda()

        outputs = model(b_x)  # output of the cnn
        cur_pred = torch.max(outputs, dim=1)[1].data.cpu().numpy().tolist()
        pred.extend(cur_pred)

    return pred


def part_c(hidden_units=1000, epochs=200, model_file="nn.model", output_file="out.txt", dev=True):
    model = NN(len(train_data[0]), hidden_units, len(l2i))
    print(model)
    train(model, model_file, epochs=epochs, batch_size=100, dev=dev)
    pred = predict(model, model_file, test_data, batch_size=100)
    save_to_file(index_2_labels(pred, i2l), output_file)


def part_d(out_channels=10, kernel_size=3, fcc=100, epochs=200, model_file="cnn.model", output_file="out.txt", dev=True):
    model = CNN(out_channels, kernel_size, fcc)
    print(model)
    train(model, model_file, epochs=epochs, batch_size=100, dev=dev)
    pred = predict(model, model_file, test_data, batch_size=100)
    save_to_file(index_2_labels(pred, i2l), output_file)


train_data, train_labels = load_data("dataset/train")
train_data = (train_data - train_data.mean(axis=0)) / train_data.std(axis=0)
test_data, test_labels = load_data("dataset/test")
test_data = (test_data - test_data.mean(axis=0)) / test_data.std(axis=0)
l2i, i2l = gen_index_for_labels(train_labels)
train_labels = lables_2_index(train_labels, l2i)

org_train_data = np.array(train_data)
org_train_labels = np.array(train_labels)
# splitting into train and dev
train_data, dev_data, train_labels, dev_labels = train_test_split(train_data, train_labels, random_state=64, test_size=0.20)

# Converting to torch variables
train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
train_labels = torch.from_numpy(train_labels).type(torch.LongTensor)
org_train_data = torch.from_numpy(org_train_data).type(torch.FloatTensor)
org_train_labels = torch.from_numpy(org_train_labels).type(torch.LongTensor)

dev_data = torch.from_numpy(dev_data).type(torch.FloatTensor)
dev_labels = torch.from_numpy(dev_labels).type(torch.LongTensor)

test_data = torch.from_numpy(test_data).type(torch.FloatTensor)

if use_cuda:
    train_data = train_data.cuda()
    train_labels = train_labels.cuda()
    dev_data = dev_data.cuda()
    dev_labels = dev_labels.cuda()
    test_data = test_data.cuda()
    org_train_data = org_train_data.cuda()
    org_train_labels = org_train_labels.cuda()

# part c
hidden_units = 1000
model_file = "nn_%d.model" % hidden_units
output_file = "nn_out_%d.txt" % hidden_units
part_c(hidden_units, epochs=200, model_file=model_file, output_file=output_file)


# part d
out_channels = 64
kernel_size = 5
fcc = 50
model_file = "cnn_%d_%d_%d.model" % (out_channels, kernel_size, fcc)
output_file = "cnn_out_%d_%d_%d.txt" % (out_channels, kernel_size, fcc)
part_d(out_channels=out_channels, kernel_size=kernel_size, fcc=fcc, epochs=200, model_file=model_file, output_file=output_file)


# competition
model = VGG("VGG13_modified")
model_file = "vgg13v2.model"
pred = predict(model, model_file, test_data, batch_size=100)
save_to_file(index_2_labels(pred, i2l), output_file)

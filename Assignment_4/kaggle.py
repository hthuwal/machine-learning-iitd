# Model file is at: https://drive.google.com/open?id=1YxAjua8Re-x7WBJgciTQbW_QohTqEVG5
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F

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

best_order = ['chair', 'skyscraper', 'banana', 'parrot', 'laptop', 'hat', 'eyeglasses', 'violin', 'spider', 'flashlight', 'penguin', 'nose', 'hand', 'trombone', 'harp', 'keyboard', 'snowman', 'foot', 'pig', 'bulldozer']


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


def load_data(folder):
    labels = []
    data = None
    files = os.listdir(folder)
    for file in files:
        base, extension = os.path.splitext(file)
        if extension == ".npy":
            temp = np.load(os.path.join(folder, file))
        labels += [base] * (temp.shape[0])

        if data is None:
            data = np.array(temp)
        else:
            data = np.append(data, temp, axis=0)

    return data, np.array(labels)


def save_to_file(pred, file):
    with open(file, "w") as f:
        f.write("ID,CATEGORY\n")
        for i in range(len(pred)):
            f.write("%d,%s\n" % (i, pred[i]))


def gen_index_for_labels(labels):
    labels = list(set(labels))
    labels.sort()
    labels = best_order
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
    for i in tqdm(range(0, len(data), batch_size)):
        x = data[i:i + batch_size]
        b_x = Variable(x, volatile=True)
        if use_cuda:
            b_x = b_x.cuda()

        outputs = model(b_x)  # output of the cnn
        cur_pred = torch.max(outputs, dim=1)[1].data.cpu().numpy().tolist()
        pred.extend(cur_pred)
    return pred


# competition
model = VGG("VGG13_modified")

msg = """
        To Train:
        python3 kaggle.py -t model_file_name

        To Predict:
        python3 kaggle.py -p model_file_name output_file_name
    """

if len(sys.argv) < 3:
    print(msg)

else:
    what_to_do = sys.argv[1]
    model_file = sys.argv[2]

    if what_to_do == "-t":
        try:
            print("Loading Training data")
            train_data, train_labels = load_data("dataset/train")
        except Exception as e:
            print("Code terminated with exception: \n", e)
            print("\n Please make sure the directories 'dataset/train/' exist")
            sys.exit(0)

        train_data = (train_data - train_data.mean(axis=0)) / train_data.std(axis=0)
        l2i, i2l = gen_index_for_labels(train_labels)
        train_labels = lables_2_index(train_labels, l2i)

        org_train_data = np.array(train_data)
        org_train_labels = np.array(train_labels)

        train_data, dev_data, train_labels, dev_labels = train_test_split(train_data, train_labels, random_state=64, test_size=0.20)

        # Converting to torch variables
        train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
        train_labels = torch.from_numpy(train_labels).type(torch.LongTensor)
        org_train_data = torch.from_numpy(org_train_data).type(torch.FloatTensor)
        org_train_labels = torch.from_numpy(org_train_labels).type(torch.LongTensor)

        dev_data = torch.from_numpy(dev_data).type(torch.FloatTensor)
        dev_labels = torch.from_numpy(dev_labels).type(torch.LongTensor)

        if use_cuda:
            train_data = train_data.cuda()
            train_labels = train_labels.cuda()
            org_train_data = org_train_data.cuda()
            org_train_labels = org_train_labels.cuda()

        print("Model File: %s" % (model_file))
        print("Training...")
        train(model, model_file, epochs=10, batch_size=64, dev=False)

    elif what_to_do == "-p":
        if len(sys.argv) < 4:
            print(msg)
            sys.exit(0)
        output_file = sys.argv[3]

        try:
            print("Loading Test data")
            test_data, test_labels = load_data("dataset/test")
        except Exception as e:
            print("Code terminated with exception: \n", e)
            print("\n Please make sure the directories 'dataset/test/' exist")
            sys.exit(0)

        test_data = (test_data - test_data.mean(axis=0)) / test_data.std(axis=0)
        test_data = torch.from_numpy(test_data).type(torch.FloatTensor)

        if use_cuda:
            test_data = test_data.cuda()

        print("Model File: %s, output_file: %s" % (model_file, output_file))
        print("Predicting...")
        pred = predict(model, model_file, test_data, batch_size=100)
        save_to_file(index_2_labels(pred, best_order), output_file)

    else:
        print("Invalid Input")
        print(msg)

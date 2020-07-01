import os
import torch
import pickle
import numpy as np 
from mmcnn import MMCNN
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Util():
    def __init__(self):
        # 记录每次训练过程中的几个关键值
        self.train_loss_log = []
        self.val_loss_log = []
        self.val_acc_log = []

    def train(self, model, device, train_loader, optimizer, epoch):
        criterion = nn.CrossEntropyLoss()  
        model.train()
        train_loss = 0.0
        for step, (data, label) in enumerate(train_loader):
            train_data, train_label = data.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(train_data)
            loss = criterion(out, train_label)
            loss.backward()
            optimizer.step()
            train_loss += loss.data / len(train_data)
        self.train_loss_log.append(float(train_loss.data))


    def val(self, model, device, test_loader, epoch):
        model.eval()
        predict = []
        label = []
        val_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for step, (data, labels) in enumerate(test_loader):
                val_data, val_label = data.to(device), labels.to(device)
                out = model(val_data)
                loss = criterion(out, val_label)
                val_loss += loss.data / len(val_data)  # len(val_data)

                output = torch.max(out.to('cpu'), 1)[1].numpy().tolist()
                val_label = val_label.to('cpu').numpy().tolist()
                predict = predict + output
                label = label + val_label

        val_acc = accuracy_score(predict, label)
        self.val_loss_log.append(float(val_loss.data))
        self.val_acc_log.append(float(val_acc))
        print("Epoch {} Average Validation loss: {} Validation acc: {}".format(
            epoch, val_loss, val_acc))
        return val_loss, val_acc


    def test(self, model, device, test_loader):
        model.eval()
        predict = []
        label = []
        with torch.no_grad():
            for step, (data, labels) in enumerate(test_loader):
                test_data, test_label = data.to(device), labels.to(device)
                test_label = test_label.unsqueeze(1)
                out = model(test_data)

                output = torch.max(out.to('cpu'), 1)[1].numpy().tolist()
                test_label = test_label.to('cpu').numpy().tolist()
                predict = predict + output
                label = label + test_label

        accuracy = accuracy_score(predict, label)
        f1 = f1_score(label, predict, average='micro')
        maxtrix = confusion_matrix(label, predict)
        # sensitivity = tp / (tp + fn)
        # specificity = tn / (fp + tn)
        print("实验准确率为: {}  F1: {}".format(accuracy, f1))
        return accuracy, f1, maxtrix


    def setup_seed(self, seed):
        '''
        固定训练过程的随机量
        '''
        if seed == None:
            pass
        else:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            # random.seed(seed)
            torch.backends.cudnn.deterministic = True


    def get_dataloader(self, Input_pickle):
        with open(Input_pickle, 'rb') as f:
            train_data, train_label, test_data, test_label = pickle.load(f)


        train_data = torch.tensor(train_data, dtype=torch.float)
        test_data = torch.tensor(test_data, dtype=torch.float)
        print("train_data.shape: {} test_data.shape: {}".format(
            train_data.shape, test_data.shape))
        train_label = torch.tensor(train_label, dtype=torch.long)
        test_label = torch.tensor(test_label, dtype=torch.long)
        #生成dataloader
        train_data = TensorDataset(train_data, train_label)
        test_data = TensorDataset(test_data, test_label)
        train_loader = DataLoader(train_data, shuffle=True,
                                batch_size=64)
        test_loader = DataLoader(test_data, shuffle=True,
                                batch_size=64)

        return train_loader, test_loader


    def run(self, input_pickle, lr, epochs, early_stop, seed):
        self.setup_seed(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader, test_loader = self.get_dataloader(input_pickle)

        model = MMCNN(channels=62).to(device)
        best_acc = 0
        best_loss = float('inf')
        plateau_period = 0
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
        # 用于调整学习率
        scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                    patience=10, verbose=True,
                                    min_lr=0.000001, factor=0.5)
        for epoch in range(1, epochs + 1):
            self.train(model, device, train_loader, optimizer, epoch)
            val_loss, val_acc = self.val(model, device, test_loader, epoch)
            scheduler.step(val_loss)
            if best_acc < val_acc:
                best_acc = val_acc
                best_loss = val_loss
                plateau_period = 0
                torch.save(model, 'best.pth')
            elif best_acc >= val_acc:
                plateau_period += 1
                # 连续50次验证集误差不再减小就终止训练
                if plateau_period >= early_stop:
                    model = torch.load('best.pth')
                    accuracy, F1, maxtrix = self.test(model, device, test_loader)
                    torch.save(model, 'best.pth')
                    print('''\n Epoch {} >>>>>>>>> Best Validation loss: {} Validation Acc: {} 
                    Test Acc: {} \n Confusion Matrix: \n {} <<<<<<<<<'''.format(epoch, best_loss, best_acc, accuracy, maxtrix))
                    break
        # return accuracy, F1, sen, spe

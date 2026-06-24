# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# For data preprocess
import numpy as np
import csv
import os

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# -------------------- 固定随机种子 --------------------
myseed = 42069
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# -------------------- 工具函数 --------------------
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_learning_curve(loss_record, title=''):
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()

def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()

# -------------------- 数据集类 --------------------
class COVID19Dataset(Dataset):
    def __init__(self, path, mode='train', target_only=False):
        self.mode = mode

        # 读取数据（去掉表头）
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)   # shape: (n_samples, 94) 或 (893, 93)

        # ---------- 标准化：对原始数据的后40列（时间序列特征）进行标准化 ----------
        # 训练/验证集有94列（特征93+目标1），测试集有93列（全是特征）
        # 后40列索引为 40:  (测试集也是 40:)
        data[:, 40:] = (data[:, 40:] - data[:, 40:].mean(axis=0, keepdims=True)) / \
                       (data[:, 40:].std(axis=0, keepdims=True) + 1e-8)   # 加小量防止除零

        # ---------- 特征选择 ----------
        if not target_only:
            feats = list(range(93))          # 使用全部93个特征
        else:
            # 只使用前40个状态特征 + 两个 tested_positive 特征（索引57和75）
            # 注意：训练数据有93个特征（0~92），索引57和75分别对应某天的特征
            feats = list(range(40)) + [57, 75]   # 共42个特征

        if mode == 'test':
            # 测试数据：893 x 93（无目标列）
            data = data[:, feats]                # 选择特征
            self.data = torch.FloatTensor(data)
        else:
            # 训练/验证数据：2700 x 94，最后一列为目标
            target = data[:, -1]                 # 目标值
            data = data[:, feats]                # 选择特征

            # 划分训练集和验证集（每10个样本取1个作为验证集）
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            else:  # 'dev'
                indices = [i for i in range(len(data)) if i % 10 == 0]

            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        self.dim = self.data.shape[1]
        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        if self.mode in ['train', 'dev']:
            return self.data[index], self.target[index]
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)

# -------------------- DataLoader 构建 --------------------
def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only)
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)
    return dataloader

# -------------------- 神经网络模型 --------------------
class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        # 简单三层全连接（可自行调整）
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        return self.criterion(pred, target)

# -------------------- 训练、验证、测试函数 --------------------
def train(tr_set, dv_set, model, config, device):
    n_epochs = config['n_epochs']
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()
        for x, y in tr_set:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            mse_loss = model.cal_loss(pred, y)
            mse_loss.backward()
            optimizer.step()
            loss_record['train'].append(mse_loss.detach().cpu().item())

        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                  .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record

def dev(dv_set, model, device):
    model.eval()
    total_loss = 0
    for x, y in dv_set:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            mse_loss = model.cal_loss(pred, y)
        total_loss += mse_loss.detach().cpu().item() * len(x)
    total_loss = total_loss / len(dv_set.dataset)
    return total_loss

def test(tt_set, model, device):
    model.eval()
    preds = []
    for x in tt_set:
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds

def save_pred(preds, file):
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

# -------------------- 主程序 --------------------
if __name__ == '__main__':
    # ---------- 配置文件路径（请根据实际路径修改） ----------
    tr_path = 'covid.train.csv'   # 训练数据文件
    tt_path = 'covid.test.csv'    # 测试数据文件

    device = get_device()
    os.makedirs('models', exist_ok=True)

    target_only = False   # 是否只使用部分特征（可设为 True 或 False）

    # 超参数配置
    config = {
        'n_epochs': 3000,
        'batch_size': 270,
        'optimizer': 'SGD',
        'optim_hparas': {
            'lr': 0.001,
            'momentum': 0.9
        },
        'early_stop': 200,
        'save_path': 'models/model.pth'
    }

    # 准备数据
    tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], target_only=target_only)
    dv_set = prep_dataloader(tr_path, 'dev', config['batch_size'], target_only=target_only)
    tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], target_only=target_only)

    # 创建模型并训练
    model = NeuralNet(tr_set.dataset.dim).to(device)
    model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)

    # 绘制学习曲线
    plot_learning_curve(model_loss_record, title='deep model')

    # 加载最佳模型并验证
    del model
    model = NeuralNet(tr_set.dataset.dim).to(device)
    ckpt = torch.load(config['save_path'], map_location='cpu')
    model.load_state_dict(ckpt)
    plot_pred(dv_set, model, device)

    # 预测测试集并保存结果
    preds = test(tt_set, model, device)
    save_pred(preds, 'pred.csv')
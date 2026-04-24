import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from expression_dataset import SeqDataset
from SeqRegressionModel import Seq2Scalar
from matplotlib import pyplot as plt
import numpy as np
import collections
import pandas as pd
from util import EarlyStopping
from tqdm import tqdm
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def draw_2d(observed, predicted,save_path,name=""):
    # 计算 Pearson 相关系数
    corr_coef = np.corrcoef(observed, predicted)[0, 1]

    # 计算 2D KDE
    xy = np.vstack([observed, predicted])
    kde = gaussian_kde(xy)

    # 创建网格
    xmin, xmax = observed.min(), observed.max()
    ymin, ymax = predicted.min(), predicted.max()
    xi, yi = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    coords = np.vstack([xi.ravel(), yi.ravel()])
    zi = kde(coords).reshape(xi.shape)

    # 绘图
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pcolormesh(xi, yi, zi, shading='auto', cmap='viridis')
    ax.scatter(observed, predicted, c='purple', s=1, alpha=0.3)  # 叠加散点

    # 设置样式
    ax.set_xlabel('Observed')
    ax.set_ylabel('Predicted')
    ax.set_title('Test set prediction')
    ax.text(0.05, 0.95, f'Pearson r = {corr_coef:.3f}',
            transform=ax.transAxes, fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # 添加颜色条
    cbar = plt.colorbar(ax.collections[0], ax=ax, label='Density')
    cbar.set_label('Density', rotation=270, labelpad=10)

    plt.title('pearson coefficient')
    plt.xlabel('real expression (Log2)')
    plt.ylabel('predict expression (Log2)')

    os.makedirs(f'{save_path}/scatter_fig', exist_ok=True)
    plt.savefig(f'{save_path}/scatter_fig/scatter_{name}.png')


class Seq2ScalarTraining:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Seq2Scalar Training')
        parser.add_argument('-batch_size', type=int, default=128)
        parser.add_argument('-lr', type=float, default=0.001)
        parser.add_argument('-gpu', type=bool, default=True)
        parser.add_argument('-patience', type=int, default=10)
        parser.add_argument('-epoch', type=int, default=100)
        parser.add_argument('-seqL', type=int, default=100)
        parser.add_argument('-mode', type=str, default='DenseLSTM2')
        parser.add_argument('-name', type=str, default='expr_')
        parser.add_argument('-symb', type=str, default='DenseLSTM2')
        parser.add_argument('-save_path', type=str, required=True)


        # 允许给定多个 CSV 路径；若传了多个则自动 concat 成一个文件再用于 SeqDataset。
        # 也支持用户在单个参数里用逗号拼成：例如 -trainset a.csv,b.csv
        parser.add_argument(
            '-trainset',
            type=str,
            nargs='+',
            default=['../data/hypothesis/simulation_data/designed_seq_50000_50_100.csv'],
        )
        parser.add_argument(
            '-validset',
            type=str,
            nargs='+',
            default=['../data/hypothesis/simulation_data/designed_seq_50000_50_100.csv'],
        )
        parser.add_argument(
            '-testset',
            type=str,
            nargs='+',
            default=['../data/hypothesis/simulation_data/designed_seq_5000_75.csv'],
        )
        parser.add_argument('-usepad', type=bool, default=True)
        parser.add_argument('-expr_key', type=str, default="expr")
        parser.add_argument('-seq_key', type=str, default="realB")

        args = parser.parse_args()
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.gpu = args.gpu
        self.patience = args.patience
        self.epoch = args.epoch
        self.seqL = args.seqL
        self.mode = args.mode
        self.name = args.name + self.mode
        self.symb = args.symb
        self.expr_key = args.expr_key
        self.seq_key = args.seq_key

        self.save_path = args.save_path
        os.makedirs(self.save_path, exist_ok=True)

        self.model_ratio = Seq2Scalar(seqL=self.seqL, mode=self.mode)


        def _normalize_paths(maybe_paths):
            # argparse(nargs='+') 会给 list[str]；但也允许用户写成 "a.csv,b.csv" 这种逗号拼接。
            flat = []
            for p in maybe_paths:
                for part in str(p).split(','):
                    part = part.strip()
                    if part:
                        flat.append(part)
            return flat

        def _make_concat_dataset(paths, *, is_train: bool, split_r=None):
            paths = _normalize_paths(paths)
            if len(paths) == 0:
                raise ValueError('Empty dataset paths are not allowed.')

            dsets = []
            for p in paths:
                kwargs = dict(
                    path=p,
                    isTrain=is_train,
                    isGpu=self.gpu,
                    usepad=args.usepad,
                    padlen=self.seqL,
                    expr_key=self.expr_key,
                    seq_key=self.seq_key,
                )
                if split_r is not None:
                    kwargs['split_r'] = split_r
                dsets.append(SeqDataset(**kwargs))
            return ConcatDataset(dsets)

        self.dataset_train = DataLoader(
            dataset=_make_concat_dataset(args.trainset, is_train=True),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.dataset_valid = DataLoader(
            dataset=_make_concat_dataset(args.validset, is_train=False),
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.dataset_test = DataLoader(
            dataset=_make_concat_dataset(args.testset, is_train=True, split_r=1.0),
            batch_size=self.batch_size,
            shuffle=False,
        )

        if self.gpu:
            self.model_ratio=self.model_ratio.cuda()
        self.loss_y = torch.nn.MSELoss()
        self.optimizer_ratio = torch.optim.Adam(self.model_ratio.parameters(), lr=self.lr)

    def training(self):
        trainingLog = collections.OrderedDict()
        trainingLog['train_loss'] = []
        trainingLog['test_coefs'] = []
        trainingLog['test_loss'] = []
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, path=f'{self.save_path}/{self.symb}_{self.name}.pth', stop_order='max')
        for ei in range(self.epoch):
            train_loss_y = 0
            train_num_y = 0
            test_loss_y = 0
            test_num = 0
            self.model_ratio.train()
            print('Training iters')
            for trainLoader in tqdm(self.dataset_train):
                train_data, train_y = trainLoader['x'], trainLoader['z']
                predict = self.model_ratio(train_data)
                predict_y = torch.squeeze(predict)
                loss_y = self.loss_y(predict_y, train_y)
                self.optimizer_ratio.zero_grad()
                loss_y.backward()
                self.optimizer_ratio.step()
                train_loss_y += loss_y
                train_num_y = train_num_y + 1
            test_predict_expr = []
            test_real_expr = []
            self.model_ratio.eval()
            print('Test iters')
            for testLoader in tqdm(self.dataset_valid):
                test_data, test_y = testLoader['x'], testLoader['z']
                predict_y = self.model_ratio(test_data)
                predict_y = torch.squeeze(predict_y.detach())
                predict_y2 = predict_y
                predict_y = predict_y.cpu().float().numpy()
                predict_y = predict_y[:]
                real_y = test_y.cpu().float().numpy()
                for i in range(np.size(real_y)):
                    test_real_expr.append(real_y[i])
                    test_predict_expr.append(predict_y[i])
                test_loss_y += self.loss_y(predict_y2, test_y)
                test_num = test_num + 1
            coefs = np.corrcoef(test_real_expr, test_predict_expr)
            coefs = coefs[0, 1]
            test_coefs = coefs
            trainingLog['test_coefs'].append(coefs)
            trainingLog['train_loss'].append(float(train_loss_y)/train_num_y)
            trainingLog['test_loss'].append(float(test_loss_y)/test_num)
            print('epoch:{}train_loss y:{} test_loss y:{} test_coefs:{}'.format(ei, train_loss_y/train_num_y, test_loss_y/test_num, coefs))
            early_stopping(val_loss=test_coefs, model=self.model_ratio)
            if early_stopping.early_stop:
                print('Early Stopping......')
                break
        predict_ratio = []
        real_ratio = []
        self.model_ratio = torch.load(f'{self.save_path}/{self.symb}_{self.name}.pth',weights_only=False)
        self.model_ratio.eval()
        for testLoader in self.dataset_test:
            test_data, test_y = testLoader['x'], testLoader['z']
            predict_y = self.model_ratio(test_data)
            predict_y = torch.squeeze(predict_y.detach())
            predict_y = predict_y.cpu().float().numpy()
            real_y = test_y.cpu().float().numpy()
            for i in range(np.size(real_y)):
                real_ratio.append(real_y[i])
                predict_ratio.append(predict_y[i])
        ## scatter
        real_expr = np.asarray(real_ratio)
        predict_expr = np.asarray(predict_ratio)
        plt.scatter(real_expr, predict_expr, alpha=0.5, c='brown')
        coefs = np.corrcoef(real_expr, predict_expr)
        coefs = coefs[0, 1]
        np.save(f'{self.save_path}/real_expr_{self.symb}_{self.name}.npy', real_expr)
        np.save(f'{self.save_path}/predict_expr_{self.symb}_{self.name}.npy', predict_expr)
        draw_2d(real_expr,predict_expr,self.save_path,self.name)
        trainingLog = pd.DataFrame(trainingLog)
        trainingLog.to_csv(f'{self.save_path}/training_log_{self.symb}_{self.name}.csv', index=False)


def main():
    analysis = Seq2ScalarTraining()
    analysis.training()

if __name__ == '__main__':
    main()

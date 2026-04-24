import numpy as np
import os
import torch
from dinuc_shuf import shuffle
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, stop_order='min'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.stop_order = stop_order
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss
        if self.stop_order == 'min':
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0
        elif self.stop_order == 'max':
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score > self.best_score + self.delta:
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Updation changed ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, self.path)
        self.val_loss_min = val_loss


class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])

class LoadData(Dataset):

    def __init__(self, data, is_train=True, gpu_ids='0'):
        self.storage = []
        self.gpu_ids = gpu_ids
        for i in range(np.size(data, 0)):
            self.storage.append(data[i])

    def __getitem__(self, item):
        in_seq = transforms.ToTensor()(self.storage[item])
        if len(self.gpu_ids) > 0:
            return in_seq[0, :].float().cuda()
        else:
            return in_seq[0, :].float()

    def __len__(self):
        return len(self.storage)

def get_number(s):
    nums = s.split(' ')
    nums = [int(float(n)) for n in nums]
    return nums


def backbone_one_hot(seq,predictor_kind:int=1):
    seq = ''.join([s.upper() for s in list(seq)])
    if predictor_kind == 0:
        charmap = {'T': 0, 'C': 1, 'G': 2, 'A': 3}
    elif predictor_kind == 1:
        charmap = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    encoded = np.zeros([len(charmap), len(seq)])
    for i in range(len(seq)):
        if seq[i] == 'M':
            encoded[:, i] = np.random.rand(4)
        else:
            if seq[i] != 'N':
                encoded[charmap[seq[i]], i] = 1
    return encoded

def backbone_one_hot_parm(seq):
    seq = ''.join([s.upper() for s in list(seq)])
    charmap = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = np.zeros([len(charmap), len(seq)])
    for i in range(len(seq)):
        if seq[i] == 'M':
            encoded[:, i] = np.random.rand(4)
        else:
            if seq[i] != 'N':
                encoded[charmap[seq[i]], i] = 1
    return encoded

def decode_oneHot(seq):
    keys = ['A', 'T', 'C', 'G', 'M', 'N', 'H', 'Z']
    dSeq = ''
    for i in range(np.size(seq, 1)):
        pos = np.argmax(seq[:, i])
        dSeq += keys[pos]
    return dSeq

# 计算编辑距离
def edit_distance(word1, word2):
    len1 = len(word1)
    len2 = len(word2)
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if word1[i - 1] == word2[j - 1]:
                temp = 0
            else:
                temp = 1
            dp[i][j] = min(dp[i - 1][j - 1] + temp, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2]

# 根据编辑距离计算相似度
def similarity_func(word1, word2):
    res = edit_distance(word1, word2)
    maxLen = max(len(word1), len(word2))
    return 1-res*1.0/maxLen

def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def write_list(l:list, path:str):
    # 将list写入可随时重新读为list的txt文件
    with open(path, 'w') as f:
        for item in l:
            f.write("%s\n" % item)

def read_list(path:str):
    # 读取list.txt文件并返回list
    with open(path, 'r') as f:
        return f.readlines()
    
def dinuc_shuffle(sequence:str):
    SEQ_ALPHABET = np.array(["A","C","G","T"], dtype="S1")
    def one_hot_encode(sequence, dtype=np.uint8):
        sequence = sequence.upper()
        seq_chararray = np.frombuffer(sequence.encode('UTF-8'), dtype='S1')
        one_hot = (seq_chararray[:,None] == SEQ_ALPHABET[None,:]).astype(dtype)

        return one_hot

    def one_hot_decode(one_hot):
        return SEQ_ALPHABET[one_hot.argmax(axis=1)].tobytes().decode('UTF-8')

    one_hot_sequence = one_hot_encode(sequence)
    shuffled_one_hot = shuffle(one_hot_sequence[None,:,:])
    shuffled = one_hot_decode(shuffled_one_hot[0,:,:])
    return shuffled
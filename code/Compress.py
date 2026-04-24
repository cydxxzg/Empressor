import torch
from torch.utils.data import DataLoader
from SeqRegressionModel import *
from util import backbone_one_hot, get_number, decode_oneHot, LoadData
from sko.GA import GA
from sko.tools import set_run_mode
from tqdm import tqdm
import time
import argparse
import os
import numpy as np
import random

parser = argparse.ArgumentParser(description="Start of combine and optimize algothrim.")
parser.add_argument('-name', type=str, help='name of promoter', required=True)
parser.add_argument('-predictor', type=str, help='predictor model', required=True)
parser.add_argument('-predictor2', type=str, help='specificity predictor model 2', default=None)
parser.add_argument('-predictor3', type=str, help='specificity predictor model 3', default=None)
parser.add_argument('-generator', type=str, help='generate model', required=True)
parser.add_argument('-promoter_path', type=str, help='promoter file', required=True)
parser.add_argument('-motif_split', type=str, help='motif split file', required=True)
parser.add_argument('-num', type=int, help='number of motif', default=8)
parser.add_argument('-tag', type=str, help='tag', default='HepG2')
parser.add_argument('-gpu', type=str, help='tag', default='0')
parser.add_argument('-gap', type=int, help='gap between motifs', default=25)
parser.add_argument('-lambda', type=float, help='lambda parameter for specificity fitness', default=1.0, dest='lambda_val')
parser.add_argument('-save_path', type=str, help='save path', default='../result')
parser.add_argument('-specificity', action='store_true', help='enable specificity compression mode')
args = parser.parse_args()

Promoter_name = args.name
motif_split = args.motif_split
promoter_path = args.promoter_path
predictor_path = args.predictor
predictor2_path = args.predictor2
predictor3_path = args.predictor3
generator_path = args.generator
MOTIF_GAP = args.gap
chosen_num = args.num
seqG = MOTIF_GAP*chosen_num
tag = args.tag
day_tag = time.strftime("%m%d", time.localtime())
lambda_param = args.lambda_val
enable_specificity = args.specificity
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)

if enable_specificity:
    if predictor2_path is None or predictor3_path is None:
        raise ValueError("When -specificity is enabled, both -predictor2 and -predictor3 are required.")
else:
    if predictor2_path is not None or predictor3_path is not None:
        raise ValueError("Set -specificity to enable specificity compression when using -predictor2/-predictor3.")

pop_num = 500 
mutate_rate = 0.05
mutate_rate_in = 0.01
n_gen = 50
gn_pop_num = 50
gn_n_gen = 10


with open(promoter_path, 'r') as f:
    promoter = f.readline()
    promoter = ''.join([s.upper() for s in promoter])

with open(motif_split, 'r') as f:
    start = f.readline()
    Motif_starts = get_number(start)
    stop = f.readline()
    Motif_stops = get_number(stop)

torch.cuda.set_device(int(args.gpu))
generator = torch.load(generator_path).cuda()
predictor = torch.load(predictor_path).cuda()
predictor2 = torch.load(predictor2_path).cuda() if predictor2_path else None
predictor3 = torch.load(predictor3_path).cuda() if predictor3_path else None

for p in generator.parameters():
    p.requires_grad = False
for p in predictor.parameters():
    p.requires_grad = False
if predictor2 is not None:
    for p in predictor2.parameters():
        p.requires_grad = False
if predictor3 is not None:
    for p in predictor3.parameters():
        p.requires_grad = False



def seq_get_score(seq, predictor):
    one_hot = [backbone_one_hot(seq)]
    dataset_input = DataLoader(
        LoadData(data=one_hot),
        batch_size=1, shuffle=False)
    seq_fitness = [predictor(i).detach().cpu().float().numpy()[0] for i in dataset_input]
    return seq_fitness[0]

def seqs_get_score(seqs, predictor):
    processed_seq_list = []
    for s in seqs:
        processed_seq_list.append(backbone_one_hot(s))

    dataset_input = DataLoader(
        LoadData(data=processed_seq_list),
        batch_size=1, shuffle=False)

    seq_fitness = [predictor(i).detach().cpu().float().numpy()[0] for i in dataset_input]

    return seq_fitness

def verify(ori_seq, new_seq):
    # 保留原序列中的非M位
    if not hasattr(verify, 'call_count'):
        verify.call_count = 0
        verify.total_correction_rate = 0.0
    corrected = 0
    total_non_M = 0
    ori_list = list(ori_seq)
    new_list = list(new_seq)
    for i in range(len(ori_list)):
        if ori_list[i] != 'M':
            total_non_M += 1
            if ori_list[i] != new_list[i]:
                new_list[i] = ori_list[i]
                corrected += 1

    # 计算本次矫正率（避免除零）
    if total_non_M > 0:
        correction_rate = corrected / total_non_M
    else:
        correction_rate = 0.0

    # 累计统计
    verify.call_count += 1
    verify.total_correction_rate += correction_rate

    # 每10000次打印平均矫正率
    if verify.call_count % 1000 == 0:
        avg_rate = verify.total_correction_rate / 1000
        print(f"[verify] After {verify.call_count} calls, average correction rate: {avg_rate:.4f}")
        verify.total_correction_rate = 0.0

    return ''.join(new_list)

class flanking_optimizer:
    def __init__(self):
        self.lb_output = -float(1)
        self.seqs, self.masks = [], []
        self.seqs_string = []
        self.seq_results = []

    def set_input(self, seqG_masked_seqs):
        self.seqs_string = seqG_masked_seqs
        self.seq_results = []
        self.seqs, self.masks = [], []
        for i in range(len(seqG_masked_seqs)):
            seq_i = seqG_masked_seqs[i]
            self.seq_results.append('')
            mask_i = np.zeros([4, len(seq_i)])
            for j in range(len(seq_i)):
                if seq_i[j] == 'M':
                    mask_i[:, j] = np.ones([4, 1])[:, 0]
            self.seqs.append(backbone_one_hot(seq_i))
            self.masks.append(mask_i)
        self.i = 0

    def opt_func(self, p):
        lb_output = self.lb_output
        seqL = self.seqs[self.i].shape[1]
        assert seqL == seqG, f"Expected seqL == seqG, got seqL={seqL}, seqG={seqG}"
        pop_num = np.size(p, 0)
        p_reshape = np.zeros([pop_num, 4, seqL])
        mask_i = self.masks[self.i]
        for i in range(pop_num):
            p_reshape[i, :, :] = np.multiply(self.seqs[self.i], 1 - mask_i) + np.multiply(mask_i, p[i, :].reshape([4, -1]))
        with torch.no_grad():
            positionData = DataLoader(LoadData(data=p_reshape), batch_size=1024, shuffle=False)
            tensorSeq = []
            for j, eval_data in enumerate(positionData):
                tensorSeq.append(generator(eval_data).detach())
            tensorSeq_full = torch.cat(tensorSeq, dim=0).cpu().float().numpy()
            for i in range(pop_num):
                for j in range(seqG):
                    maxId = np.argsort(tensorSeq_full[i, :, j])
                    tensorSeq_full[i, :, j] = 0
                    tensorSeq_full[i, maxId[-1], j] = 1
            generateData = DataLoader(LoadData(data=tensorSeq_full), batch_size=1024, shuffle=False)
            predictions = []
            seq_generate = []
            for j, eval_data in enumerate(generateData):
                seq_generate.append(eval_data)
                predictions.append(predictor(eval_data).detach())
            seq_generate = torch.cat(seq_generate, dim=0).cpu().float().numpy()
            predictions = torch.cat(predictions, dim=0).cpu().float().numpy()
            for k in range(np.size(predictions, 0)):
                seq_decode_k = decode_oneHot(np.squeeze(seq_generate[k, :, :]).reshape([4, -1]))
                for m_j in range(seqL):
                    if self.seqs_string[self.i][m_j] != 'M' and self.seqs_string[self.i][m_j] != seq_decode_k[m_j]:
                        predictions[k] = lb_output
                        break
            preList = np.argsort(-predictions)
            seq_max = seq_generate[preList[0]]
            seq_opt = decode_oneHot(np.squeeze(seq_max))

            seq_opt = verify(self.seqs_string[self.i], seq_opt)

            self.seq_results[self.i] = seq_opt

            return -predictions

    def optimization(self):
        mode = 'vectorization'
        set_run_mode(self.opt_func, mode)
        seqL = self.seqs[0].shape[1]
        for i in range(len(self.seqs)):
            self.i = i
            lb, ub = [], []
            for _ in range(4*seqL):
                lb.append(0)
                ub.append(1)
            ga = GA(func=self.opt_func, n_dim=4*seqL, size_pop=gn_pop_num, max_iter=gn_n_gen, prob_mut=mutate_rate_in, lb=lb, ub=ub,
                    precision=1e-7)
            ga.run()

def random_init(chromo_len):
    pop = np.zeros((pop_num, chromo_len))
    for chromo in pop:
        i = 0
        while i < chosen_num:
            index = random.randint(0, chromo_len-1)
            if chromo[index] == 0:
                chromo[index] = 1
                i += 1
    return pop

class Agency:
    def __init__(self):
        self.seq = promoter
        self.start = Motif_starts
        self.stop = Motif_stops
        self.motif_num = len(self.start)
        self.pop = random_init(self.motif_num)
        self.now_fitness = []
        self.seq_list = []
        self.ever_best = []
        self.ever_score = []
        self.ever_mean_seqlenth = []
        self.flanking_opt = flanking_optimizer()

    def decode_chromo(self,chromo):
        seq = ['M'] * seqG
        id = 0
        for i in range(len(chromo)):
            if chromo[i] == 1:
                motif_len = self.stop[i] - self.start[i] + 1
                center = MOTIF_GAP * id + int(MOTIF_GAP // 2)
                insert_left = max(center - motif_len // 2, 0)
                insert_right = min(center + motif_len - motif_len // 2, seqG)
                
                motif_seq = self.seq[self.start[i]-1:self.stop[i]]
                if center - motif_len // 2 < 0:
                    left_overflow = -(center - motif_len // 2)
                    motif_seq = motif_seq[left_overflow:]
                if center + motif_len - motif_len // 2 > seqG:
                    right_overflow = center + motif_len - motif_len // 2 - seqG
                    motif_seq = motif_seq[:-right_overflow] if right_overflow > 0 else motif_seq
                
                target_len = insert_right - insert_left
                motif_list = list(motif_seq)
                if len(motif_list) != target_len:
                    if len(motif_list) > target_len:
                        motif_list = motif_list[:target_len]
                    else:
                        motif_list = motif_list + ['M'] * (target_len - len(motif_list))
                
                seq[insert_left:insert_right] = motif_list
                id += 1
        TF_units = ''.join(seq)
        assert len(TF_units) == seqG, f"TF_units length {len(TF_units)} != seqG {seqG}"
        return TF_units

    def flanking_generate(self, TF_units):
        padding = seqG - len(TF_units) % seqG if len(TF_units) % seqG != 0 else 0
        padded_seq = "N" * padding + TF_units
        self.flanking_opt.set_input([padded_seq])
        self.flanking_opt.optimization()
        result = self.flanking_opt.seq_results[0][padding:]

        result = verify(TF_units, result)
        return result

    def fitness(self):
        self.seq_list = []
        for chromo in self.pop:
            TF_units = self.decode_chromo(chromo)
            compressed_seq = self.flanking_generate(TF_units)
            self.seq_list.append(compressed_seq)
        fits = np.asarray(seqs_get_score(self.seq_list, predictor))
        if enable_specificity:
            fits2 = np.asarray(seqs_get_score(self.seq_list, predictor2))
            fits3 = np.asarray(seqs_get_score(self.seq_list, predictor3))
            fits2_clipped = np.clip(fits2, None, 10)
            fits3_clipped = np.clip(fits3, None, 10)
            self.now_fitness = fits - lambda_param * np.log(1 + np.exp(fits2_clipped)) - lambda_param * np.log(1 + np.exp(fits3_clipped))
        else:
            self.now_fitness = fits
        return self.now_fitness

    def crossover(self,chromo1,chromo2):
        index1 = [i for i in range(self.motif_num) if chromo1[i] == 1 and chromo2[i] == 0]
        index2 = [i for i in range(self.motif_num) if chromo1[i] == 0 and chromo2[i] == 1]
        def sample_geometric(p, n):
            k_vals = np.arange(1, n + 1)
            probs = (1 - p) ** (k_vals - 1)
            return np.random.choice(k_vals, p=probs / probs.sum())

        if len(index1) > 0:
            k = sample_geometric(0.5, len(index1))
            index1 = random.sample(index1, k)
            index2 = random.sample(index2, k)
            for i in range(k):
                chromo1[index1[i]] = 0
                chromo1[index2[i]] = 1
                chromo2[index1[i]] = 1
                chromo2[index2[i]] = 0
        return chromo1,chromo2

    def mutate(self,chromo):
        if random.random() < mutate_rate:
            is_1_index = [i for i in range(self.motif_num) if chromo[i] == 1]
            is_0_index = [i for i in range(self.motif_num) if chromo[i] == 0]
            if len(is_1_index) == 0 or len(is_0_index) == 0:
                pass
            else:
                index_1 = random.choice(is_1_index)
                index_0 = random.choice(is_0_index)
                chromo[index_1] = 0
                chromo[index_0] = 1
        return chromo


    def evolution(self):
        parent_fitness = self.fitness()
        parent_fitness = np.exp(parent_fitness - np.max(parent_fitness))
        parent_fitness = parent_fitness / parent_fitness.sum()
        parent_idx = np.random.choice(np.arange(pop_num), size=int(pop_num*0.1), replace=False, p=parent_fitness)
        children = []
        for i in range(0, int(pop_num*0.1), 2):
            child1, child2 = self.crossover(self.pop[parent_idx[i]].copy(), self.pop[parent_idx[i+1]].copy())
            children.append(self.mutate(child1))
            children.append(self.mutate(child2))
        surivial_idx = np.random.choice(np.arange(pop_num), size=int(pop_num*0.9), replace=False, p=parent_fitness)
        surivial = self.pop[surivial_idx]
        return np.concatenate((surivial, children), axis=0)

    def run(self):
        for i in tqdm(range(n_gen)):
            print("epoch:" + str(i) + "\n")
            self.pop = self.evolution()
            self.ever_best.append(self.seq_list[np.argmax(self.now_fitness)])
            self.ever_score.append(self.now_fitness)
        return self.pop

    def save(self):
        sorted_idx = np.argsort(-self.now_fitness)
        best_seq = [self.seq_list[i] for i in sorted_idx]
        os.makedirs(f'{save_path}/{Promoter_name}_{chosen_num}_{tag}_{day_tag}', exist_ok=True)
        with open(f'{save_path}/{Promoter_name}_{chosen_num}_{tag}_{day_tag}/best_seq.txt', 'w') as f:
            for seq in best_seq:
                f.write(seq+'\n')

        with open(f'{save_path}/{Promoter_name}_{chosen_num}_{tag}_{day_tag}/ever_score.txt','w') as f:
            for i,score_list in enumerate(self.ever_score):
                f.write("epoch:" + str(i) + "\n")
                for score in score_list:
                    f.write(str(score)+'\n')

        with open(f'{save_path}/{Promoter_name}_{chosen_num}_{tag}_{day_tag}/ever_best_seq.txt','w') as f:
            for seq in self.ever_best:
                f.write(seq+'\n')


if __name__ == '__main__':
    m = Agency()
    m.run()
    m.save()







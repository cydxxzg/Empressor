import pandas as pd

def mask_motifs_from_fimo(fimo_file, fasta_file, output_csv):
    # 读取fimo结果
    fimo_df = pd.read_csv(fimo_file, sep='\t', comment='#')  # 跳过注释行（FIMO通常有#开头的注释）
    
    # 读取fasta序列
    seq_dict = {}
    with open(fasta_file, 'r') as f:
        name = None
        seq = ""
        for line in f:
            if line.startswith(">"):
                if name:
                    seq_dict[name] = seq
                name = line[1:].strip()
                seq = ""
            else:
                seq += line.strip()
        if name:
            seq_dict[name] = seq
    
    results = []
    for seq_name in seq_dict:
        original_seq = seq_dict[seq_name]
        seq_len = len(original_seq)
        
        # 跳过含 'N' 的序列（按你原逻辑）
        if 'N' in original_seq:
            continue
        
        # 初始化：全部掩蔽为 'M'（表示非motif）
        masked_seq = ['M'] * seq_len
        
        # 获取该序列的所有motif区域
        seq_motifs = fimo_df[fimo_df['sequence_name'] == seq_name].copy()
        if seq_motifs.empty:
            # 如果没有motif，则整条序列都是 'M'，仍保留
            pass
        else:
            # FIMO的start和stop是1-based，inclusive
            for _, motif in seq_motifs.iterrows():
                start = int(motif['start']) - 1
                stop = int(motif['stop'])
                for i in range(start, min(stop, seq_len)):
                    masked_seq[i] = original_seq[i]
        
        results.append({
            'realB': original_seq,
            'realA': ''.join(masked_seq)
        })
    
    # 保存结果
    result_df = pd.DataFrame(results)

    return result_df


if __name__ == "__main__":

    fimo_file = f'../data/generator_data/fimo.tsv'
    fasta_file = f'../data/generator_data/human_enhancer_hepG2_cage_200.fa'
    output_csv = f'../data/generator_data/human_enhancer_hepG2_cage_mask_200.csv'
    result_df = mask_motifs_from_fimo(fimo_file, fasta_file, output_csv)

    result_df.to_csv(output_csv, index=False)
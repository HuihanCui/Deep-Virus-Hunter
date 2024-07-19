import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
import torch
import torch.nn as nn
import torch.nn.functional as F
import model


def adjust_uncertain_nt(seq):
    mapping = {
        'N': 'G', 'X': 'A', 'H': 'T', 'M': 'C',
        'K': 'G', 'D': 'A', 'R': 'G', 'Y': 'T',
        'S': 'C', 'W': 'A', 'B': 'C', 'V': 'G'
    }
    return ''.join([mapping.get(nt, nt) for nt in seq])

def nt2num(nt):
    mapping = {
        'A': 0, 'C': 1, 'G': 2, 'T': 3,
        'N': 2, 'X': 0, 'H': 3, 'M': 1,
        'K': 2, 'D': 0, 'R': 2, 'Y': 3,
        'S': 1, 'W': 0, 'B': 1, 'V': 2
    }
    return mapping.get(nt, -1)

def nt2onehot(seq, L):
    r_seq = str(Seq(seq).reverse_complement())

    seq = adjust_uncertain_nt(seq.upper())
    r_seq = adjust_uncertain_nt(r_seq.upper())

    fw_onehot = np.zeros((L, 4), dtype=np.int8)
    for i in range(min(L, len(seq))):
        if seq[i] == 'A':
            fw_onehot[i] = [0, 0, 0, 1]
        elif seq[i] == 'C':
            fw_onehot[i] = [0, 0, 1, 0]
        elif seq[i] == 'G':
            fw_onehot[i] = [0, 1, 0, 0]
        elif seq[i] == 'T':
            fw_onehot[i] = [1, 0, 0, 0]

    bw_onehot = np.zeros((L, 4), dtype=np.int8)
    for i in range(min(L, len(r_seq))):
        if r_seq[i] == 'A':
            bw_onehot[i] = [0, 0, 0, 1]
        elif r_seq[i] == 'C':
            bw_onehot[i] = [0, 0, 1, 0]
        elif r_seq[i] == 'G':
            bw_onehot[i] = [0, 1, 0, 0]
        elif r_seq[i] == 'T':
            bw_onehot[i] = [1, 0, 0, 0]
    return np.concatenate((fw_onehot, bw_onehot), axis=0).T

def condon2onehot(seq, L):
    r_seq = str(Seq(seq).reverse_complement())

    seq = adjust_uncertain_nt(seq.upper())
    r_seq = adjust_uncertain_nt(r_seq.upper())

    def process_seq(sequence):
        n = 0
        condon_onehot = np.zeros((64, L//3), dtype=np.int8)
        for i in range(0, min(L, len(sequence))-4, 3):
            idx = nt2num(sequence[i]) * 16 + nt2num(sequence[i+1]) * 4 + nt2num(sequence[i+2])
            condon_onehot[idx, n] = 1
            n += 1
        return condon_onehot

    condon_fw1 = process_seq(seq)
    condon_fw2 = process_seq(seq[1:])
    condon_fw3 = process_seq(seq[2:])
    
    condon_bw1 = process_seq(r_seq)
    condon_bw2 = process_seq(r_seq[1:])
    condon_bw3 = process_seq(r_seq[2:])

    condon_onehot = np.concatenate((
        condon_fw1, condon_fw2, condon_fw3, 
        condon_bw1, condon_bw2, condon_bw3
    ), axis=1)
    return condon_onehot

def process_fasta(file_path):
    groups = {'A': [], 'B': [], 'C': [], 'D': []}
    for record in SeqIO.parse(file_path, "fasta"):
        seq = str(record.seq)
        length = len(seq)
        if 100 <= length <= 400:
            groups['A'].append((record.id, seq, length))
        elif 401 <= length <= 800:
            groups['B'].append((record.id, seq, length))
        elif 801 <= length <= 1200:
            groups['C'].append((record.id, seq, length))
        else:
            groups['D'].append((record.id, seq, length))

    return groups

def process_d(seq, m):
    chunks = [seq[i:i + 1200] for i in range(0, len(seq), 1200)]
    if len(chunks[-1]) < 100:
        chunks = chunks[:-1]

    results = []
    for chunk in chunks[:-1]:
        b_input = torch.tensor(nt2onehot(chunk, 1200), dtype=torch.float32) 
        b_input = torch.unsqueeze(b_input, 0) 
        c_input = torch.tensor(condon2onehot(chunk, 1200), dtype=torch.float32) 
        c_input = torch.unsqueeze(c_input, 0) 
        result_raw = m(b_input, c_input)
        result = [item for sublist in result_raw.tolist() for item in sublist]
        results.append(result)

    b_input = torch.tensor(nt2onehot(chunks[-1], len(chunks[-1])), dtype=torch.float32) 
    b_input = torch.unsqueeze(b_input, 0) 
    c_input = torch.tensor(condon2onehot(chunks[-1], len(chunks[-1])), dtype=torch.float32) 
    c_input = torch.unsqueeze(c_input, 0) 
    result_raw = m(b_input, c_input)
    result = [item for sublist in result_raw.tolist() for item in sublist]
    results.append(result)

    weights = [1200] * (len(results) - 1) + [len(chunks[-1])]
    data_array = np.array(results)
    weighted_averages = np.average(data_array, axis=0, weights=weights)
    weighted_result = weighted_averages.tolist()

    return weighted_result

def predict(results):
    for result in results:
        max_index = result.index(max(result))

        if max_index == 0:
            result.append("Phages")
        elif max_index == 2:
            result.append("Plasmids")
        else:
            result.append("Chromosomes")

    return results



def run(name):
    groups = process_fasta(name)
    b_inputs = []
    c_inputs = []
    output = []
    m = model.BiPathCNN()

    # The groups are to use later where models for different length could be different
    for group in ['A', 'B', 'C']:
        for (_, seq, length) in groups[group]:
            b_inputs.append(nt2onehot(seq, length))
            c_inputs.append(condon2onehot(seq, length))

    for i in range(len(b_inputs)):
        b_input = torch.tensor(b_inputs[i], dtype=torch.float32) 
        b_input = torch.unsqueeze(b_input, 0) 

        c_input = torch.tensor(c_inputs[i], dtype=torch.float32) 
        c_input = torch.unsqueeze(c_input, 0) 

        result_raw = m(b_input, c_input)
        result = [item for sublist in result_raw.tolist() for item in sublist]
        output.append(result)

    for (_, seq, length) in groups['D']:
        output.append(process_d(seq, m))

    output = predict(output)
        
    
    return output


fasta_file = "example.fna"
output_file = "result.txt"
output = run(fasta_file)

with open(output_file, 'w') as file:
    for sublist in output:
        line = ' '.join(map(str, sublist))
        file.write(line + '\n')
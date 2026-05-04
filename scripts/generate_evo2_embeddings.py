## this script generates evo2 embeddings for sorfs

from evo2 import Evo2
import torch
from Bio import SeqIO
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


# loading evo2 model
model = Evo2("evo2_7b")

batch_size = 1024
seqs, labels, seq_ids = [], [], []
all_embeds = []

# load in the sequences 
for record in SeqIO.parse("scratch/evo2/working/orfs_to_score.fa", "fasta"):
    seqs.append(str(record.seq).upper())
    header = record.id.split('|')
    labels.append(int(header[-1].split('=')[1]))
    seq_ids.append(record.id)

print(f"loadingg {len(seqs)} sequences")


layer_names = ['blocks.28.mlp.l3']  # selecting the best performing intermediate to final layer (found 28 to be optimal)

# running evo2 and saving embeddings in batches
for i in range(0, len(seqs), batch_size):
    batch_seqs = seqs[i:i+batch_size]
    batch_tokens = [model.tokenizer.tokenize(s) for s in batch_seqs]
    max_b_len = max(len(t) for t in batch_tokens)
    padded = torch.tensor([t + [0]*(max_b_len-len(t)) for t in batch_tokens], dtype=torch.int).to('cuda:0')
    
    with torch.no_grad():
        outputs = model(padded, return_embeddings=True, layer_names=layer_names)
        batch_embeds = outputs[1][layer_names[0]].float().mean(dim=1).cpu().numpy()
    
    # save to disk bc of limits
    np.save(f'scratch/evo2/working/embeds/batch_{i}.npy', batch_embeds)
    print(f"batch {i//batch_size + 1}: saved")
import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

data_folder   = './processed'
combined_file = data_folder + '/all_species_combined.parquet'
out_file      = data_folder + '/subset_300k.parquet'

print('loading columns for stats...')
df = pd.read_parquet(combined_file, columns=['label', 'seq_length', 'species'])

n_total = len(df)
n_pos   = (df.label == 1).sum()
n_neg   = (df.label == 0).sum()

print('\n' + '='*60)
print('FULL DATASET')
print('='*60)
print('total rows:  ' + str(n_total))
print('Y=1 (genes): ' + str(n_pos) + '  (' + str(round(100*n_pos/n_total, 1)) + '%)')
print('Y=0 (noise): ' + str(n_neg) + '  (' + str(round(100*n_neg/n_total, 1)) + '%)')

print('\n' + '='*60)
print('SHORT ORFs (<=90 nt)')
print('='*60)
short   = df[df.seq_length <= 90]
n_short = len(short)
ns_pos  = (short.label == 1).sum()
ns_neg  = (short.label == 0).sum()
print('total short: ' + str(n_short) + '  (' + str(round(100*n_short/n_total, 1)) + '% of all)')
print('Y=1 short:   ' + str(ns_pos)  + '  (' + str(round(100*ns_pos/n_short,  1)) + '%)')
print('Y=0 short:   ' + str(ns_neg)  + '  (' + str(round(100*ns_neg/n_short,  1)) + '%)')

print('\n' + '='*60)
print('LENGTH BINS')
print('='*60)
bin_edges  = [0, 90, 150, 300, 500, 1000, 3000]
bin_labels = ['30-90', '91-150', '151-300', '301-500', '501-1000', '1001-3000']
df['bin'] = pd.cut(df.seq_length, bins=bin_edges, labels=bin_labels)
for b in bin_labels:
    b_df  = df[df.bin == b]
    b_pos = (b_df.label == 1).sum()
    b_neg = (b_df.label == 0).sum()
    b_tot = len(b_df)
    print(b.ljust(12) +
          '  total=' + str(b_tot) +
          '  pos='   + str(b_pos) +
          '  neg='   + str(b_neg) +
          '  (' + str(round(100*b_tot/n_total, 1)) + '% of all)')

print('\n' + '='*60)
print('PER SPECIES')
print('='*60)
for sp in sorted(df.species.unique()):
    sp_df    = df[df.species == sp]
    sp_pos   = (sp_df.label == 1).sum()
    sp_neg   = (sp_df.label == 0).sum()
    sp_short = (sp_df.seq_length <= 90).sum()
    sp_med   = int(sp_df.seq_length.median())
    print(sp.ljust(20) +
          '  total='       + str(len(sp_df)) +
          '  pos='         + str(sp_pos) +
          '  neg='         + str(sp_neg) +
          '  short='       + str(sp_short) +
          '  median_len='  + str(sp_med) + 'nt')

print('\n' + '='*60)
print('BUILDING 300k SUBSET')
print('='*60)

df = df.drop(columns=['bin'])

target_per_class = 150000
sp_list          = sorted(df.species.unique())
n_species        = len(sp_list)
per_species      = target_per_class // n_species
seed             = 42

print('target per class:      ' + str(target_per_class))
print('species:               ' + str(n_species))
print('per species per class: ' + str(per_species))

pos_indices = []
neg_indices = []

for sp in sp_list:
    sp_df = df[df.species == sp]

    sp_pos = sp_df[sp_df.label == 1]
    sp_neg = sp_df[sp_df.label == 0]

    sp_pos_short = sp_pos[sp_pos.seq_length <= 90]
    sp_pos_long  = sp_pos[sp_pos.seq_length >  90]
    n_short_pos  = min(len(sp_pos_short), per_species)
    n_long_pos   = per_species - n_short_pos

    picked_pos = []
    if n_short_pos > 0:
        picked_pos += sp_pos_short.sample(n=n_short_pos, random_state=seed).index.tolist()
    if n_long_pos > 0 and len(sp_pos_long) > 0:
        picked_pos += sp_pos_long.sample(n=min(n_long_pos, len(sp_pos_long)), random_state=seed).index.tolist()

    sp_neg_short = sp_neg[sp_neg.seq_length <= 90]
    sp_neg_long  = sp_neg[sp_neg.seq_length >  90]
    n_short_neg  = min(len(sp_neg_short), per_species)
    n_long_neg   = per_species - n_short_neg

    picked_neg = []
    if n_short_neg > 0:
        picked_neg += sp_neg_short.sample(n=n_short_neg, random_state=seed).index.tolist()
    if n_long_neg > 0 and len(sp_neg_long) > 0:
        picked_neg += sp_neg_long.sample(n=min(n_long_neg, len(sp_neg_long)), random_state=seed).index.tolist()

    pos_indices += picked_pos
    neg_indices += picked_neg

    print('  ' + sp.ljust(20) + '  pos=' + str(len(picked_pos)) + '  neg=' + str(len(picked_neg)))

rng     = np.random.default_rng(seed)
n_final = min(len(pos_indices), len(neg_indices))
pos_indices  = rng.choice(pos_indices, size=n_final, replace=False).tolist()
neg_indices  = rng.choice(neg_indices, size=n_final, replace=False).tolist()
keep = sorted(pos_indices + neg_indices)

print('\ntotal rows selected: ' + str(len(keep)))
print('reading full rows from parquet...')

df_full   = pd.read_parquet(combined_file)
df_subset = df_full.iloc[keep].sample(frac=1, random_state=seed).reset_index(drop=True)
del df_full

n_short_sub = (df_subset.seq_length <= 90).sum()
print('\nsubset summary:')
print('  total rows:      ' + str(len(df_subset)))
print('  Y=1 (genes):     ' + str((df_subset.label == 1).sum()))
print('  Y=0 (noise):     ' + str((df_subset.label == 0).sum()))
print('  short <=90:      ' + str(n_short_sub) + '  (' + str(round(100*n_short_sub/len(df_subset), 1)) + '%)')
print('  median orf len:  ' + str(int(df_subset.seq_length.median())) + ' nt')
print('  species breakdown:')
for sp in sorted(df_subset.species.unique()):
    print('    ' + sp.ljust(20) + '  ' + str((df_subset.species == sp).sum()))

df_subset.to_parquet(out_file, index=False)
size_mb = os.path.getsize(out_file) / 1e6
print('\nsaved -> ' + out_file + '  (' + str(round(size_mb, 1)) + ' MB)')
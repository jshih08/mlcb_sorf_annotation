import os
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

data_folder = './processed'
out_file = data_folder + '/subset_300k_short.parquet'

skip_files = ['all_species_combined.parquet', 'subset_300k.parquet', 'subset_300k_short.parquet']

all_files = []
for fname in sorted(os.listdir(data_folder)):
    if fname.endswith('.parquet') and fname not in skip_files:
        all_files.append(data_folder + '/' + fname)

print('found ' + str(len(all_files)) + ' species files')

target_per_class = 150000
n_species = len(all_files)
per_species = target_per_class // n_species
rng = np.random.default_rng(42)

print('per species per class: ' + str(per_species))

writer = None

for fpath in all_files:
    sp_name = os.path.basename(fpath).replace('.parquet', '')
    print('reading ' + sp_name + '...')

    pf = pq.ParquetFile(fpath)
    data = pf.read(['label', 'seq_length']).to_pydict()
    labels = np.array(data['label'])
    lengths = np.array(data['seq_length'])
    del data

    pos_idx = np.where((lengths <= 90) & (labels == 1))[0]
    neg_idx = np.where((lengths <= 90) & (labels == 0))[0]
    del labels, lengths

    print('  short pos=' + str(len(pos_idx)) + '  short neg=' + str(len(neg_idx)))

    n_pos = min(len(pos_idx), per_species)
    n_neg = min(len(neg_idx), per_species)

    picked_pos = rng.choice(pos_idx, size=n_pos, replace=False)
    picked_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    keep = np.sort(np.concatenate([picked_pos, picked_neg]))
    del pos_idx, neg_idx, picked_pos, picked_neg

    print('  keeping ' + str(len(keep)) + ' rows')

    full_table = pf.read()
    subset = full_table.take(keep)
    del full_table, keep

    if writer is None:
        writer = pq.ParquetWriter(out_file, subset.schema)
    writer.write_table(subset)
    del subset

if writer:
    writer.close()

print('\nverifying...')
pf = pq.ParquetFile(out_file)
data = pf.read(['label', 'seq_length', 'species']).to_pydict()
labels = np.array(data['label'])
lengths = np.array(data['seq_length'])
species = data['species']
del data

print('total rows: ' + str(len(labels)))
print('Y=1:        ' + str((labels == 1).sum()))
print('Y=0:        ' + str((labels == 0).sum()))
print('all short:  ' + str((lengths <= 90).sum()))

unique_sp = sorted(set(species))
print('\nspecies breakdown:')
for sp in unique_sp:
    sp_mask = np.array([s == sp for s in species])
    print('  ' + sp.ljust(20) +
          '  total=' + str(sp_mask.sum()) +
          '  pos=' + str(((labels == 1) & sp_mask).sum()) +
          '  neg=' + str(((labels == 0) & sp_mask).sum()))

size_mb = os.path.getsize(out_file) / 1e6
print('\nsaved -> ' + out_file + '  (' + str(round(size_mb, 1)) + ' MB)')
import os
import re
import math
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from Bio import SeqIO
from Bio.Seq import Seq
from itertools import product

# output folders
out_folder = './processed'
genome_folder = './raw_genomes'

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

# all the species we want to process
species_list = {
    'ecoli':         'GCF_000005845.2',
    'bsubtilis':     'GCF_000009045.1',
    'scoelicolor':   'GCF_000203835.1',
    'mgenitalium':   'GCF_000027345.1',
    'ccrescentus':   'GCF_000006905.1',
    'paeruginosa':   'GCF_000006765.1',
    'mtuberculosis': 'GCF_000195955.2',
    'saureus':       'GCF_000013425.1',
    'hpylori':       'GCF_000008525.1',
    'cdiff':         'GCF_000009205.1',
    'dradiodurans':  'GCF_000008565.1',
    'tthermophilus': 'GCF_000091545.1',
    'vcholerae':     'GCF_000006745.1',
}

start_codons = ['ATG', 'GTG', 'TTG']
stop_codons  = ['TAA', 'TAG', 'TGA']

# all possible 3-mers from ACGT
all_kmers = []
for a in 'ACGT':
    for b in 'ACGT':
        for c in 'ACGT':
            all_kmers.append(a + b + c)

# Shine-Dalgarno patterns and their scores
sd_list = [
    (re.compile(r'AGGAGG'), 1.0),
    (re.compile(r'AAGGAG'), 0.9),
    (re.compile(r'AGGAG'),  0.8),
    (re.compile(r'AGGGG'),  0.7),
    (re.compile(r'GAGG'),   0.5),
    (re.compile(r'AAGG'),   0.4),
]

feature_names = (
    all_kmers
    + ['gc1', 'gc2', 'gc3']
    + ['start_ATG', 'start_GTG', 'start_TTG']
    + ['stop_TAA', 'stop_TGA', 'stop_TAG']
    + ['seq_length']
    + ['left_dist', 'right_dist']
    + ['rbs_score']
    + ['genomic_start', 'genomic_end']
    + ['orf_length']
    + ['strand_encoded']
    + ['chrom_encoded']
)


# check if we already processed this species
def already_done(sp):
    pq_file  = out_folder + '/' + sp + '.parquet'
    done_file = out_folder + '/' + sp + '.done'
    if os.path.exists(pq_file) and os.path.exists(done_file):
        return True
    return False

def write_done_marker(sp):
    done_file = out_folder + '/' + sp + '.done'
    f = open(done_file, 'w')
    f.write('done')
    f.close()


# find the genome and annotation files for a species
def get_files(sp):
    folder = genome_folder + '/' + sp
    fna_file = None
    gff_file = None
    for root, dirs, files in os.walk(folder):
        for fname in files:
            if fname.endswith('.fna') and fna_file is None:
                fna_file = root + '/' + fname
            if fname == 'genomic.gff' and gff_file is None:
                gff_file = root + '/' + fname
    if fna_file is None:
        raise FileNotFoundError('no fna file found in ' + folder)
    if gff_file is None:
        raise FileNotFoundError('no gff file found in ' + folder)
    return fna_file, gff_file


# load genome sequences from fasta file
def load_genome(fna_path):
    chroms = {}
    for rec in SeqIO.parse(fna_path, 'fasta'):
        chroms[rec.id] = str(rec.seq).upper()
        print('  loaded ' + rec.id + ' len=' + str(len(chroms[rec.id])))
    return chroms


# parse the gff file to get CDS locations
def read_gff(gff_path):
    cds_list = []
    fh = open(gff_path)
    for line in fh:
        if line.startswith('#'):
            continue
        cols = line.strip().split('\t')
        if len(cols) < 9:
            continue
        if cols[2] != 'CDS':
            continue
        # parse the attributes column
        attrs = {}
        for item in cols[8].split(';'):
            if '=' in item:
                k, v = item.split('=', 1)
                attrs[k.strip()] = v.strip()
        entry = {}
        entry['chrom']  = cols[0]
        entry['start']  = int(cols[3]) - 1   # convert to 0-based
        entry['end']    = int(cols[4])
        entry['strand'] = cols[6]
        entry['id']     = attrs.get('ID', '')
        cds_list.append(entry)
    fh.close()
    print('  found ' + str(len(cds_list)) + ' CDS entries')
    return cds_list


# scan a sequence for ORFs in all 3 frames
def scan_orfs(seq, strand, chrom_id, min_len=30, max_len=3000):
    found = []
    n = len(seq)
    for frame in range(3):
        i = frame
        while i < n - 2:
            codon = seq[i:i+3]
            if codon in start_codons:
                # look for stop codon in same frame
                j = i + 3
                while j < n - 2:
                    stop = seq[j:j+3]
                    if stop in stop_codons:
                        orf_seq = seq[i:j+3]
                        orf_len = len(orf_seq)
                        if min_len <= orf_len <= max_len:
                            if strand == '+':
                                gs = i
                                ge = j + 3
                            else:
                                gs = n - (j + 3)
                                ge = n - i
                            rec = {}
                            rec['chrom']       = chrom_id
                            rec['start']       = gs
                            rec['end']         = ge
                            rec['strand']      = strand
                            rec['length']      = orf_len
                            rec['sequence']    = orf_seq
                            rec['start_codon'] = codon
                            rec['stop_codon']  = stop
                            found.append(rec)
                        break
                    j += 3
            i += 3
    return found


def get_all_orfs(chroms):
    all_orfs = []
    for chrom_id in chroms:
        seq = chroms[chrom_id]
        print('  scanning ' + chrom_id)
        fwd = scan_orfs(seq, '+', chrom_id)
        rev_seq = str(Seq(seq).reverse_complement())
        rev = scan_orfs(rev_seq, '-', chrom_id)
        print('    fwd=' + str(len(fwd)) + ' rev=' + str(len(rev)))
        for o in fwd:
            all_orfs.append(o)
        for o in rev:
            all_orfs.append(o)
    print('  total orfs: ' + str(len(all_orfs)))
    return all_orfs


# label each orf as 1 if it overlaps a known CDS, else 0
def label_orfs(orfs, cds_list):
    print('  labeling ' + str(len(orfs)) + ' orfs...')

    # group CDS by chrom+strand
    cds_by_key = {}
    for c in cds_list:
        key = c['chrom'] + '_' + c['strand']
        if key not in cds_by_key:
            cds_by_key[key] = []
        cds_by_key[key].append(c)

    count = 0
    for orf in orfs:
        key = orf['chrom'] + '_' + orf['strand']
        orf['label'] = 0
        if key not in cds_by_key:
            count += 1
            if count % 50000 == 0:
                print('    done ' + str(count))
            continue
        orf_s = orf['start']
        orf_e = orf['end']
        orf_len = orf_e - orf_s
        # loop through all CDS in this group and check overlap
        for c in cds_by_key[key]:
            overlap = max(0, min(orf_e, c['end']) - max(orf_s, c['start']))
            if orf_len > 0 and overlap / orf_len >= 0.8:
                orf['label'] = 1
                break
        count += 1
        if count % 50000 == 0:
            print('    done ' + str(count))

    n1 = 0
    n0 = 0
    for o in orfs:
        if o['label'] == 1:
            n1 += 1
        else:
            n0 += 1
    print('  label=1: ' + str(n1) + '  label=0: ' + str(n0))
    return orfs


# compute kmer frequencies for a sequence
def get_kmer_freqs(seq):
    counts = {}
    for km in all_kmers:
        counts[km] = 0
    total = 0
    i = 0
    while i < len(seq) - 2:
        c = seq[i:i+3]
        if c in counts:
            counts[c] += 1
            total += 1
        i += 3
    result = []
    for km in all_kmers:
        if total == 0:
            result.append(0.0)
        else:
            result.append(counts[km] / total)
    return result


# get GC content at each codon position
def get_gc123(seq):
    cnt = [0, 0, 0]
    tot = [0, 0, 0]
    i = 0
    while i < len(seq) - 2:
        for p in range(3):
            if i + p < len(seq):
                b = seq[i + p]
                if b in 'ACGT':
                    tot[p] += 1
                    if b in 'GC':
                        cnt[p] += 1
        i += 3
    res = []
    for p in range(3):
        if tot[p] > 0:
            res.append(cnt[p] / tot[p])
        else:
            res.append(0.0)
    return res


def get_rbs(orf, genome_seq):
    if orf['strand'] == '-':
        return 0.5
    start = orf['start']
    upstream = genome_seq[max(0, start - 20) : start]
    for pat, score in sd_list:
        if pat.search(upstream):
            return score
    return 0.0


# get distance to nearest CDS on left and right
def get_dists(orf, cds_by_key, cap=5000):
    key = orf['chrom'] + '_' + orf['strand']
    if key not in cds_by_key:
        return [cap, cap]
    group = cds_by_key[key]
    orf_s = orf['start']
    orf_e = orf['end']
    left_d = cap
    right_d = cap
    for c in group:
        if c['end'] <= orf_s:
            d = orf_s - c['end']
            if d < left_d:
                left_d = d
        if c['start'] >= orf_e:
            d = c['start'] - orf_e
            if d < right_d:
                right_d = d
    return [min(left_d, cap), min(right_d, cap)]


def make_features(orf, cds_by_key, genome_seq):
    seq = orf['sequence']
    feats = []

    # kmer freqs (64 values)
    feats += get_kmer_freqs(seq)

    # gc at each codon position
    feats += get_gc123(seq)

    # start codon one-hot
    sc = seq[:3]
    feats.append(1 if sc == 'ATG' else 0)
    feats.append(1 if sc == 'GTG' else 0)
    feats.append(1 if sc == 'TTG' else 0)

    # stop codon one-hot
    ec = seq[-3:]
    feats.append(1 if ec == 'TAA' else 0)
    feats.append(1 if ec == 'TGA' else 0)
    feats.append(1 if ec == 'TAG' else 0)

    # length
    feats.append(len(seq))

    # intergenic distances
    feats += get_dists(orf, cds_by_key)

    # rbs score
    feats.append(get_rbs(orf, genome_seq))

    return feats   # 77 features total


def run_species(sp):
    print('\n--- ' + sp + ' ---')

    if already_done(sp):
        print('already processed, skipping')
        return out_folder + '/' + sp + '.parquet'

    try:
        fna_path, gff_path = get_files(sp)
    except FileNotFoundError as e:
        print('skipping: ' + str(e))
        return None

    chroms   = load_genome(fna_path)
    cds_list = read_gff(gff_path)
    orfs     = get_all_orfs(chroms)
    orfs     = label_orfs(orfs, cds_list)

    # build cds lookup for distance calc
    cds_by_key = {}
    for c in cds_list:
        key = c['chrom'] + '_' + c['strand']
        if key not in cds_by_key:
            cds_by_key[key] = []
        cds_by_key[key].append(c)

    # encode chromosomes as ints
    chrom_names = sorted(chroms.keys())
    chrom_map = {}
    for i, name in enumerate(chrom_names):
        chrom_map[name] = i

    out_path = out_folder + '/' + sp + '.parquet'
    print('building rows...')

    rows = []
    chunk_size = 5000
    writer = None

    for idx in range(len(orfs)):
        orf = orfs[idx]
        genome_seq = chroms.get(orf['chrom'], '')
        num_feats  = make_features(orf, cds_by_key, genome_seq)

        row = {}
        row['species']        = sp
        row['chrom']          = orf['chrom']
        row['strand']         = orf['strand']
        row['start_codon']    = orf['start_codon']
        row['stop_codon']     = orf['stop_codon']
        row['dna_sequence']   = orf['sequence']
        row['genomic_start']  = orf['start']
        row['genomic_end']    = orf['end']
        row['orf_length']     = orf['length']
        row['strand_encoded'] = 1 if orf['strand'] == '+' else 0
        row['chrom_encoded']  = chrom_map.get(orf['chrom'], -1)
        row['label']          = orf['label']

        for i in range(len(num_feats)):
            row[feature_names[i]] = num_feats[i]

        rows.append(row)

        # write in chunks so we dont run out of memory
        if len(rows) >= chunk_size or idx == len(orfs) - 1:
            df = pd.DataFrame(rows)
            tbl = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, tbl.schema)
            writer.write_table(tbl)
            rows = []

        if (idx + 1) % 50000 == 0:
            print('  ' + str(idx + 1) + ' / ' + str(len(orfs)))

    if writer:
        writer.close()

    write_done_marker(sp)

    size = os.path.getsize(out_path) / 1e6
    check = pd.read_parquet(out_path, columns=['label'])
    n1 = int((check.label == 1).sum())
    n0 = int((check.label == 0).sum())
    print('saved ' + str(n1 + n0) + ' rows, Y=1=' + str(n1) + ' Y=0=' + str(n0) + ' size=' + str(round(size, 1)) + 'MB')
    return out_path


# run all species
saved = []
for sp in species_list:
    try:
        p = run_species(sp)
        if p is not None:
            saved.append(p)
    except Exception as e:
        import traceback
        print('error on ' + sp + ': ' + str(e))
        traceback.print_exc()
        continue


# combine all parquet files into one
combined = out_folder + '/all_species_combined.parquet'
print('\ncombining ' + str(len(saved)) + ' files...')

writer = None
total = 0

for p in saved:
    sp_name = os.path.basename(p).replace('.parquet', '')
    print('  adding ' + sp_name)
    tbl = pq.read_table(p)
    if writer is None:
        writer = pq.ParquetWriter(combined, tbl.schema)
    writer.write_table(tbl)
    total += tbl.num_rows
    del tbl

if writer:
    writer.close()

print('done!')
print('output: ' + combined)
print('rows:   ' + str(total))
print('size:   ' + str(round(os.path.getsize(combined) / 1e6, 1)) + ' MB')
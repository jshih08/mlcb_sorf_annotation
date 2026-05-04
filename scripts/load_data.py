import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path

OUTPUT_DIR  = Path('.')  # data path
subset_path = OUTPUT_DIR / 'subset_300k_short.parquet'

# loading
df = pd.read_parquet(subset_path)

print('Shape:        ' + str(df.shape))
print('Columns:      ' + str(list(df.columns)))
print('Y=1 (genes):  ' + str((df.label == 1).sum()))
print('Y=0 (noise):  ' + str((df.label == 0).sum()))

# splitting into x and y

# y is label
Y = df['label']

# X_model = only the numerical columns, what you feed to the classifier
# Drop all string columns the model cannot use
STRING_COLS = ['species', 'chrom', 'strand', 'start_codon',
               'stop_codon', 'dna_sequence', 'label']
X = df.drop(columns=[c for c in STRING_COLS if c in df.columns])

print('\nX shape: ' + str(X.shape))   # (n_rows, 82)
print('Y shape: ' + str(Y.shape))    # (n_rows,)

# generate trainand test split for model
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y 
)

print('\nX_train: ' + str(X_train.shape))
print('X_test:  ' + str(X_test.shape))
print('Y_train pos: ' + str((Y_train == 1).sum()) +
      '  neg: ' + str((Y_train == 0).sum()))
print('Y_test  pos: ' + str((Y_test  == 1).sum()) +
      '  neg: ' + str((Y_test  == 0).sum()))

# this script runs 3 models to generate auprc curves to measure structural annotation performance

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
from Bio import SeqIO
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import os
import joblib
os.chdir('scratch/evo2/working')



# loading in  data, can simply replace subset with full dataset for full analysis

OUTPUT_DIR  = Path('.') 
subset_path = OUTPUT_DIR / 'scratch/evo2/working/subset_300k_short.parquet'


df = pd.read_parquet(subset_path)

print('Shape:        ' + str(df.shape))
print('Columns:      ' + str(list(df.columns)))
print('Y=1 (genes):  ' + str((df.label == 1).sum()))
print('Y=0 (noise):  ' + str((df.label == 0).sum()))


Y = df['label']

STRING_COLS = ['species', 'chrom', 'strand', 'start_codon',
               'stop_codon', 'dna_sequence', 'label']
X = df.drop(columns=[c for c in STRING_COLS if c in df.columns])

print('\nX shape: ' + str(X.shape))   # (n_rows, 82)
print('Y shape: ' + str(Y.shape))    # (n_rows,)



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


# done loading



# MODEL 3: mlp model  ------------------------------------------




# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# turn data into tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32) 
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32).unsqueeze(1)


train_ds = TensorDataset(X_train_tensor, Y_train_tensor)
train_dl = DataLoader(train_ds, batch_size=512, shuffle=True, pin_memory=True, num_workers=0)


# the baby mlp model
class MLP(nn.Module):
    def __init__(self, in_dim=82):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = MLP()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn   = nn.BCELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# time to train
for epoch in range(10):
    model.train()
    total_loss = 0
    for xb, yb in train_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"epoch {epoch+1:2d} | loss: {total_loss/len(train_dl):.4f}")

# eval
model.eval()
with torch.no_grad():
    X_test_tensor = X_test_tensor.to(device) # moving to gpu
    preds = (model(X_test_tensor) > 0.5).float().squeeze().cpu().numpy()


print(classification_report(Y_test_tensor.numpy(), preds, target_names=['noise','gene']))

model.eval()
with torch.no_grad():
    X_test_tensor = X_test_tensor.to(device)
    # extract probs for auprc
    probs = model(X_test_tensor).squeeze().cpu().numpy()


precision, recall, thresholds = precision_recall_curve(Y_test_tensor.numpy(), probs)

# get imabalnce metric
auprc = average_precision_score(Y_test_tensor.numpy(), probs)

print(f"AUPRC: {auprc:.4f}")

roc_auc = roc_auc_score(Y_test_tensor.numpy(), probs)
print(f"ROC AUC: {roc_auc:.4f}")

probs = model(X_test_tensor).squeeze() 


plt.figure(figsize=(8, 6))
plt.plot(recall, precision, linewidth=2, label=f'Model 3 Curve (AUPRC={auprc:.4f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout()


# --------------------------------------------------


# MODEL 4: running evo2 embed + log regress  ---------------------


batch_size = 1024
seqs, labels, seq_ids = [], [], []

for record in SeqIO.parse("scratch/evo2/working/orfs_to_score.fa", "fasta"):
    seqs.append(str(record.seq).upper())
    header = record.id.split('|')
    labels.append(int(header[-1].split('=')[1]))
    seq_ids.append(record.id)

print(f"Loaded {len(seqs)} sequences")

# loading saved embeddings
all_embeds = np.concatenate([np.load(f'scratch/evo2/working/embeds/layer_28/batch_{i}.npy') for i in range(0, len(seqs), batch_size)])


# lr = LogisticRegression(max_iter=2000)
# lr.fit(all_embeds, labels)

# joblib.dump(lr, 'logistic_regression_model.joblib')
# saving bc takes forever
lrrr = joblib.load('logistic_regression_model.joblib')


pred = lrrr.predict(all_embeds)
proba = lrrr.predict_proba(all_embeds)[:, 1]

accuracy = accuracy_score(labels, pred)
auc = roc_auc_score(labels, proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")


accuracy = accuracy_score(labels, pred)
roc_auc = roc_auc_score(labels, proba)

precision, recall, thresholds = precision_recall_curve(labels, proba)
auprc = average_precision_score(labels, proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"AUPRC: {auprc:.4f}")

plt.plot(recall, precision, linewidth=2, label=f'Model 4 Curve (AUPRC={auprc:.4f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout()


# ------------------------------------



# MODEL 5: Running evo2 embed + mlp ------------------------------------------


scaler_eng = StandardScaler()
X_train_eng_scaled = scaler_eng.fit_transform(X_train)
X_test_eng_scaled = scaler_eng.transform(X_test)


batch_size = 1024
seqs, labels, seq_ids = [], [], []

for record in SeqIO.parse("scratch/evo2/working/orfs_to_score.fa", "fasta"):
    seqs.append(str(record.seq).upper())
    header = record.id.split('|')
    labels.append(int(header[-1].split('=')[1]))
    seq_ids.append(record.id)

print(f"Loaded {len(seqs)} sequences")

scaler_evo2 = StandardScaler()
all_embeds = np.concatenate([np.load(f'scratch/evo2/working/embeds/layer_28/batch_{i}.npy') for i in range(0, len(seqs), batch_size)])
#  all_embeds is a full embeddings array (n_samples, 4096)

evo2_train_scaled = scaler_evo2.fit_transform(all_embeds[:len(X_train)])
evo2_test_scaled = scaler_evo2.transform(all_embeds[len(X_train):])

X_train_combined = np.hstack([X_train_eng_scaled, evo2_train_scaled])
X_test_combined = np.hstack([X_test_eng_scaled, evo2_test_scaled])

X_train_combined[:, :82] *= 2  # scaling the engineered features by 2
X_test_combined[:, :82] *= 2

print(f"Combined training shape: {X_train_combined.shape}")  #(n, 82+4096)


X_train_tensor = torch.tensor(X_train_combined, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_combined, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32).unsqueeze(1)

train_ds = TensorDataset(X_train_tensor, Y_train_tensor)
train_dl = DataLoader(train_ds, batch_size=512, shuffle=True, pin_memory=True, num_workers=0)

class MLP(nn.Module):
    def __init__(self, in_dim=4178):  # 82 engineered + 4096 embeddings
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

model = MLP(in_dim=X_train_combined.shape[1])
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.BCELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(100):
    model.train()
    total_loss = 0
    for xb, yb in train_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"epoch {epoch+1:2d} | loss: {total_loss/len(train_dl):.4f}")

model.eval()
with torch.no_grad():
    X_test_tensor = X_test_tensor.to(device)
    preds = (model(X_test_tensor) > 0.5).float().squeeze().cpu().numpy()


print(classification_report(Y_test_tensor.numpy(), preds, target_names=['noise','gene']))


model.eval()
with torch.no_grad():
    X_test_tensor = X_test_tensor.to(device)
    probs = model(X_test_tensor).squeeze().cpu().numpy()

precision, recall, thresholds = precision_recall_curve(Y_test_tensor.numpy(), probs)


auprc = average_precision_score(Y_test_tensor.numpy(), probs)

print(f"AUPRC: {auprc:.4f}")

roc_auc = roc_auc_score(Y_test_tensor.numpy(), probs)
print(f"ROC AUC: {roc_auc:.4f}")

probs = model(X_test_tensor).squeeze() 


plt.plot(recall, precision, linewidth=2, label=f'Model 5 (AUPRC={auprc:.4f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.9)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig('pr_curve.png', dpi=300)
plt.show()
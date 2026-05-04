import os
import time
import gc
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)

# folders
data_folder  = './processed'
model_folder = './models'

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

# columns that are not features (we drop these before training)
skip_cols = ['species', 'chrom', 'strand', 'start_codon',
             'stop_codon', 'dna_sequence', 'label']


# load training data
print('loading train set...')
df_train = pd.read_parquet(data_folder + '/train_set.parquet')

# drop string columns, keep only numeric features
cols_to_drop = [c for c in skip_cols if c in df_train.columns]
X_all = df_train.drop(columns=cols_to_drop)
Y_all = df_train['label']
del df_train
gc.collect()
print('train shape: ' + str(X_all.shape))


# load test data (we only use this at the very end)
print('loading test set...')
df_test = pd.read_parquet(data_folder + '/test_set_300k.parquet')
cols_to_drop = [c for c in skip_cols if c in df_test.columns]
X_test = df_test.drop(columns=cols_to_drop)
Y_test = df_test['label']
del df_test
gc.collect()
print('test shape: ' + str(X_test.shape))


# split train into 80% train and 20% validation
print('\nsplitting into train/val...')
X_train, X_val, Y_train, Y_val = train_test_split(
    X_all, Y_all,
    test_size=0.20,
    random_state=42,
    stratify=Y_all,
)
del X_all, Y_all
gc.collect()

print('X_train: ' + str(X_train.shape) + '  pos=' + str((Y_train==1).sum()) + '  neg=' + str((Y_train==0).sum()))
print('X_val:   ' + str(X_val.shape)   + '  pos=' + str((Y_val==1).sum())   + '  neg=' + str((Y_val==0).sum()))
print('X_test:  ' + str(X_test.shape)  + '  pos=' + str((Y_test==1).sum())  + '  neg=' + str((Y_test==0).sum()))


# compute a bunch of metrics for a model on one split
def get_metrics(model, X, Y, split_name, use_proba=True):
    print('\n  --- ' + split_name + ' ---')

    preds = model.predict(X)

    # get probability scores for auroc/auprc
    if use_proba:
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.decision_function(X)

    acc      = accuracy_score(Y, preds)
    bal_acc  = balanced_accuracy_score(Y, preds)
    prec     = precision_score(Y, preds, zero_division=0)
    rec      = recall_score(Y, preds, zero_division=0)
    f1       = f1_score(Y, preds, average='macro')
    f1_gene  = f1_score(Y, preds, pos_label=1, zero_division=0)
    f1_noise = f1_score(Y, preds, pos_label=0, zero_division=0)
    mcc      = matthews_corrcoef(Y, preds)
    auroc    = roc_auc_score(Y, probs)
    auprc    = average_precision_score(Y, probs)

    tn, fp, fn, tp = confusion_matrix(Y, preds).ravel()

    # accuracy can be misleading if classes are unbalanced
    print('    Accuracy:          ' + str(round(acc,      4)) + '  (misleading if imbalanced)')
    # balanced accuracy accounts for class imbalance
    print('    Balanced Accuracy: ' + str(round(bal_acc,  4)) + '  (corrects for imbalance)')
    # precision = of predicted genes, how many are real
    print('    Precision:         ' + str(round(prec,     4)) + '  (of predicted genes, how many real?)')
    # recall = of real genes, how many did we find
    print('    Recall:            ' + str(round(rec,      4)) + '  (of real genes, how many caught?)')
    print('    F1 macro:          ' + str(round(f1,       4)) + '  (harmonic mean, both classes)')
    print('    F1 gene class:     ' + str(round(f1_gene,  4)) + '  (F1 for gene class only)')
    print('    F1 noise class:    ' + str(round(f1_noise, 4)) + '  (F1 for noise class only)')
    # mcc is the most honest metric, not fooled by imbalance
    print('    MCC:               ' + str(round(mcc,      4)) + '  (most honest, -1 to +1, 0=random)')
    # auroc 0.5 = random, 1.0 = perfect
    print('    AUROC:             ' + str(round(auroc,    4)) + '  (0.5=random, 1.0=perfect)')
    # auprc is better than auroc when classes are very unbalanced
    print('    AUPRC:             ' + str(round(auprc,    4)) + '  (better than AUROC for imbalanced)')
    print('    Confusion matrix:')
    print('      TP (gene caught):   ' + str(tp))
    print('      FN (gene missed):   ' + str(fn) + '  <- silent loss')
    print('      FP (noise as gene): ' + str(fp) + '  <- false alarm')
    print('      TN (noise caught):  ' + str(tn))

    res = {}
    res['split']     = split_name
    res['accuracy']  = round(acc,      4)
    res['bal_acc']   = round(bal_acc,  4)
    res['precision'] = round(prec,     4)
    res['recall']    = round(rec,      4)
    res['f1_macro']  = round(f1,       4)
    res['f1_gene']   = round(f1_gene,  4)
    res['f1_noise']  = round(f1_noise, 4)
    res['mcc']       = round(mcc,      4)
    res['auroc']     = round(auroc,    4)
    res['auprc']     = round(auprc,    4)
    res['tp']        = tp
    res['fn']        = fn
    res['fp']        = fp
    res['tn']        = tn
    return res


# evaluate on all three splits
def run_eval(model, name, use_proba=True):
    print('\n' + '='*60)
    print(name)
    print('='*60)
    all_results = []
    all_results.append(get_metrics(model, X_train, Y_train, 'Train',      use_proba))
    all_results.append(get_metrics(model, X_val,   Y_val,   'Validation', use_proba))
    all_results.append(get_metrics(model, X_test,  Y_test,  'Test (final, unseen)', use_proba))
    return all_results


# --- decision tree ---
print('\n' + '='*60)
print('training decision tree...')
t0 = time.time()

dt = DecisionTreeClassifier(
    max_depth=10,
    min_samples_leaf=20,
    class_weight='balanced',
    random_state=42,
)
dt.fit(X_train, Y_train)
print('done in ' + str(round(time.time() - t0, 1)) + 's')

dt_results = run_eval(dt, 'DECISION TREE')

# save the model
joblib.dump(dt, model_folder + '/decision_tree.pkl')
print('saved -> models/decision_tree.pkl')

del dt
gc.collect()


# --- random forest ---
print('\n' + '='*60)
print('training random forest...')
t0 = time.time()

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, Y_train)
print('done in ' + str(round(time.time() - t0, 1)) + 's')

rf_results = run_eval(rf, 'RANDOM FOREST')

# save the model
joblib.dump(rf, model_folder + '/random_forest.pkl')
print('saved -> models/random_forest.pkl')

# print top 20 most important features
print('\ntop 20 features (random forest):')
feat_scores = pd.Series(rf.feature_importances_, index=X_train.columns)
feat_scores = feat_scores.sort_values(ascending=False)
top20 = feat_scores.head(20)
for feat, score in top20.items():
    bar = '#' * int(score * 300)
    print('  ' + feat.ljust(20) + '  ' + str(round(score, 5)).ljust(9) + bar)

del rf
gc.collect()


# --- compare both models on test set ---
print('\n' + '='*60)
print('FINAL COMPARISON - TEST SET ONLY')
print('='*60)

metric_names = ['accuracy', 'bal_acc', 'precision', 'recall',
                'f1_macro', 'f1_gene', 'mcc', 'auroc', 'auprc']

# grab only the test row from each model's results
dt_test = None
for r in dt_results:
    if r['split'] == 'Test (final, unseen)':
        dt_test = r

rf_test = None
for r in rf_results:
    if r['split'] == 'Test (final, unseen)':
        rf_test = r

header = 'Metric              Decision Tree   Random Forest'
print(header)
print('-' * len(header))

for m in metric_names:
    print(m.ljust(20) + str(dt_test[m]).ljust(16) + str(rf_test[m]))


# save all results to csv
all_rows = []
for r in dt_results:
    r['model'] = 'Decision Tree'
    all_rows.append(r)
for r in rf_results:
    r['model'] = 'Random Forest'
    all_rows.append(r)

results_df = pd.DataFrame(all_rows)
results_df.to_csv(model_folder + '/results.csv', index=False)
print('\nresults saved -> models/results.csv')
print('all done.')
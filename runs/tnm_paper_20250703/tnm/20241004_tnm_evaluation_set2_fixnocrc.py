"""Evaluate the TNM extraction algorithm on future test data

Note that in future test data, there are three reports labelled with T0 staging for skin cancer,
while the actual staging of these tumours was other than T0. 
The 'T0' label was used because the report did not describe a colorectal tumour,
but the label should have been 'null' because these pathology reports do not give information about CRC tumours.

The staging of these non-CRC tumours is assigned to be 'null' in this analysis,
because we only want the algorithm to pick up stages for CRC tumours,
and the stages extracted for non-CRC tumours would be considered as errors.
However, later we will also evaluate how well the staging given in letters and numbers is picked up overall,
regardless of whether the staging is for historical or non-CRC tumours.

2024-10-04
Comment updated for better clarity on 2025-07-17
"""
import pandas as pd
import numpy as np
from pathlib import Path
from textmining.constants import PROJECT_ROOT
from textmining.evaluate import evaluate_tnm, _prepare_data


# Identify reports that discuss skin cancer but have T stage set to 0 instead of null
out_dir = PROJECT_ROOT / 'labelled_data' / 'processed' 
eval_path = out_dir / 'set2_tnm_20230815_pnfix.csv'
truth_path = PROJECT_ROOT / 'labelled_data' / 'processed' / 'set2_tnm_labelled_HJ_AT_20240214.csv' 
assert truth_path.exists
assert eval_path.exists

df0 = pd.read_csv(truth_path)
df1 = pd.read_csv(eval_path)

mask = (df0['T'] == '0') & (~df1['T'].isin(['0', 'null'])) & df0.note.str.lower().str.contains('skin')
print(mask.sum())
print(df0.loc[mask, 'T']) 
print(df1.loc[mask, 'T'])

for i, row in df0.loc[mask].iterrows():
    print('\n-----', row.note)
    print(repr(row.report_text_anon))

print(df0.loc[mask, 'T'])
print(df1.loc[mask, 'T'])  # in two cases, staging was prediced as 1, in one case as 2
print(df0['T'].value_counts())
df0['T'].unique()
df0.loc[mask, 'T'] = ' '  # use ' ' for missing value as previously in that data
print(df0.loc[mask, 'T'])
print(df0['T'].value_counts())
df0.loc[mask, 'report_type']

# save updated reports to disk
out_path = PROJECT_ROOT / 'labelled_data' / 'processed' / 'set2_tnm_labelled_HJ_AT_AT_20241004.csv' 
df0.to_csv(out_path, index=False)

# evaluate again
evaluate_tnm(out_path, eval_path, split='test-skinfix', brc='ouh')

# Display results
d = pd.read_csv(r'Z:\Andres\project_textmining\textmining\results\results-tnm_sum-print_brc-ouh_split-test-skinfix.csv')
d[['report_type', 'tnm_cat', 'ppv', 'sens']]
d.loc[(d.report_type=='pathology_future') & (d.tnm_cat == 'T')].transpose()


# ------- Dbl check sensitivity for T when excluding inferred 0 or X staging --------
from textmining.evaluate import _prepare_data
eval_path = out_dir / 'set2_tnm_20230815_pnfix.csv'
truth_path = PROJECT_ROOT / 'labelled_data/processed' / 'set2_tnm_labelled_HJ_AT_AT_20241004.csv'  # 'set2_tnm_labelled_HJ_20240203.csv'  # 

df0, df1 = _prepare_data(truth_path, eval_path, cols=['T', 'N', 'M', 'T_pre'])
df0, df1 = df0[['report_type', 'T', 'N', 'M', 'T_pre', 'report_text_anon', 'note']], df1[['report_type', 'T', 'N', 'M', 'T_pre']]
df0.columns = [c + '_true' if c in ['T', 'N', 'M', 'T_pre'] else c for c in df0.columns]
df1.columns = [c + '_pred' if c in ['T', 'N', 'M', 'T_pre'] else c for c in df1.columns]
assert (df0.report_type == df1.report_type).all()
df = pd.concat(objs=[df0, df1.drop(labels='report_type', axis=1)], axis=1)

# Get pathology reports
dfsub = df.loc[df.report_type == 'pathology_future'].copy()

# Original sensitivity
test = dfsub.T_true == dfsub.T_pred
test[dfsub.T_true != 'null'].mean() * 100

# Get cases where reports do not contain explicity T0 or TX stage but where it was inferred
mask = (dfsub.T_pred == 'null') & (dfsub.T_true.isin(['0', 'x']))
mask.sum()

# Dbl check that these reports do not indeed contain explicity TNM stage
dfsub.report_text_anon = dfsub.report_text_anon.str.replace(r'\n+|\r+', ' <n> ', regex=True)
esub = dfsub.loc[mask].sort_values(by='T_true')
esub.report_text_anon = esub.report_text_anon.str.replace(r'\n+|\r+', ' <n> ', regex=True)
esub = esub.sort_values(by='T_true')
for j, (i, row) in enumerate(esub.iterrows()):
    print('\n\n-----', j, row.T_true, row.T_pred, '|', row.note)
    print(row.report_text_anon)

from textmining.utils import extract
import re
extract(esub, 'report_text_anon', 't\W*\d', flags=re.I|re.DOTALL, pad_left=40, pad_right=10)
extract(esub, 'report_text_anon', r't\W*is\b', flags=re.I|re.DOTALL, pad_left=10, pad_right=10)

# Sensitivity after excluding cases where T inferred
dfsub.loc[mask, 'T_true'] = 'null'
test = dfsub.T_true == dfsub.T_pred
test[dfsub.T_true != 'null'].mean() * 100 



# ------- Sensitivty for T and M with CIs, when including inferred staging 0/X for T and 0/1/(X) for M --------
out_dir = PROJECT_ROOT / 'labelled_data' / 'processed' 
eval_path = out_dir / 'set2_tnm_20230815_pnfix.csv'
truth_path = PROJECT_ROOT / 'labelled_data' / 'processed' / 'set2_tnm_labelled_HJ_AT_AT_20241004.csv' 
assert truth_path.exists
assert eval_path.exists

df0 = pd.read_csv(truth_path)
df1 = pd.read_csv(eval_path)

df0['T'].unique()
df1['T'].unique()
df0['M'].unique()
df1['M'].unique()

mask = df0['T'].isin(['0', 'X']) & (df1['T'].isna())
mask.sum()
df0.loc[mask, 'T'] = ' '

df0.M.unique()
mask = (df0['M'].isin(['0', 'X'])) & (df1['M'].isna())
mask.sum()
df0.loc[mask, 'M'] = ' '

# save updated reports to disk
out_path = PROJECT_ROOT / 'labelled_data' / 'processed' / 'set2_tnm_labelled_HJ_AT_AT_no-implicit-t0x-m0x_20241004.csv' 
df0.to_csv(out_path, index=False)

# evaluate again
evaluate_tnm(out_path, eval_path, split='test-skinfix-noimplicit', brc='ouh')

d = pd.read_csv(r'Z:\Andres\project_textmining\textmining\results\results-tnm_sum-print_brc-ouh_split-test-skinfix-noimplicit.csv')
d.loc[d.tnm_cat.isin(['T', 'M']), ['report_type', 'tnm_cat', 'ppv', 'sens']]



# ----------- TNM confusion matrix for all data --------
from textmining.evaluate import _prepare_data
eval_path = out_dir / 'set2_tnm_20230815_pnfix.csv'
truth_path = PROJECT_ROOT / 'labelled_data/processed' / 'set2_tnm_labelled_HJ_AT_AT_20241004.csv'  # 'set2_tnm_labelled_HJ_20240203.csv'  # 

df0, df1 = _prepare_data(truth_path, eval_path, cols=['T', 'N', 'M'])
df0, df1 = df0[['report_type', 'T', 'N', 'M', 'report_text_anon', 'note']], df1[['report_type', 'T', 'N', 'M']]
df0.columns = [c + '_true' if c in ['T', 'N', 'M'] else c for c in df0.columns]
df1.columns = [c + '_pred' if c in ['T', 'N', 'M'] else c for c in df1.columns]
assert (df0.report_type == df1.report_type).all()
df = pd.concat(objs=[df0, df1.drop(labels='report_type', axis=1)], axis=1)

save_path = PROJECT_ROOT / 'results'
cmat = pd.DataFrame()
for report_type in ['pathology_future', 'imaging_future']:
    for category in ['T', 'N', 'M']:
        col_true = category + '_true'
        col_pred = category + '_pred'
        esub = df.loc[(df.report_type == report_type)].copy()
        esub[col_true] = esub[col_true].str.replace('[a-d]', '', regex=True)
        esub[col_pred] = esub[col_pred].str.replace('[a-d]', '', regex=True)
                
        esub = esub.sort_values(by=col_true)
        v = esub[[col_true, col_pred]].value_counts(sort=False).reset_index()
        v.columns = ['true_value', 'predicted_value', 'count']
        v['category'] = category
        v['report_type'] = report_type
        v = v[['report_type', 'category'] + [c for c in v.columns if c not in ['category', 'report_type']]]
        cmat = pd.concat(objs=[cmat, v], axis=0)
cmat
cmat.to_csv(save_path / 'results-tnm_confusion-tnm_split-test-skinfix.csv', index=False)
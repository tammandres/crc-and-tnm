"""Evaluate TNM stage extraction algorithm on future test data"""
import pandas as pd
import numpy as np
from pathlib import Path
from textmining.constants import PROJECT_ROOT
from textmining.evaluate import evaluate_tnm
from textmining.tnm.tnm import get_tnm_phrase, get_tnm_values
from textmining.tnm.clean import add_tumour_tnm
import time


# ---- 1. Evaluate main algorithm ----
#region
out_dir = PROJECT_ROOT / 'labelled_data' / 'processed' 
include_bug_fix = True

if include_bug_fix:
    rerun = False
    if rerun:
        tic = time.time()

        # Run again with pn bug fixed
        df = pd.read_csv(out_dir / 'set2_tnm_20230815.csv')
        df = df[['brc', 'subject_id', 'imaging_date', 'report_date', 'imaging_code', 'report_text_anon', 'report_type',
                'crc_nlp', 'false_crc_nlp']]

        matches_tnm, check_phrases_tnm, check_cleaning_tnm, check_rm_tnm = get_tnm_phrase(
            df=df, col='report_text_anon', remove_unusual=True, remove_historical=False, 
            remove_falsepos=True, extract_solitary=True
            )

        matches_tnm = add_tumour_tnm(df, matches_tnm, col_report='report_text_anon', targetcol='target_before_clean')

        df, check_values_tnm = get_tnm_values(df, matches=matches_tnm, col='report_text_anon', pathology_prefix=False)

        toc = time.time()
        print(toc - tic)

        # Save 
        eval_path = out_dir / 'set2_tnm_20230815_pnfix.csv'
        df.to_csv(eval_path, index=False)
    else:
        eval_path = out_dir / 'set2_tnm_20230815_pnfix.csv'
else:
    eval_path = out_dir / 'set2_tnm_20230815.csv'

# Evaluate
truth_path = PROJECT_ROOT / 'labelled_data/processed' / 'set2_tnm_labelled_HJ_AT_20240214.csv'  # 'set2_tnm_labelled_HJ_20240203.csv'  # 
assert truth_path.exists
assert eval_path.exists
evaluate_tnm(truth_path, eval_path, split='test', brc='ouh')

# Display results
d = pd.read_csv(r'Z:\Andres\project_textmining\textmining\results\results-tnm_sum-print_brc-ouh_split-test.csv')
d2 = pd.read_csv(r'Z:\Andres\project_textmining\textmining\results\results-tnm_cat-print_brc-ouh_split-test.csv')

cpath = pd.read_csv(r'Z:\Andres\project_textmining\textmining\results\results-tnm_confusion-t-pathology_future_brc-ouh_split-test.csv')
cimg = pd.read_csv(r'Z:\Andres\project_textmining\textmining\results\results-tnm_confusion-t-imaging_future_brc-ouh_split-test.csv')


pd.set_option('display.min_rows', 500, 'display.max_rows', 500)
d
d2

# Look at confusion matrix
t = pd.read_csv(r'Z:\Andres\project_textmining\textmining\results\results-tnm_true-pred_brc-ouh_split-test.csv',
                na_values=None, keep_default_na=False)
t.groupby('T_true')['T_pred'].value_counts()
t.groupby('T_pred')['T_true'].value_counts()             

# Look at errors
e = pd.read_csv(r'Z:\Andres\project_textmining\textmining\results\results-tnm_errors_brc-ouh_split-test.csv',
                na_values=None, keep_default_na=False)
print(e.T_true.unique(), e.T_pred.unique())
print(e.N_true.unique(), e.N_pred.unique())


# T - all errors, 78 instances
esub = e.loc[(e.T_true != e.T_pred)]
print(esub.shape)

esub.groupby('T_true')['T_pred'].value_counts()
esub.groupby('T_pred')['T_true'].value_counts()

esub.report_text_anon = esub.report_text_anon.str.replace(r'\n+|\r+', ' <n> ', regex=True).str.replace(r'\-+', ' ', regex=True)
for i, row in esub.iterrows():
    print('\n\n-----', row.row_num, row.T_true, row.T_pred)
    print(row.report_text_anon)


# T - where true T stage is 0 or x, 73 instances. 
# Mostly predicted as null because not explicitly given staging. 
#  [comment about individual reports removed - AT 20250721]
esubsub = esub.loc[esub.T_true.isin(['0', 'x'])]
print(esubsub.shape)
esubsub.groupby('T_true')['T_pred'].value_counts()

print(esubsub.shape)
esubsub.report_text_anon = esubsub.report_text_anon.str.replace(r'\n+|\r+', ' <n> ', regex=True).str.replace(r'\-+', ' ', regex=True)
for i, row in esubsub.iterrows():
    print('\n\n-----', row.row_num + 1, row.T_true, row.T_pred)
    print(row.report_text_anon)


# T - where true T stage is 0 or x: 7 instances where it was mispredicted as a higher stage
#  [comment about individual reports removed - AT 20250721]
esubsub = esubsub.loc[~esubsub.T_pred.isin(['0', 'null', 'x'])]
print(esubsub.shape)
esubsub.report_text_anon = esubsub.report_text_anon.str.replace(r'\n+|\r+', ' <n> ', regex=True).str.replace(r'\-+', ' ', regex=True)
for i, row in esubsub.iterrows():
    print('\n\n-----', row.row_num + 1, row.T_true, row.T_pred)
    print(row.report_text_anon)


# T - where true T stage is NOT 0 or x, 5 instances
esubsub = esub.loc[~esub.T_true.isin(['0', 'x'])]
print(esubsub.shape)
esubsub.report_text_anon = esubsub.report_text_anon.str.replace(r'\n+|\r+', ' <n> ', regex=True).str.replace(r'\-+', ' ', regex=True)
for i, row in esubsub.iterrows():
    print('\n\n-----', row.row_num, row.T_true, row.T_pred)
    print(row.report_text_anon)


esubsub = esub.loc[esub.T_true == 'null']
print(esubsub.shape)
esubsub.report_text_anon = esubsub.report_text_anon.str.replace(r'\n+|\r+', ' <n> ', regex=True).str.replace(r'\-+', ' ', regex=True)
for i, row in esubsub.iterrows():
    print('\n\n-----', row.row_num, row.T_true, row.T_pred)
    print(row.report_text_anon)


esub = e.loc[(e.N_true != e.N_pred)]
esub.report_text_anon = esub.report_text_anon.str.replace(r'\n+|\r+', ' <n> ', regex=True).str.replace(r'\-+', ' ', regex=True)
for i, row in esub.iterrows():
    print('\n\n-----', row.row_num, row.N_true, row.N_pred)
    print(row.report_text_anon)


# Examine cases where no T stage in numbers: indeed no numbers in 10 of 10 cases (T0 - inferred)
esub = e.loc[(e.T_pred != e.T_true) & (e.T_true != 'null') & (e.report_type=='pathology')].copy()
esub.report_text_anon = esub.report_text_anon.str.replace(r'\n+|\r+', ' <n> ', regex=True)
for j, (i, row) in enumerate(esub.iterrows()):
    print('\n\n-----', j, row.T_true, row.T_pred)
    print(row.report_text_anon)

esub = e.loc[(e.T_pred != e.T_true) & (e.T_true != 'null') & (e.report_type=='imaging')].copy()
esub.report_text_anon = esub.report_text_anon.str.replace(r'\n+|\r+', ' <n> ', regex=True)
for j, (i, row) in enumerate(esub.iterrows()):
    print('\n\n-----', j, row.T_true, row.T_pred)
    print(row.report_text_anon)


# Examine cases where no M stage in numbers: indeed 6 of 7 (1 of 7 error)
esub = e.loc[(e.M_pred != e.M_true) & (e.M_true != 'null') & (e.report_type=='imaging')].copy()
esub.report_text_anon = esub.report_text_anon.str.replace(r'\n+|\r+', ' <n> ', regex=True)
for j, (i, row) in enumerate(esub.iterrows()):
    print('\n\n-----', j, row.M_true, row.M_pred)
    print(row.report_text_anon)


# Examine false pos
esub = e.loc[(e.T_true != e.T_pred) & (e.T_pred != 'null')]
esub.report_text_anon = esub.report_text_anon.str.replace(r'\n+|\r+', ' <n> ', regex=True).str.replace(r'\-+', ' ', regex=True)
for i, row in esub.iterrows():
    print('\n\n-----', row.row_num, row.T_true, row.T_pred, row.report_type)
    print(row.report_text_anon)

#endregion

# ---- 2. Confusion matrix (2024-08-19) ----
#region

# TNM confusion matrix for errors
e = pd.read_csv(r'Z:\Andres\project_textmining\textmining\results\results-tnm_errors_brc-ouh_split-test.csv',
                na_values=None, keep_default_na=False)
save_path = PROJECT_ROOT / 'results'
cmat = pd.DataFrame()
for report_type in ['pathology_future', 'imaging_future']:
    for category in ['T', 'N', 'M']:
        col_true = category + '_true'
        col_pred = category + '_pred'
        esub = e.loc[(e[col_true] != e[col_pred]) & (e.report_type == report_type)]
        esub = esub.sort_values(by=col_true)
        v = esub[[col_true, col_pred]].value_counts(sort=False).reset_index()
        v.columns = ['true_value', 'predicted_value', 'count']
        v['category'] = category
        v['report_type'] = report_type
        v = v[['report_type', 'category'] + [c for c in v.columns if c not in ['category', 'report_type']]]
        cmat = pd.concat(objs=[cmat, v], axis=0)
cmat
cmat.to_csv(save_path / 'results-tnm_confusion-tnm_split-test_errors.csv', index=False)

# TNM confusion matrix for all data
from textmining.evaluate import _prepare_data
eval_path = out_dir / 'set2_tnm_20230815_pnfix.csv'
truth_path = PROJECT_ROOT / 'labelled_data/processed' / 'set2_tnm_labelled_HJ_AT_20240214.csv'  # 'set2_tnm_labelled_HJ_20240203.csv'  # 

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
cmat.to_csv(save_path / 'results-tnm_confusion-tnm_split-test.csv', index=False)
#endregion

# ---- 3. Deeper exploration of errors (2024-08-20) -----

# .... Prepare data
#region
from textmining.evaluate import _prepare_data
eval_path = out_dir / 'set2_tnm_20230815_pnfix.csv'
truth_path = PROJECT_ROOT / 'labelled_data/processed' / 'set2_tnm_labelled_HJ_AT_20240214.csv'  # 'set2_tnm_labelled_HJ_20240203.csv'  # 

df0, df1 = _prepare_data(truth_path, eval_path, cols=['T', 'N', 'M', 'T_pre'])
df0, df1 = df0[['report_type', 'T', 'N', 'M', 'T_pre', 'report_text_anon', 'note']], df1[['report_type', 'T', 'N', 'M', 'T_pre']]
df0.columns = [c + '_true' if c in ['T', 'N', 'M', 'T_pre'] else c for c in df0.columns]
df1.columns = [c + '_pred' if c in ['T', 'N', 'M', 'T_pre'] else c for c in df1.columns]
assert (df0.report_type == df1.report_type).all()
df = pd.concat(objs=[df0, df1.drop(labels='report_type', axis=1)], axis=1)
#endregion

# .... 3.1. Pathology T, reduction in sensitivity due to inferred staging ....
#region

## Get pathology reports
dfsub = df.loc[df.report_type == 'pathology_future'].copy()

## Original sensitivity
test = dfsub.T_true == dfsub.T_pred
test[dfsub.T_true != 'null'].mean() * 100

## Get cases where reports do not contain explicity T0 or TX stage but where it was inferred
mask = (dfsub.T_pred == 'null') & (dfsub.T_true.isin(['0', 'x']))
mask.sum()

## Dbl check that these reports do not indeed contain explicity TNM stage
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

esub.to_csv(save_path / 'results-tnm_errors-t0-and-tx_reports-pathology_split-test.csv', index=False)

## Sensitivity after excluding cases where T inferred
dfsub.loc[mask, 'T_true'] = 'null'
test = dfsub.T_true == dfsub.T_pred
test[dfsub.T_true != 'null'].mean() * 100 
#endregion

# .... 3.2. Imaging T, reduction in sensitivity due to inferred staging ....
#region

## Get imaging reports
dfsub = df.loc[df.report_type == 'imaging_future'].copy()

## Original sensitivity
test = dfsub.T_true == dfsub.T_pred
test[dfsub.T_true != 'null'].mean() * 100

## Get cases where reports do not contain explicity T0 or TX stage but where it was inferred
mask = (dfsub.T_pred == 'null') & (dfsub.T_true.isin(['0', 'x']))
mask.sum()

## Dbl check that these reports do not indeed contain explicity TNM stage
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

esub.to_csv(save_path / 'results-tnm_errors-t0-and-tx_reports-imaging_split-test.csv', index=False)

## Sensitivity after excluding cases where T inferred
dfsub.loc[mask, 'T_true'] = 'null'
test = dfsub.T_true == dfsub.T_pred
test[dfsub.T_true != 'null'].mean() * 100 
#endregion

# .... 3.3. Imaging M, reduction in sensitivity due to inferred staging ....
#region

## Original sensitivity
test = dfsub.M_true == dfsub.M_pred
test[dfsub.M_true != 'null'].mean() * 100

## Get cases where reports do not contain explicity M0 or MX stage but where it was inferred
mask = (dfsub.M_pred == 'null') & (dfsub.M_true != 'null')
mask.sum()
dfsub.loc[mask, 'M_true'].value_counts()

## Dbl check that these reports do not indeed contain explicity TNM stage
dfsub.report_text_anon = dfsub.report_text_anon.str.replace(r'\n+|\r+', ' <n> ', regex=True)
esub = dfsub.loc[mask].sort_values(by='T_true')
esub.report_text_anon = esub.report_text_anon.str.replace(r'\n+|\r+', ' <n> ', regex=True)
esub = esub.sort_values(by='M_true')
for j, (i, row) in enumerate(esub.iterrows()):
    print('\n\n-----', j, row.M_true, row.M_pred, '|', row.note)
    print(row.report_text_anon)

from textmining.utils import extract
import re
extract(esub, 'report_text_anon', 'm\W*\d', flags=re.I|re.DOTALL, pad_left=10, pad_right=10)
esub.to_csv(save_path / 'results-tnm_errors-m0-and-mx_reports-imaging_split-test.csv', index=False)

## Sensitivity after excluding cases where T inferred
dfsub.loc[mask, 'M_true'] = 'null'
test = dfsub.M_true == dfsub.M_pred
test[dfsub.M_true != 'null'].mean() * 100 
#endregion

# .... 3.4. Pathology T, reduction in PPV ....
#region

## Get pathology reports
dfsub = df.loc[df.report_type == 'pathology_future'].copy()

mask = (dfsub.T_pred != 'null') & (dfsub.T_true != dfsub.T_pred)
mask.sum()  # prediction errors when predicted value is not null

## Original PPV
test = dfsub.T_true == dfsub.T_pred
test[dfsub.T_pred != 'null'].mean() * 100

## Explore
#  [comment about individual reports removed - AT 20250721]
dfsub.report_text_anon = dfsub.report_text_anon.str.replace(r'\n+|\r+', ' <n> ', regex=True)
esub = dfsub.loc[mask].sort_values(by='T_true')
esub.report_text_anon = esub.report_text_anon.str.replace(r'\n+|\r+', ' <n> ', regex=True)
esub = esub.sort_values(by='T_true')
for j, (i, row) in enumerate(esub.iterrows()):
    print('\n\n-----', j, row.T_true, row.T_pred, '|', row.note)
    print(row.report_text_anon)
#endregion

# .... 3.5. Assign error categories ....
#region

assign_cat = False
if assign_cat:
    res = pd.DataFrame()
    for category in ['T', 'N', 'M']:
        print('=======', category)
        for report_type in ['imaging_future', 'pathology_future']:
            print('........', report_type)
            col_true = category + '_true'
            col_pred = category + '_pred'
            esub = df.loc[(df.report_type == report_type)].copy()
            esub = esub.sort_values(by=[col_true, col_pred])
            esub.report_text_anon = esub.report_text_anon.str.replace(r'\n+|\r+', ' <n> ', regex=True)
            esub = esub.loc[esub[col_true] != esub[col_pred]]
            esub = esub.reset_index(drop=True)
            if esub.shape[0] > 0:
                esub['error_category'] = np.nan
                for j, (i, row) in enumerate(esub.iterrows()):
                    print('\n\n-----', j, 'true', row[col_true], 'pred', row[col_pred], '|', row.note)
                    print(row.report_text_anon)
                    esub.loc[i, 'error_category'] = input("error cat:")
                r = esub[['report_text_anon', 'report_type', 'error_category']]
                r['tnm_category'] = category
                res = pd.concat(objs=[res, r], axis=0)       

    res.columns
    res.groupby(['report_type', 'tnm_category']).error_category.value_counts()
    res.to_csv(out_dir / 'results-tnm_errors-with-cat_brc-ouh_split-test.csv', index=False)
else:
    s = pd.read_csv(out_dir / 'results-tnm_errors-with-cat_brc-ouh_split-test.csv')
    s = s.groupby(['report_type', 'tnm_category']).error_category.value_counts().rename('count')
    s = s.reset_index()
    s.to_csv(out_dir / 'results-tnm_errors-summary_brc-ouh_split-test.csv', index=False)

#endregion

### 2024-10-04 Double check that total error count matches with total mismatch count
e = pd.read_csv(out_dir / 'results-tnm_errors-summary_brc-ouh_split-test.csv')
for cat in ['T', 'N', 'M']:
    for report_type in ['imaging_future', 'pathology_future']:
        esub = e.loc[(e.tnm_category==cat) & (e.report_type==report_type)]
        error_count = esub['count'].sum()
        col_true = cat + '_true'
        col_pred = cat + '_pred'
        dfsub = df.loc[df.report_type==report_type]
        msub = dfsub.loc[dfsub[col_true] != dfsub[col_pred]]
        mismatch_count = msub.shape[0] 
        print(cat, report_type, ': ecount', error_count, ', mcount:', mismatch_count)


# .... 3.6. Explore errors for T_pre ....
#region

df[['report_type', 'T_pre_true', 'T_pre_pred']].value_counts(sort=False)
df.loc[df.T_pre_pred == 'y'].report_text_anon.iloc[0]

#endregion


# ---- 4. Evaluate simplified algorithm ----
#region

# Run simplified version of alg
df = pd.read_csv(eval_path)
df = df[['brc', 'subject_id', 'imaging_date', 'report_date', 'imaging_code', 'report_text_anon', 'report_type',
          'crc_nlp', 'false_crc_nlp']]

tic = time.time()
#df = pd.DataFrame([text], columns=['report_text_anon'])
matches_tnm, check_phrases_tnm, check_cleaning_tnm, check_rm_tnm = get_tnm_phrase(
    df=df, col='report_text_anon', remove_unusual=True, remove_historical=False, 
    remove_falsepos=True, simplicity=0, extract_solitary=True
    )

# Add nearby tumour keywords (can help decide which tumour the TNM phrase refers to, if needed)
matches_tnm = add_tumour_tnm(df, matches_tnm, col_report='report_text_anon', targetcol='target_before_clean')

# Get TNM values from phrases
df, check_values_tnm = get_tnm_values(df, matches=matches_tnm, col='report_text_anon', pathology_prefix=False)
toc = time.time()
print(toc - tic)

# Save
eval_path = PROJECT_ROOT / 'labelled_data/processed' / 'set2_tnm_20240209_simplicity-0.csv'
df.to_csv(eval_path, index=False)

# Evaluate
evaluate_tnm(truth_path, eval_path, split='test', brc='ouh', suffix='_simple')

# Inspect
d = pd.read_csv(r'Z:\Andres\project_textmining\textmining\results\results-tnm_sum-print_brc-ouh_split-test_simple.csv')
d2 = pd.read_csv(r'Z:\Andres\project_textmining\textmining\results\results-tnm_cat-print_brc-ouh_split-test_simple.csv')
d

e = pd.read_csv(r'Z:\Andres\project_textmining\textmining\results\results-tnm_errors_brc-ouh_split-test_simple.csv')

esub = e.loc[(~e.T_true.isna()) & (e.T_true != e.T_pred)]
esub.report_text_anon = esub.report_text_anon.str.replace(r'\n+|\r+', ' <n> ', regex=True).str.replace(r'\-+', ' ', regex=True)
for i, row in esub.iterrows():
    print('\n\n-----', i, row.T_true, row.T_pred)
    print(row.report_text_anon)


s = check_cleaning_tnm
s = s.loc[s.target_before_clean.str.lower().str.contains('signal')]
s
#endregion
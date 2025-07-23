"""Evaluate CRC detection

This script computes performance metrics for the CRC detection algorithm that were included
in the manuscript submitted to BMJ Health & Care Informatics.
It is also contains code that was used to perform an error analysis.

The version of the CRC algorithm from 20230206 commit (from github.com/tammandres/textmining) is used.
Compared to previous runs of the algorithm, it gives the same result on test data,
but a different result for one report on training data.

Furthermore, even though the 20230206 code is slightly worse than the version after,
it could not have received any inputs from the new FIT dataset that was exported in 2023
and on which I ran the code. This helps ensure future test data is independent from that run,
even though it is unlikely that I updated the code itself using the new FIT dataset.

Andres Tamm (AT)
2024-08-17

This comment above was updated on 2025-07-07
"""
import pandas as pd
import numpy as np
from pathlib import Path
from textmining.constants import PROJECT_ROOT, RESULTS_DIR
from textmining.evaluate import evaluate_crc
from textmining.reports_20230206_negbugfix import get_crc_reports


# ---- 1. Rerun the algorithm ----
#region

# Train data
file_orig = 'set1_crc.csv'
file_labelled = 'set1_crc_labelled_AT-HJ-AT_20231201.xlsx'

truth_path = PROJECT_ROOT / 'labelled_data/processed' / file_labelled
eval_path = PROJECT_ROOT / 'labelled_data' / file_orig

## Rerun CRC extraction 
df = pd.read_csv(eval_path)
print(df.groupby(['brc', 'report_type'])['crc_nlp'].sum())
__, matches_crc = get_crc_reports(df, 'report_text_anon', add_subj_to_matches=True, subjcol='subject_id', negation_bugfix=True)
df['row'] = np.arange(df.shape[0])
df['crc_nlp'] = 0
matches_incl = matches_crc.loc[matches_crc.exclusion_indicator==0]
df.loc[df.row.isin(matches_incl.row), 'crc_nlp'] = 1
print(df.groupby(['brc', 'report_type'])['crc_nlp'].sum())

eval_path_new = PROJECT_ROOT / 'labelled_data' / 'processed' / 'set1_crc_rerun_20240816.csv'
if eval_path != eval_path_new:
    df.to_csv(eval_path_new, index=False)

assert truth_path.exists
assert eval_path_new.exists

evaluate_crc(truth_path, eval_path_new, split='train-rerun_20240816', older_code=True)


# Test data (OUH future)
file_orig = 'set2_crc.csv'
file_labelled = 'set2_crc_labelled_AT-HJ-AT_20231201.xlsx'

truth_path = PROJECT_ROOT / 'labelled_data/processed' / file_labelled
eval_path = PROJECT_ROOT / 'labelled_data' / file_orig

## Rerun CRC extraction 
df = pd.read_csv(eval_path)
print(df.groupby(['brc', 'report_type'])['crc_nlp'].sum())
__, matches_crc = get_crc_reports(df, 'report_text_anon', add_subj_to_matches=True, subjcol='subject_id', negation_bugfix=True)
df['row'] = np.arange(df.shape[0])
df['crc_nlp'] = 0
matches_incl = matches_crc.loc[matches_crc.exclusion_indicator==0]
df.loc[df.row.isin(matches_incl.row), 'crc_nlp'] = 1
print(df.groupby(['brc', 'report_type'])['crc_nlp'].sum())

eval_path_new = PROJECT_ROOT / 'labelled_data' / 'processed' / 'set2_crc_rerun_20240816.csv'
if eval_path != eval_path_new:
    df.to_csv(eval_path_new, index=False)

assert truth_path.exists
assert eval_path_new.exists

evaluate_crc(truth_path, eval_path_new, split='test-rerun_20240816', older_code=True)


## In test data, mark supplementary MMR reports as crc_nlp = 0 if:
##  they have no included CRC matches
##  and the excluded CRC matches are not negated (as there are two cases when negation led to exclusion of valid matches)
## and re-evaluate
df0 = pd.read_excel(truth_path)
df0.note.str.lower().str.contains('mmr').sum()
mask = df0.report_text_anon.str.lower().str.contains('^\W*supplement.*mmr') & (df.crc_nlp == 0)
mask.sum()  ## 13 reports that are supplementary mmr and crc_nlp = 0

submatches = matches_crc.loc[matches_crc.row.isin(df.loc[mask].row)]
submatches.exclusion_reason.value_counts()
submatches = submatches.loc[submatches.exclusion_reason.str.contains('neg')]
for c in ['left', 'target', 'right']:
    submatches[c] = submatches[c].str.replace(r'\r', '<r>')
for i, row in submatches.iterrows(): ## 2 reports are falsely negated
    print('-----\n\n', row.left, '|', row.target, '|', row.right, '|', row.exclusion_reason)

mask = mask & ~ (df.row.isin(submatches.row))
mask.sum()
submatches = matches_crc.loc[matches_crc.row.isin(df.loc[mask].row)]
submatches.exclusion_reason.value_counts() # 11 reports, if they have any matches, are excluded due to no site and general and historic only

df0_sub = df0.loc[mask]
for i, row in df0_sub.iterrows():
    print('\n-----', row.report_text_anon)

df0.loc[mask, 'crc_nlp'].value_counts()
df0.loc[mask, 'crc_nlp'] = 0
truth_path_new = PROJECT_ROOT / 'labelled_data' / 'processed' / 'set2_crc_relabel_20240816.csv'
df0.to_csv(truth_path_new, index=False)
assert truth_path_new.exists
assert eval_path_new.exists

evaluate_crc(truth_path_new, eval_path_new, split='test-rerun-relabel_20240816', older_code=True)
pd.read_csv(r'Z:\Andres\project_textmining\textmining\results\results-crc_sum-print_brc-ouh_split-test-rerun-relabel_20240816.csv')


# Print results
r_train = pd.read_csv(r'Z:\Andres\project_textmining\textmining\results\results-crc_sum-print_brc-ouh_split-train-rerun_20240816.csv')
r_test = pd.read_csv(r'Z:\Andres\project_textmining\textmining\results\results-crc_sum-print_brc-ouh_split-test-rerun_20240816.csv')

r_train
r_test

#endregion


# ---- 2. Error analysis: test data ----
#region 

# .... 2.1. Future imaging reports ....

# Explore errors on future imaging reports (excluding the previous 11 MMR reports)
e_file = Path(r'Z:\Andres\project_textmining\textmining\results\results-crc_errors_brc-ouh_split-test-rerun_20240816.csv')
e = pd.read_csv(e_file)
print(e.shape)
#e['phrase'] = e.left.str.lower() + e.target.str.upper() + e.right.str.lower()
e['text'] = e.report_text_anon.str.replace(r'\n|\r', ' <n> ', regex=True)

e = e.loc[e.report_type == 'imaging_future']

__, matches_e = get_crc_reports(e, 'report_text_anon')
for c in ['left', 'target', 'right']:
    matches_e[c] = matches_e[c].str.replace('\n|\r', ' <n> ')
e['row'] = np.arange(e.shape[0])

# Assign error cat
assign_error_cat = False
if assign_error_cat:
    e['error_category'] = np.nan
    for i, row in e.iterrows():
        print('\n---report {}, true {}, pred {}, note {}'.format(row.row, row.crc_true, row.crc_pred, row.note))
        #print(row.text)
        submatches = matches_e.loc[matches_e.row.isin([row.row])]
        for j, row2 in submatches.iterrows():
            print('\n', row2.left, '<<', row2.target.upper(), '>>', row2.right, '|', row2.exclusion_reason)
        e.loc[i, 'error_category'] = input("Error cat: ")

    e.error_category.value_counts()
    e.row.nunique()
    e.row.shape

    e_file_new = Path(r'Z:\Andres\project_textmining\textmining\results\results-crc_errors-with-cat-imaging_brc-ouh_split-test-rerun_20240816.csv')
    e.to_csv(e_file_new, index=False)
else:
    e_file_new = Path(r'Z:\Andres\project_textmining\textmining\results\results-crc_errors-with-cat-imaging_brc-ouh_split-test-rerun_20240816.csv')
    e = pd.read_csv(e_file_new)
    e.error_category.value_counts()

    for i, row in e.iterrows():
        print('\n---report {}, true {}, pred {}, cat {}, note {}'.format(row.row, row.crc_true, row.crc_pred, row.error_category, row.note))
        #print(row.text)
        submatches = matches_e.loc[matches_e.row.isin([row.row])]
        for j, row2 in submatches.iterrows():
            print('\n', row2.left, '<<', row2.target.upper(), '>>', row2.right, '|', row2.exclusion_reason)
        input("press any key")

esub = e.loc[e.error_category == 'treatment_response']
for i, row in esub.iterrows():
    print('\n---report {}, true {}, pred {}, cat {}'.format(row.row, row.crc_true, row.crc_pred, row.error_category))
    print(row.text)
    submatches = matches_e.loc[matches_e.row.isin([row.row])]
    for j, row2 in submatches.iterrows():
        print('\n', row2.left, '<<', row2.target.upper(), '>>', row2.right, '|', row2.exclusion_reason)
    input("press any key")



#  [comment about individual reports removed - AT 20250721]


# Lack of PPV
esub = e.loc[e.crc_pred == 1]
esub.shape
for i, row in esub.iterrows():
    print('\n---report {}, true {}, pred {}, cat {}'.format(row.row, row.crc_true, row.crc_pred, row.error_category))
    #print(row.text)
    submatches = matches_e.loc[matches_e.row.isin([row.row])]
    for j, row2 in submatches.iterrows():
        print('\n', row2.left, '<<', row2.target.upper(), '>>', row2.right, '|', row2.exclusion_reason)
    input("press any key")

esub.error_category.value_counts()


# .... 2.2. Future pathology reports ....

e_file = Path(r'Z:\Andres\project_textmining\textmining\results\results-crc_errors_brc-ouh_split-test-rerun_20240816.csv')
e = pd.read_csv(e_file)
print(e.shape)
#e['phrase'] = e.left.str.lower() + e.target.str.upper() + e.right.str.lower()
e['text'] = e.report_text_anon.str.replace(r'\n|\r', ' <n> ', regex=True)

e = e.loc[e.report_type == 'pathology_future']

__, matches_e = get_crc_reports(e, 'report_text_anon')
for c in ['left', 'target', 'right']:
    matches_e[c] = matches_e[c].str.replace('\n|\r', ' <n> ')
e['row'] = np.arange(e.shape[0])

# Assign error cat
assign_error_cat = False
if assign_error_cat:
    e['error_category'] = np.nan
    for i, row in e.iterrows():
        print('\n---report {}, true {}, pred {}, note {}'.format(row.row, row.crc_true, row.crc_pred, row.note))
        print(row.text)
        submatches = matches_e.loc[matches_e.row.isin([row.row])]
        for j, row2 in submatches.iterrows():
            print('\n', row2.left, '<<', row2.target.upper(), '>>', row2.right, '|', row2.exclusion_reason)
        e.loc[i, 'error_category'] = input("Error cat: ")

    e.error_category.value_counts()
    e.row.nunique()
    e.row.shape

    e_file_new = Path(r'Z:\Andres\project_textmining\textmining\results\results-crc_errors-with-cat-pathology_brc-ouh_split-test-rerun_20240816.csv')
    e.to_csv(e_file_new, index=False)
else:
    e_file_new = Path(r'Z:\Andres\project_textmining\textmining\results\results-crc_errors-with-cat-pathology_brc-ouh_split-test-rerun_20240816.csv')
    e = pd.read_csv(e_file_new)
    e.error_category.value_counts()

    for i, row in e.iterrows():
        print('\n---report {}, true {}, pred {}, cat {}, note {}'.format(row.row, row.crc_true, row.crc_pred, row.error_category, row.note))
        #print(row.text)
        submatches = matches_e.loc[matches_e.row.isin([row.row])]
        for j, row2 in submatches.iterrows():
            print('\n', row2.left, '<<', row2.target.upper(), '>>', row2.right, '|', row2.exclusion_reason)
        input("press any key")

esub = e.loc[e.error_category == 'crc_site_missed']
for i, row in esub.iterrows():
    print('\n---report {}, true {}, pred {}, cat {}'.format(row.row, row.crc_true, row.crc_pred, row.error_category))
    #print(row.text)
    submatches = matches_e.loc[matches_e.row.isin([row.row])]
    for j, row2 in submatches.iterrows():
        print('\n', row2.left, '<<', row2.target.upper(), '>>', row2.right, '|', row2.exclusion_reason)
    input("press any key")


# .... 2.3. Explore performance when further gene test reports excluded (set to crc_nlp = 0) ....
file_labelled = 'set2_crc_labelled_AT-HJ-AT_20231201.xlsx'
truth_path = PROJECT_ROOT / 'labelled_data/processed' / file_labelled
eval_path_new = PROJECT_ROOT / 'labelled_data' / 'processed' / 'set2_crc_rerun_20240816.csv'

df0 = pd.read_excel(truth_path)
df0.note.str.lower().str.contains('mmr').sum()
mask = df0.report_text_anon.str.lower().str.contains('^\W*supplement.*mmr') & (df.crc_nlp == 0)
mask.sum()  ## 13 reports that are supplementary mmr and crc_nlp = 0

submatches = matches_crc.loc[matches_crc.row.isin(df.loc[mask].row)]
submatches.exclusion_reason.value_counts()
submatches = submatches.loc[submatches.exclusion_reason.str.contains('neg')]
for c in ['left', 'target', 'right']:
    submatches[c] = submatches[c].str.replace(r'\r', '<r>')
for i, row in submatches.iterrows(): ## 2 reports are falsely negated
    print('-----\n\n', row.left, '|', row.target, '|', row.right, '|', row.exclusion_reason)

mask = mask & ~ (df.row.isin(submatches.row))
mask.sum()
submatches = matches_crc.loc[matches_crc.row.isin(df.loc[mask].row)]
submatches.exclusion_reason.value_counts() # 11 reports, if they have any matches, are excluded due to no site and general and historic only

df0.loc[mask, 'crc_nlp'].value_counts()
df0.loc[mask, 'crc_nlp'] = 0

e_file_new = Path(r'Z:\Andres\project_textmining\textmining\results\results-crc_errors-with-cat-pathology_brc-ouh_split-test-rerun_20240816.csv')
e = pd.read_csv(e_file_new)
esub = e.loc[e.error_category == 'crc_implied_by_gene_test']
esub.report_text_anon = esub.report_text_anon.str.replace('\n|\r', ' <n> ')
for i, row in esub.iterrows(): ## 2 reports are falsely negated
    print('-----\n\n', row.report_text_anon)

mask = df0.report_text_anon.isin(esub.report_text_anon)
mask.sum()
df0.loc[mask, 'crc_nlp'] = 0

truth_path_new = PROJECT_ROOT / 'labelled_data' / 'processed' / 'set2_crc_relabel2_20240816.csv'
df0.to_csv(truth_path_new, index=False)
assert truth_path_new.exists
assert eval_path_new.exists

evaluate_crc(truth_path_new, eval_path_new, split='test-rerun-relabel2_20240816', older_code=True)
pd.read_csv(r'Z:\Andres\project_textmining\textmining\results\results-crc_sum-print_brc-ouh_split-test-rerun-relabel2_20240816.csv')


#endregion


# ---- 3. Error analysis: train data ----
#region 

# .... 3.1. Imaging reports ....

e_file = Path(r'Z:\Andres\project_textmining\textmining\results\results-crc_errors_brc-ouh_split-train-rerun_20240816.csv')
e = pd.read_csv(e_file)
print(e.shape)
#e['phrase'] = e.left.str.lower() + e.target.str.upper() + e.right.str.lower()
e['text'] = e.report_text_anon.str.replace(r'\n|\r', ' <n> ', regex=True)

e = e.loc[e.report_type == 'imaging']

__, matches_e = get_crc_reports(e, 'report_text_anon')
for c in ['left', 'target', 'right']:
    matches_e[c] = matches_e[c].str.replace('\n|\r', ' <n> ')
e['row'] = np.arange(e.shape[0])

# Assign error cat
assign_error_cat = False
if assign_error_cat:
    e['error_category'] = np.nan
    for i, row in e.iterrows():
        print('\n---report {}, true {}, pred {}, note {}'.format(row.row, row.crc_true, row.crc_pred, row.note))
        #print(row.text)
        submatches = matches_e.loc[matches_e.row.isin([row.row])]
        for j, row2 in submatches.iterrows():
            print('\n', row2.left, '<<', row2.target.upper(), '>>', row2.right, '|', row2.exclusion_reason)
        e.loc[i, 'error_category'] = input("Error cat: ")

    e.error_category.value_counts()
    e.error_category = e.error_category.replace({'possiblw': 'possible', 'crc_site_missed': 'falsely_assigned_noncrc_site'})
    e.row.nunique()
    e.row.shape

    e_file_new = Path(r'Z:\Andres\project_textmining\textmining\results\results-crc_errors-with-cat-imaging_brc-ouh_split-train-rerun_20240816.csv')
    e.to_csv(e_file_new, index=False)
else:
    e_file_new = Path(r'Z:\Andres\project_textmining\textmining\results\results-crc_errors-with-cat-imaging_brc-ouh_split-train-rerun_20240816.csv')
    e = pd.read_csv(e_file_new)
    e.error_category.value_counts()

    for i, row in e.iterrows():
        print('\n---report {}, true {}, pred {}, cat {}, note {}'.format(row.row, row.crc_true, row.crc_pred, row.error_category, row.note))
        #print(row.text)
        submatches = matches_e.loc[matches_e.row.isin([row.row])]
        for j, row2 in submatches.iterrows():
            print('\n', row2.left, '<<', row2.target.upper(), '>>', row2.right, '|', row2.exclusion_reason)
        input("press any key")

esub = e.loc[e.error_category == 'possible']
for i, row in esub.iterrows():
    print('\n---report {}, true {}, pred {}, cat {}'.format(row.row, row.crc_true, row.crc_pred, row.error_category))
    #print(row.text)
    submatches = matches_e.loc[matches_e.row.isin([row.row])]
    for j, row2 in submatches.iterrows():
        print('\n', row2.left, '<<', row2.target.upper(), '>>', row2.right, '|', row2.exclusion_reason)
    input("press any key")

# Lack of PPV
esub = e.loc[e.crc_pred == 1]
for i, row in esub.iterrows():
    print('\n---report {}, true {}, pred {}, cat {}'.format(row.row, row.crc_true, row.crc_pred, row.error_category))
    #print(row.text)
    submatches = matches_e.loc[matches_e.row.isin([row.row])]
    for j, row2 in submatches.iterrows():
        print('\n', row2.left, '<<', row2.target.upper(), '>>', row2.right, '|', row2.exclusion_reason)
    input("press any key")

esub.error_category.value_counts()

e_file = Path(r'Z:\Andres\project_textmining\textmining\results\results-crc_errors_brc-ouh_split-train-rerun_20240816.csv')
e = pd.read_csv(e_file)
e = e.loc[e.report_text_anon.str.lower().str.contains('am certain')]
e.report_text_anon.item()

e_file = Path(r'Z:\Andres\project_textmining\textmining\results\results-crc_errors_brc-ouh_split-test-rerun_20240816.csv')
e = pd.read_csv(e_file)
e = e.loc[e.report_text_anon.str.lower().str.contains('certain')]
e.report_text_anon.item()


# .... 3.2. Pathology reports ....

e_file = Path(r'Z:\Andres\project_textmining\textmining\results\results-crc_errors_brc-ouh_split-train-rerun_20240816.csv')
e = pd.read_csv(e_file)
print(e.shape)
#e['phrase'] = e.left.str.lower() + e.target.str.upper() + e.right.str.lower()
e['text'] = e.report_text_anon.str.replace(r'\n|\r', ' <n> ', regex=True)

e = e.loc[e.report_type == 'pathology']

__, matches_e = get_crc_reports(e, 'report_text_anon')
for c in ['left', 'target', 'right']:
    matches_e[c] = matches_e[c].str.replace('\n|\r', ' <n> ')
e['row'] = np.arange(e.shape[0])

# Assign error cat
assign_error_cat = False
if assign_error_cat:
    e['error_category'] = np.nan
    for i, row in e.iterrows():
        print('\n---report {}, true {}, pred {}, note {}'.format(row.row, row.crc_true, row.crc_pred, row.note))
        #print(row.text)
        submatches = matches_e.loc[matches_e.row.isin([row.row])]
        for j, row2 in submatches.iterrows():
            print('\n', row2.left, '<<', row2.target.upper(), '>>', row2.right, '|', row2.exclusion_reason)
        e.loc[i, 'error_category'] = input("Error cat: ")

    e.error_category.value_counts()
    e.error_category = e.error_category.replace({'gene_test_implies_crc': 'crc_implied_by_gene_test'})
    e.row.nunique()
    e.row.shape

    e_file_new = Path(r'Z:\Andres\project_textmining\textmining\results\results-crc_errors-with-cat-pathology_brc-ouh_split-train-rerun_20240816.csv')
    e.to_csv(e_file_new, index=False)
else:
    e_file_new = Path(r'Z:\Andres\project_textmining\textmining\results\results-crc_errors-with-cat-pathology_brc-ouh_split-train-rerun_20240816.csv')
    e = pd.read_csv(e_file_new)
    e.error_category.value_counts()

    for i, row in e.iterrows():
        print('\n---report {}, true {}, pred {}, cat {}, note {}'.format(row.row, row.crc_true, row.crc_pred, row.error_category, row.note))
        #print(row.text)
        submatches = matches_e.loc[matches_e.row.isin([row.row])]
        for j, row2 in submatches.iterrows():
            print('\n', row2.left, '<<', row2.target.upper(), '>>', row2.right, '|', row2.exclusion_reason)
        input("press any key")

esub = e.loc[e.error_category == 'dysplasia_not_cancer']
for i, row in esub.iterrows():
    print('\n---report {}, true {}, pred {}, cat {}'.format(row.row, row.crc_true, row.crc_pred, row.error_category))
    #print(row.text)
    submatches = matches_e.loc[matches_e.row.isin([row.row])]
    for j, row2 in submatches.iterrows():
        print('\n', row2.left, '<<', row2.target.upper(), '>>', row2.right, '|', row2.exclusion_reason)
    input("press any key")


e_file = Path(r'Z:\Andres\project_textmining\textmining\results\results-crc_errors_brc-ouh_split-train-rerun_20240816.csv')
e = pd.read_csv(e_file)
e = e.loc[e.report_text_anon.str.lower().str.contains('am certain')]
e.report_text_anon.item()

e_file = Path(r'Z:\Andres\project_textmining\textmining\results\results-crc_errors_brc-ouh_split-test-rerun_20240816.csv')
e = pd.read_csv(e_file)
e = e.loc[e.report_text_anon.str.lower().str.contains('certain')]
e.report_text_anon.item()


# .... 3.3. Explore performance when further gene test reports excluded (set to crc_nlp = 0) ....
file_labelled = 'set1_crc_labelled_AT-HJ-AT_20231201.xlsx'
truth_path = PROJECT_ROOT / 'labelled_data/processed' / file_labelled
df0 = pd.read_excel(truth_path)


e_file_new = Path(r'Z:\Andres\project_textmining\textmining\results\results-crc_errors-with-cat-pathology_brc-ouh_split-train-rerun_20240816.csv')
e = pd.read_csv(e_file_new)
esub = e.loc[e.error_category == 'crc_implied_by_gene_test']
esub.report_text_anon = esub.report_text_anon.str.replace('\n|\r', ' <n> ')
for i, row in esub.iterrows(): ## 2 reports are falsely negated
    print('-----\n\n', row.report_text_anon)

mask = df0.report_text_anon.isin(esub.report_text_anon)
mask.sum()
df0.loc[mask, 'crc_nlp'] = 0
eval_path_new = PROJECT_ROOT / 'labelled_data' / 'processed' / 'set1_crc_rerun_20240816.csv'

truth_path_new = PROJECT_ROOT / 'labelled_data' / 'processed' / 'set1_crc_relabel2_20240816.csv'
df0.to_csv(truth_path_new, index=False)
assert truth_path_new.exists
assert eval_path_new.exists


evaluate_crc(truth_path_new, eval_path_new, split='train-rerun-relabel2_20240816', older_code=True)
pd.read_csv(r'Z:\Andres\project_textmining\textmining\results\results-crc_sum-print_brc-ouh_split-train-rerun-relabel2_20240816.csv')



#endregion


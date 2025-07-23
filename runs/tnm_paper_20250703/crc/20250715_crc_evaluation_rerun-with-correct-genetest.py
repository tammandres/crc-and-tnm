"""Evaluate CRC detection on updated annotations that better capture gene testing reports

Andres Tamm (AT)
2025-07-07
"""
import pandas as pd
import numpy as np
from pathlib import Path
from textmining.constants import PROJECT_ROOT, RESULTS_DIR
from textmining.evaluate import evaluate_crc
from textmining.reports_20230206_negbugfix import get_crc_reports


# ---- 1. Rerun the algorithm on data where gene test reports were better annotated ----
#region

# .... 1.1. Train data ....
file_orig = 'set1_crc.csv'
file_labelled = 'set1_crc_labelled_AT-HJ-AT-AT_20250703_genetest.csv'

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

eval_path_new = PROJECT_ROOT / 'labelled_data' / 'processed' / 'set1_crc_rerun_20250703.csv'
if eval_path != eval_path_new:
    df.to_csv(eval_path_new, index=False)

assert truth_path.exists
assert eval_path_new.exists

evaluate_crc(truth_path, eval_path_new, split='train-rerun_20250703', older_code=True)


# .... 1.2. Test data (OUH future) ....
file_orig = 'set2_crc.csv'
file_labelled = 'set2_crc_labelled_AT-HJ-AT-AT_20250703_genetest.csv'

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

eval_path_new = PROJECT_ROOT / 'labelled_data' / 'processed' / 'set2_crc_rerun_20250703.csv'
if eval_path != eval_path_new:
    df.to_csv(eval_path_new, index=False)

assert truth_path.exists
assert eval_path_new.exists

evaluate_crc(truth_path, eval_path_new, split='test-rerun_20250703', older_code=True)


# .... 1.3. Print results ....
r_train = pd.read_csv(r'Z:\Andres\project_textmining\textmining\results\results-crc_sum-print_brc-ouh_split-train-rerun_20250703.csv')
r_test = pd.read_csv(r'Z:\Andres\project_textmining\textmining\results\results-crc_sum-print_brc-ouh_split-test-rerun_20250703.csv')

r_train
r_test

#endregion


# ---- 2. Rerun the algorithm on a subset of data that excludes supplementary gene testing reports ----
#region

# ..... 2.1. Training data 

# Get predicted CRC status from previous run of the algorithm
# And updated ground truth
df_pred = pd.read_csv(PROJECT_ROOT / 'labelled_data' / 'processed' / 'set1_crc_rerun_20250703.csv')
df_true = pd.read_csv(PROJECT_ROOT / 'labelled_data/processed' / 'set1_crc_labelled_AT-HJ-AT-AT_20250703_genetest.csv')

# Dbl check that reports in df_pred and df_true are in the same order: they should
# Reports in df_true have \n and \r replaced with <n> to facilitate viewing in Excel
assert (df_true.report_text_anon.str.replace(r' <n> |\r', r'\n', regex=True) == df_pred.report_text_anon).all()

# And filter to gene testing reports only
df_pred_nogenetest = df_pred.loc[df_true.gene_testing == 0].reset_index(drop=True)

# Double check that gene testing only reports are also in the same order as in the no genetest subset
truth_path_nogenetest = PROJECT_ROOT / 'labelled_data/processed' / 'set1_crc_labelled_AT-HJ-AT-AT_20250703_nogenetest.csv'
df_true_nogenetest = pd.read_csv(truth_path_nogenetest)
assert (df_true_nogenetest.report_text_anon.str.replace(r' <n> |\r', r'\n', regex=True) == df_pred_nogenetest.report_text_anon).all()

# Save to disk
eval_path_nogenetest = PROJECT_ROOT / 'labelled_data' / 'processed' / 'set1_crc_rerun_20250703_nogenetest.csv'
df_pred_nogenetest.to_csv(eval_path_nogenetest, index=False)

assert truth_path_nogenetest.exists
assert eval_path_nogenetest.exists

evaluate_crc(truth_path_nogenetest, eval_path_nogenetest, split='train-rerun_20250703_nogenetest', older_code=True)


# .... 2.2. Future test data ....

# Get predicted CRC status from previous run of the algorithm
# And updated ground truth
df_pred = pd.read_csv(PROJECT_ROOT / 'labelled_data' / 'processed' / 'set2_crc_rerun_20250703.csv')
df_true = pd.read_csv(PROJECT_ROOT / 'labelled_data/processed' / 'set2_crc_labelled_AT-HJ-AT-AT_20250703_genetest.csv')

# Dbl check that reports in df_pred and df_true are in the same order: they should
# Reports in df_true have \n and \r replaced with <n> to facilitate viewing in Excel
assert (df_true.report_text_anon.str.replace(r' <n> |\r', r'\n', regex=True).str.replace(r'\s+', '', regex=True) \
        == df_pred.report_text_anon.str.replace(r'\s+', '', regex=True)).all()

# And filter to gene testing reports only
df_pred_nogenetest = df_pred.loc[df_true.gene_testing == 0].reset_index(drop=True)

# Double check that gene testing only reports are also in the same order as in the no genetest subset
truth_path_nogenetest = PROJECT_ROOT / 'labelled_data/processed' / 'set2_crc_labelled_AT-HJ-AT-AT_20250703_nogenetest.csv'
df_true_nogenetest = pd.read_csv(truth_path_nogenetest)
assert (df_true_nogenetest.report_text_anon.str.replace(r' <n> |\r', r'\n', regex=True).str.replace(r'\s+', '', regex=True) \
        == df_pred_nogenetest.report_text_anon.str.replace(r'\s+', '', regex=True)).all()

# Save to disk
eval_path_nogenetest = PROJECT_ROOT / 'labelled_data' / 'processed' / 'set2_crc_rerun_20250703_nogenetest.csv'
df_pred_nogenetest.to_csv(eval_path_nogenetest, index=False)

assert truth_path_nogenetest.exists
assert eval_path_nogenetest.exists

evaluate_crc(truth_path_nogenetest, eval_path_nogenetest, split='test-rerun_20250703_nogenetest', older_code=True)


# .... 2.3. Print results ....
r_train = pd.read_csv(r'Z:\Andres\project_textmining\textmining\results\results-crc_sum-print_brc-ouh_split-train-rerun_20250703_nogenetest.csv')
r_test = pd.read_csv(r'Z:\Andres\project_textmining\textmining\results\results-crc_sum-print_brc-ouh_split-test-rerun_20250703_nogenetest.csv')

r_train
r_test

#endregion


# ---- 3. Update error analysis for test data ----
#region

# Read all test data reports
df_test = pd.read_csv(PROJECT_ROOT / 'labelled_data' / 'processed' / 'set2_crc_labelled_AT-HJ-AT-AT_20250703_genetest.csv')

# Apply the minimal processing that was used by evaluate._prepare_data
# This was done because the reports in xlsx file that were reviewed by HJ had \n and \r replaced with <n> to facilitate viewing
df_test.report_text_anon = df_test.report_text_anon.str.replace(r' <n> |\r', r'\n', regex=True) 

# Read error categories from the previous run of the algorithm
e_old_path = pd.read_csv(RESULTS_DIR / 'results-crc_errors-with-cat-pathology_brc-ouh_split-test-rerun_20240816.csv')
e_old_path.error_category.value_counts()
e_old_path.error_category.value_counts().sum()

e_old_img = pd.read_csv(RESULTS_DIR / 'results-crc_errors-with-cat-imaging_brc-ouh_split-test-rerun_20240816.csv')
e_old_img.error_category.value_counts()
e_old_img.error_category.value_counts().sum()

# Read the errors from the new run of the analysis, where error categories have not yet been assigned
e = pd.read_csv(RESULTS_DIR / 'results-crc_errors_brc-ouh_split-test-rerun_20250703.csv')

# Check that all report texts in e_old can be matched to e
# One text cannot be matched - but this is because it is not an error in the new run (see below)
mask = e_old_path.report_text_anon.isin(e.report_text_anon)
mask.mean()
(~mask).sum()
text = e_old_path.loc[~mask].report_text_anon.iloc[0]
print(repr(text))  # this is the gene testing report that was previously marked as containing no CRC ... but falsely predicted as CRC based on patterns

assert e_old_img.report_text_anon.isin(e.report_text_anon).mean() == 1.0

# Check that all report texts in e_old_path can be matched to df_test, as we want to merge to get gene testing reports
assert e_old_path.report_text_anon.isin(df_test.report_text_anon).mean() == 1.0

# Check that report texts are unique
assert e_old_path.report_text_anon.nunique() == e_old_path.shape[0]
assert e_old_img.report_text_anon.nunique() == e_old_img.shape[0]
assert e.report_text_anon.nunique() == e.shape[0]
assert df_test.report_text_anon.nunique() == df_test.shape[0]


# .... Future pathology reports ....

# Get misclassified pathology reports
e_path = e.loc[e.report_type=='pathology_future']
print(e_old_path.columns, e_path.columns)

# Add error categories from the previous run
e_path.shape
e_path = e_path.merge(e_old_path[['report_text_anon', 'crc_true', 'note', 'crc_pred', 'row', 'error_category']], 
                      how='left', on=['report_text_anon', 'crc_true', 'note', 'crc_pred'])
e_path.shape

# Add the gene testing indicator from the reports table
e_path = e_path.merge(df_test[['report_type', 'report_text_anon', 'gene_testing']], how='left', on=['report_type', 'report_text_anon'])
e_path.shape

e_path.gene_testing.sum()
e_path.loc[(e_path.error_category.isna()) & (e_path.gene_testing == 1), 'error_category'] = 'crc_implied_by_gene_test'
assert e_path.error_category.isna().sum() == 0
e_path.error_category.value_counts()

# Dbl check where site was missed due to redaction: site was partially redacted
# Place this under the crc_site_missed category
e_path.loc[e_path.error_category == 'crc_missed_due_to_redaction'].report_text_anon.iloc[0]  # Site is partially redacted, hence missed
e_path.loc[e_path.error_category == 'crc_missed_due_to_redaction', 'error_category'] = 'crc_site_missed'

# Dbl check where summary does not state cancer: in that report the text says there is tumour, but the summary says dysplasia
# so the text probably did not mean malignant tumour.
e_path.loc[e_path.error_category == 'summary_does_not_state_cancer'].report_text_anon.iloc[0] # S
e_path.loc[e_path.error_category == 'summary_does_not_state_cancer'].note.iloc[0]

e_path.shape
e_path.report_text_anon.nunique()

# Double check that performance metrics computed from crc_true and crc_pred are the same 
# as computed in the crc_evaluation_rerun-with-corrected-genetest_20250703.py script
df_true = pd.read_csv(PROJECT_ROOT / 'labelled_data' / 'processed' / 'set2_crc_labelled_AT-HJ-AT-AT_20250703_genetest.csv')
df_true = df_true[['subject_id', 'report_date', 'report_text_anon', 'report_type', 'crc_nlp']].rename(columns={'crc_nlp': 'crc_true'})
df_true.crc_true.value_counts()

df_pred = pd.read_csv(PROJECT_ROOT / 'labelled_data' / 'processed' / 'set2_crc_rerun_20250703.csv')
df_pred = df_pred[['subject_id', 'report_date', 'report_text_anon', 'report_type', 'crc_nlp']].rename(columns={'crc_nlp': 'crc_pred'})
df_pred.crc_pred.value_counts()

df_true['crc_pred'] = df_pred.crc_pred

df_true_path = df_true.loc[df_true.report_type=='pathology_future']
mask = df_true_path.crc_pred != df_true_path.crc_true
mask.sum() # 43 errors in total

df_true_path.loc[df_true_path.crc_pred == 1].crc_true.mean() # 93% PPV
df_true_path.loc[df_true_path.crc_pred == 1].crc_true.sum() / df_true_path.crc_true.sum()  #72.1% sens
(1 - df_true_path.loc[df_true_path.crc_pred == 0].crc_true).mean() # 64% NPV
(1 - df_true_path.loc[df_true_path.crc_pred == 0].crc_true).sum() / (1 - df_true_path.crc_true).sum() # 90.1% spec

## Save to disk
e_path.to_csv(RESULTS_DIR / 'results-crc_errors-with-cat-pathology_brc-ouh_split-test-rerun_20250703.csv')


# .... Future imaging reports: same error cats, no modification ....

# Read misclassified imaging reports
e_img = e.loc[e.report_type=='imaging_future']

# Add error categories from the previous run
e_img.shape
e_img = e_img.merge(e_old_img[['report_text_anon', 'crc_true', 'note', 'crc_pred', 'row', 'error_category']], 
                    how='left', on=['report_text_anon', 'crc_true', 'note', 'crc_pred'])

# These categories are the same as before, as labels of future imaging reports were not modified in any way
e_img.error_category.value_counts()
e_img.error_category.value_counts().sum()
e_img.error_category.isna().sum() == 0

# Save to disk
e_img.to_csv(RESULTS_DIR / 'results-crc_errors-with-cat-imaging_brc-ouh_split-test-rerun_20250703.csv')

# Double check that performance metrics computed from crc_true and crc_pred are the same 
# as computed in the crc_evaluation_rerun-with-corrected-genetest_20250703.py script
df_true_img = df_true.loc[df_true.report_type=='imaging_future']
mask = df_true_img.crc_pred != df_true_img.crc_true
mask.sum() # 29 errors in total

df_true_img.loc[df_true_img.crc_pred == 1].crc_true.mean() # 78% PPV
df_true_img.loc[df_true_img.crc_pred == 1].crc_true.sum() / df_true_img.crc_true.sum()  #91.8% sens
(1 - df_true_img.loc[df_true_img.crc_pred == 0].crc_true).mean() # 93% NPV
(1 - df_true_img.loc[df_true_img.crc_pred == 0].crc_true).sum() / (1 - df_true_img.crc_true).sum() # 80.9% spec

#endregion


# ---- 4. Update error analysis for training data ----
#region

# Read all training data reports
df_test = pd.read_csv(PROJECT_ROOT / 'labelled_data' / 'processed' / 'set1_crc_labelled_AT-HJ-AT-AT_20250703_genetest.csv')

# Apply the minimal processing that was used by evaluate._prepare_data
# This was done because the reports in xlsx file that were reviewed by HJ had \n and \r replaced with <n> to facilitate viewing
df_test.report_text_anon = df_test.report_text_anon.str.replace(r' <n> |\r', r'\n', regex=True) 

# Read error categories from the previous run of the algorithm
e_old_path = pd.read_csv(RESULTS_DIR / 'results-crc_errors-with-cat-pathology_brc-ouh_split-train-rerun_20240816.csv')
e_old_path.error_category.value_counts()
e_old_path.error_category.value_counts().sum()

e_old_img = pd.read_csv(RESULTS_DIR / 'results-crc_errors-with-cat-imaging_brc-ouh_split-train-rerun_20240816.csv')
e_old_img.error_category.value_counts()
e_old_img.error_category.value_counts().sum()

# Read the errors from the new run of the analysis, where error categories have not yet been assigned
e = pd.read_csv(RESULTS_DIR / 'results-crc_errors_brc-ouh_split-train-rerun_20250703.csv')

# Check that all report texts in e_old can be matched to e
# One cannot be matched - but that is ok because it is not an error in the new rerun (see below)
assert e_old_path.report_text_anon.isin(e.report_text_anon).mean() == 1.0

mask = e_old_img.report_text_anon.isin(e.report_text_anon)
mask.sum()

(~mask).sum()
text = e_old_img.loc[~mask].report_text_anon.iloc[0]
e_old_img.loc[~mask].note.iloc[0]
print(repr(text))  # this is the report that describes mets but was accidentally marked as crc_nlp = 1

# Check that all report texts in e_old can be matched to df_test, as we want to get gene_testing indicator
assert e_old_path.report_text_anon.isin(df_test.report_text_anon).mean() == 1.0

# Check that report texts are unique
assert e_old_path.report_text_anon.nunique() == e_old_path.shape[0]
assert e_old_img.report_text_anon.nunique() == e_old_img.shape[0]
assert e.report_text_anon.nunique() == e.shape[0]
assert df_test.report_text_anon.nunique() == df_test.shape[0]


# .... Training data pathology reports ....

# Get misclassified pathology reports
e_path = e.loc[e.report_type=='pathology']
print(e_old_path.columns, e_path.columns)

# Add error categories from the previous run
e_path.shape
e_path = e_path.merge(e_old_path[['report_text_anon', 'crc_true', 'note', 'crc_pred', 'row', 'error_category']], 
                      how='left', on=['report_text_anon', 'crc_true', 'note', 'crc_pred'])
e_path.shape

# Add the gene testing indicator from the reports table
e_path = e_path.merge(df_test[['report_type', 'report_text_anon', 'gene_testing']], how='left', on=['report_type', 'report_text_anon'])
e_path.shape

e_path.gene_testing.sum()
e_path.loc[(e_path.error_category.isna()) & (e_path.gene_testing == 1), 'error_category'] = 'crc_implied_by_gene_test'
assert e_path.error_category.isna().sum() == 0
e_path.error_category.value_counts()
e_path.error_category.value_counts().sum()

# Double check that performance metrics computed from crc_true and crc_pred are the same 
# as computed in the crc_evaluation_rerun-with-corrected-genetest_20250703.py script
# Note that 1 pathology report has crc_nlp nan, because it is not possible to determine due to redaction
df_true = pd.read_csv(PROJECT_ROOT / 'labelled_data' / 'processed' / 'set1_crc_labelled_AT-HJ-AT-AT_20250703_genetest.csv')
df_true = df_true[['subject_id', 'report_date', 'report_text_anon', 'report_type', 'crc_nlp']].rename(columns={'crc_nlp': 'crc_true'})
df_true.crc_true.value_counts()

df_pred = pd.read_csv(PROJECT_ROOT / 'labelled_data' / 'processed' / 'set1_crc_rerun_20250703.csv')
df_pred = df_pred[['subject_id', 'report_date', 'report_text_anon', 'report_type', 'crc_nlp']].rename(columns={'crc_nlp': 'crc_pred'})
df_pred.crc_pred.value_counts()

df_true['crc_pred'] = df_pred.crc_pred

df_true_path = df_true.loc[df_true.report_type=='pathology']
df_true_path = df_true_path.loc[~df_true_path.crc_true.isna()]
mask = df_true_path.crc_pred != df_true_path.crc_true
mask.sum() # 30 errors in total

df_true_path.loc[df_true_path.crc_pred == 1].crc_true.mean() # 94% PPV
df_true_path.loc[df_true_path.crc_pred == 1].crc_true.sum() / df_true_path.crc_true.sum()  #79.6% sens
(1 - df_true_path.loc[df_true_path.crc_pred == 0].crc_true).mean() # 75.8% NPV
(1 - df_true_path.loc[df_true_path.crc_pred == 0].crc_true).sum() / (1 - df_true_path.crc_true).sum() # 92.6% spec

# Save to disk
e_path.to_csv(RESULTS_DIR / 'results-crc_errors-with-cat-pathology_brc-ouh_split-train-rerun_20250703.csv')


# .... Training data imaging reports: same error cats, no modification ....

# Read misclassified imaging reports
e_img = e.loc[e.report_type=='imaging']

# Add error categories from the previous run
e_img.shape
e_img = e_img.merge(e_old_img[['report_text_anon', 'crc_true', 'note', 'crc_pred', 'row', 'error_category']], 
                    how='left', on=['report_text_anon', 'crc_true', 'note', 'crc_pred'])
e_old_img.error_category.value_counts().sum() #16
e_old_img.loc[e_old_img.error_category=='mets_not_primary'].report_text_anon.iloc[0]  # This is an img report whose label was corrected to be 0 as per HJ comment

# These categories are the same as before, as labels of future imaging reports were not modified in any way
e_img.error_category.value_counts()
e_img.error_category.value_counts().sum() # 15
e_img.error_category.isna().sum() == 0

# Save to disk
e_img.to_csv(RESULTS_DIR / 'results-crc_errors-with-cat-imaging_brc-ouh_split-train-rerun_20250703.csv')

# Double check that performance metrics computed from crc_true and crc_pred are the same 
# as computed in the crc_evaluation_rerun-with-corrected-genetest_20250703.py script
df_true_img = df_true.loc[df_true.report_type=='imaging']
mask = df_true_img.crc_pred != df_true_img.crc_true
mask.sum() # 15 errors in total

df_true_img.loc[df_true_img.crc_pred == 1].crc_true.mean() # 89.9 ppv
df_true_img.loc[df_true_img.crc_pred == 1].crc_true.sum() / df_true_img.crc_true.sum() # 94.7 sens
(1 - df_true_img.loc[df_true_img.crc_pred == 0].crc_true).mean() # 95.0 npv
(1 - df_true_img.loc[df_true_img.crc_pred == 0].crc_true).sum() / (1 - df_true_img.crc_true).sum() # 90.6 spec

#endregion


# ---- 5. view results ----
#region

a = pd.read_csv(RESULTS_DIR / 'results-crc_errors-with-cat-pathology_brc-ouh_split-train-rerun_20250703.csv')
b = pd.read_csv(RESULTS_DIR / 'results-crc_errors-with-cat-imaging_brc-ouh_split-train-rerun_20250703.csv')
c = pd.read_csv(RESULTS_DIR / 'results-crc_errors-with-cat-pathology_brc-ouh_split-test-rerun_20250703.csv')
d = pd.read_csv(RESULTS_DIR / 'results-crc_errors-with-cat-imaging_brc-ouh_split-test-rerun_20250703.csv')

for df, label in zip([a, b, c, d], ['path-train', 'img-train', 'path-test', 'img-test']):
    print('\n---', label)
    print(df.error_category.value_counts())
    print('Total', df.shape[0])
#endregion
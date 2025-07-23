"""
This script reviews the data that was annotated for extracting CRC status from imaging and pathology reports.

The annotations were done in a sequence:
(1) the initial annotations were created by AT (set1_crc_labelled.csv, set2_crc_labelled.csv),
(2) these were then reviewed by HJ (set1_crc_labelled.xlsx, set2_crc_labelled.xlsx),
(3) and changes suggested by HJ were made by AT (set1_crc_labelled_AT-HJ-AT_20231201, set2_crc_labelled_AT-HJ-AT_20231201.xlsx).
This script compares these annotated files, to trace how they changed over time in a single script.

In addition, HJ recommended that supplementary gene testing reports nearly always indicate CRC,
as these tests are only done when CRC is found.
When I initially updated annotations with HJ feedback, I tried to find all gene testing reports
using the regex '^\W*supplement.*mmr' (mmr: mismatch repair), but this did not cover all gene testing reports. 
This script uses a much more comprehensive list of genes that were present in reports that HJ
suggested as gene testing reports:
r'supplement.*(mismatch repair|mmr|TP53|PTEN|PIK3CA|PDGFRA|KRAS|NRAS|EGFR|BRAF|MSH2|MSH6|MLH1|PMS2)'

The annotations are thus updated, so that nearly all reports that match this regex are marked 
as gene testing reports (and implying that they indicate CRC), with a few exceptions
(e.g. a report that explicitly discuses a non-CRC tumour along with gene tests)

Andres Tamm (AT)
2025-07-07
"""
import pandas as pd
import numpy as np
from pathlib import Path
from textmining.constants import PROJECT_ROOT
from textmining.utils import extract
import re
from textmining.reports_20230206_negbugfix import get_crc_reports


data_path = PROJECT_ROOT / 'labelled_data'


inp = False

# ----- FUTURE TEST SET LABELS -----
#region

df0 = pd.read_csv(data_path / 'set2_crc.csv')

# df1 is the initial labelling round by me
df1 = pd.read_csv(data_path / 'set2_crc_labelled.csv')
test = (df0.crc_nlp == df1.crc_nlp).mean()
df1.note.unique()

# df2 is for sharing the data with HJ - same as df1
df2 = pd.read_excel(data_path / 'set2_crc_labelled.xlsx')
print(df2.note.value_counts())
assert (df1.crc_nlp == df2.crc_nlp).all()
assert (df1.note == df2.note).all()

# df3 is the data reviewed by HJ
# the crc_nlp field was not edited, but H added notes, which I later used to edit crc_nlp
df3 = pd.read_excel(r"Z:\for Helen\textmining_review_2023-11-22\set2_crc_labelled.xlsx")
(df3.crc_nlp == df2.crc_nlp).mean()
(df3.note == df2.note).mean()
mask = df3.note != df2.note
for i in range(sum(mask)):
    print('\n---', df2.loc[mask, 'note'].iloc[i], '|', df3.loc[mask, 'note'].iloc[i])

# In df4, labels (crc_nlp column) of 24 reports were edited based on HJ's comments.
# One additional comment was made for a report where CRC was clearly present but not commented by H.
# Reports describing gene testing were marked for cancer as indicated by HJ.
# 14 edits: 0 -> 1, gene testing supplementary report
#  4 edits: 1 -> 0, CRC not seen on the scan/sample
#  3 edits: 1 -> 0, metastasis not primary
#  1 edit: 1 -> 0, benign
#  1 edit: 1 -> 0, can't be sure if crc
#  1 edit: 0 -> 1: crc is present
df4 = pd.read_excel(data_path / 'processed' / 'set2_crc_labelled_AT-HJ-AT_20231201.xlsx')
(df4.crc_nlp == df3.crc_nlp).mean()
(df4.crc_nlp != df3.crc_nlp).sum()
(df4.note == df3.note).mean()
mask = df4.note != df3.note
for i in range(sum(mask)):
    print('\n---', df2.loc[mask, 'note'].iloc[0], '|', df3.loc[mask, 'note'].iloc[i], '|', df4.loc[mask, 'note'].iloc[i])

mask = df3.crc_nlp != df4.crc_nlp
df3sub = df3.loc[mask]
df4sub = df4.loc[mask]
print(mask.sum())
for i in range(df3sub.shape[0]):
    row3 = df3sub.iloc[i]
    row4 = df4sub.iloc[i]
    idx = df3sub.index[i]
    print('\n------', idx)
    print(row3.crc_nlp, row4.crc_nlp)
    print(row3.note, '|', row4.note)
    print(repr(row3.report_text_anon))
    if inp:
        key = input('...')


# Does the future test set have more reports describing gene testing that were previously missed (marked as crc_nlp = 0)?
# Yes - most look valid, expect two
#  [comment about individual reports removed - AT 20250721]
# A total of 20 reports.
pat = r'supplement.*(mismatch repair|mmr|TP53|PTEN|PIK3CA|PDGFRA|KRAS|NRAS|EGFR|BRAF|MSH2|MSH6|MLH1|PMS2)'
df4['row'] = np.arange(df4.shape[0])
row_no_crc = df4.loc[df4.crc_nlp == 0].row

all_matches = extract(df4, 'report_text_anon', pat, flags=re.I|re.DOTALL)
matches = all_matches.loc[all_matches.row.isin(row_no_crc)]

matches.row.nunique()
for i, row in matches.iterrows():
    print('\n-----', row.row, '----\n', repr(row.target))
    if inp:
        key = input('...')

matches_incl = matches.loc[~matches.row.isin([16, 258])]
for i, row in matches_incl.iterrows():
    print('\n-----', row.row, '----\n', repr(row.target))
    if inp:
        key = input('...')

print(matches_incl.row.nunique())

all_matches_incl = all_matches.loc[~all_matches.row.isin([16, 258])]
for i, row in all_matches_incl.iterrows():
    print('\n-----', row.row, '----\n', repr(row.target))
    if inp:
        key = input('...')

# [comment about individual reports removed - AT 20250721]
all_matches_incl = all_matches.loc[~all_matches.row.isin([16, 258, 204, 271, 361])]
for i, row in all_matches_incl.iterrows():
    print('\n-----', row.row, '----\n', repr(row.target))
    if inp:
        key = input('...')

all_matches_incl.row.nunique()  # 44
matches_incl.row.nunique()  # 20
df4_corrected = df4.copy()
mask = df4_corrected.row.isin(matches_incl.row)
assert df4_corrected.loc[mask, 'crc_nlp'].unique() == [0]
assert (df4_corrected.loc[mask, 'report_type'] == 'pathology_future').all()
df4_corrected.loc[mask, 'crc_nlp'] = 1

df4_corrected['gene_testing'] = 0
df4_corrected.loc[df4_corrected.row.isin(all_matches_incl.row), 'gene_testing'] = 1
assert (df4_corrected.loc[df4_corrected.gene_testing == 1, 'report_type'] == 'pathology_future').all()

df4_corrected.gene_testing.sum()
df4_corrected.to_csv(data_path / 'processed' / 'set2_crc_labelled_AT-HJ-AT-AT_20250703_genetest.csv')

df4_corrected_no_gene_test = df4_corrected.loc[df4_corrected.gene_testing==0]
df4_corrected_no_gene_test.to_csv(data_path / 'processed' / 'set2_crc_labelled_AT-HJ-AT-AT_20250703_nogenetest.csv')

# note that in previous run, PPV was 92.0. When run on corrected gene test data, it is 93.0. 
# this is because there is one gene testing report that was previously marked as a non-CRC report.
# The CRC algorithm marks it as a CRC report, but for the wrong reason:
# the word 'tumour' and 'colorectal' occur close by, but do not refer to colorectal cancer.
__, matches_crc = get_crc_reports(df4_corrected, 'report_text_anon', add_subj_to_matches=True, subjcol='subject_id', negation_bugfix=True)
matches_incl = matches_crc.loc[matches_crc.exclusion_indicator==0]
df4_path = df4.loc[df4.report_type=='pathology_future']
df4_corrected_path = df4_corrected.loc[df4_corrected.report_type=='pathology_future']
mask = df4_corrected_path.row.isin(matches_incl.row)
a = df4_corrected_path.loc[mask]
b = df4_path.loc[mask]
print(a.crc_nlp.mean(), b.crc_nlp.mean())
print(a.crc_nlp.sum(), b.crc_nlp.sum())
delta = mask & (a.crc_nlp != b.crc_nlp) 
a.loc[delta].report_text_anon.iloc[0]
a.loc[delta].note.iloc[0]

# Note that some gene testing reports can be marked as containing CRC purely based on other patterns, e.g. 
# "tumour cells from the adenocarcinoma in the [anatomical location]"
df4sub = df4_corrected.loc[df4_corrected.gene_testing == 1]
df4sub.crc_nlp.sum()
for i, row in df4sub.iterrows():
    print(i, '\n----', repr(row.report_text_anon))

t, m = get_crc_reports(df4sub, 'report_text_anon')
m_incl =  m.loc[m.exclusion_indicator==0]
for i, row in m_incl.iterrows():
    print('\n----', row.left, row.target, row.right)

# Are there any other supplementary reports that describe gene testing?
# Nope - only 5 supplementary pathology reports are left, none describe gene testing
t = df4_corrected.loc[~df4_corrected.row.isin(all_matches.row)]
t = t.loc[t.report_type=='pathology_future']
t = t.loc[t.report_text_anon.str.lower().str.contains('^.{,100}supplement')]
t.shape
for i, row in t.iterrows():
    print('\n----', i)
    print(repr(row.report_text_anon))
    if inp:
        key = input('...')

df4_corrected.gene_testing.sum()  # 49 reports
#endregion


# ----- TRAINING SUBSET LABELS -----
#region
df0 = pd.read_csv(data_path / 'set1_crc.csv')

# df1 is the initial labelling round by me
df1 = pd.read_csv(data_path / 'set1_crc_labelled.csv')
test = (df0.crc_nlp == df1.crc_nlp).mean()
df1.note.unique()

# df2 is for sharing the data with HJ - same as df1
df2 = pd.read_excel(data_path / 'set1_crc_labelled.xlsx')
print(df2.note.value_counts())
assert (df1.crc_nlp == df2.crc_nlp).all()
assert (df1.note == df2.note).all()

# df3 is the data reviewed by HJ
# the crc_nlp field was not edited, but H added notes, which I later used to edit crc_nlp
df3 = pd.read_excel(r"Z:\for Helen\textmining_review_2023-11-22\set1_crc_labelled.xlsx")
(df3.crc_nlp == df2.crc_nlp).mean()
(df3.note == df2.note).mean()
mask = df3.note != df2.note
for i in range(sum(mask)):
    print('\n---', df2.loc[mask, 'note'].iloc[i], '|', df3.loc[mask, 'note'].iloc[i])

# In df4, labels (crc_nlp column) of 20 reports were edited based on HJ's comments.
# Reports describing gene testing were marked for cancer as indicated by HJ.
#  5 edits: 0 -> 1, gene testing supplementary report
#  4 edits: 1 -> 0, not definite CRC
#  6 edits: 1 -> 0, metastasis or recurrence, not primary CRC
#  2 edits: 1 -> 0, pathology other than CRC
#  1 edit: 1 -> nan, cannot determine due to redaction
#  1 edit: 0 -> 1, debatable but strictly CRC
#  1 edit: 0 -> 1: HJ marks it as 0 due to no primary, though I marked it as 1, probably by mistake (this is corrected below) 
df4 = pd.read_excel(data_path / 'processed' / 'set1_crc_labelled_AT-HJ-AT_20231201.xlsx')
(df4.crc_nlp == df3.crc_nlp).mean()
(df4.crc_nlp != df3.crc_nlp).sum()
(df4.note == df3.note).mean()
mask = df4.note != df3.note
for i in range(sum(mask)):
    print('\n---', df2.loc[mask, 'note'].iloc[0], '|', df3.loc[mask, 'note'].iloc[i], '|', df4.loc[mask, 'note'].iloc[i])

mask = df3.crc_nlp != df4.crc_nlp
print(mask.sum())
df3sub = df3.loc[mask]
df4sub = df4.loc[mask]
print(mask.sum())
for i in range(df3sub.shape[0]):
    row3 = df3sub.iloc[i]
    row4 = df4sub.iloc[i]
    idx = df3sub.index[i]
    print('\n------', idx)
    print(row3.crc_nlp, row4.crc_nlp)
    print(row3.note, '|', row4.note)
    print(repr(row3.report_text_anon))

# Does the training test set have more reports describing gene testing that were previously missed (marked as crc_nlp = 0)?
# Yes, 19. 
pat = r'supplement.*(mismatch repair|mmr|TP53|PTEN|PIK3CA|PDGFRA|KRAS|NRAS|EGFR|BRAF|MSH2|MSH6|MLH1|PMS2)'
df4['row'] = np.arange(df4.shape[0])
row_no_crc = df4.loc[df4.crc_nlp == 0].row
matches = extract(df4, 'report_text_anon', pat, flags=re.I)
all_matches = matches.copy()
matches = matches.loc[matches.row.isin(row_no_crc)]

matches.row.nunique() #19
for i, row in matches.iterrows():
    print('\n-----', row.row, '----\n', repr(row.target))
    if inp:
        key = input('...')

all_matches.row.nunique() #32
for i, row in all_matches.iterrows():
    print('\n-----', row.row, '----\n', repr(row.target))
    if inp:
        key = input('...')

#  [comment about individual reports removed - AT 20250721]
all_matches_incl = all_matches.loc[~all_matches.row.isin([83, 236, 293])]
all_matches_incl.row.nunique() # 29
for i, row in all_matches_incl.iterrows():
    print('\n-----', row.row, '----\n', repr(row.target))
    if inp:
        key = input('...')

df4_corrected = df4.copy()

# First, correct report 121 to have no CRC as per HJs comment (AT must have made a mistake on this)
df4_corrected.loc[121].note
df4_corrected.loc[121, 'crc_nlp'] = 0

# Then mark additional gene testing reports as containing CRC
mask = df4_corrected.row.isin(matches.row)
assert df4_corrected.loc[mask, 'crc_nlp'].unique() == [0]
df4_corrected.loc[mask, 'crc_nlp'] = 1

# Also create an indicator gene testing reports
df4_corrected['gene_testing'] = 0
df4_corrected.loc[df4_corrected.row.isin(all_matches_incl.row), 'gene_testing'] = 1
df4_corrected.gene_testing.sum() #29

# Save to disk
df4_corrected.to_csv(data_path / 'processed' / 'set1_crc_labelled_AT-HJ-AT-AT_20250703_genetest.csv')
df4_corrected_no_gene_test = df4_corrected.loc[df4_corrected.gene_testing==0]
df4_corrected_no_gene_test.to_csv(data_path / 'processed' / 'set1_crc_labelled_AT-HJ-AT-AT_20250703_nogenetest.csv')

# Are there any other supplementary reports that describe gene testing? (relevant only for reports marked as non-crc)
# Nope - only 7 supplementary pathology reports are left, none are relevant
#  [comment about individual reports removed - AT 20250721]
t = df4_corrected.loc[~df4_corrected.row.isin(matches.row)]
t = t.loc[t.crc_nlp == 0]
t = t.loc[t.report_type=='pathology']
t = t.loc[t.report_text_anon.str.lower().str.contains('^.{,100}supplement')]
t.shape
for i, row in t.iterrows():
    print('\n----', i)
    print(repr(row.report_text_anon))
    if inp:
        key = input('...')


df4_corrected.gene_testing.sum()  # 32 reports

#endregion
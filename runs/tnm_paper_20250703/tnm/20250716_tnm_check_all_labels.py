"""Review edits to TNM staging labels in a single script

Andres Tamm (AT)
2025-07-23
"""

import pandas as pd
from pathlib import Path
from textmining.constants import PROJECT_ROOT
import os


data_path = PROJECT_ROOT / 'labelled_data' / 'processed' 


# ---- Training data labels ----
#region

files = os.listdir(data_path)
files = [f for f in files if f.startswith('set1_tnm_labelled')]

# Read data
df0 = pd.read_csv(data_path / 'set1_tnm_labelled_ND_20230926.csv')
df1 = pd.read_csv(data_path / 'set1_tnm_labelled_ND_AT_20231218.csv')
df2 = pd.read_csv(data_path / 'set1_tnm_labelled_ND_AT_AT_20240214.csv')
df3 = pd.read_csv(data_path / 'set1_tnm_labelled_ND_AT_AT_HJ_20240221.csv')
df4 = pd.read_csv(data_path / 'set1_tnm_labelled_ND_AT_AT_HJ_AT_20240306.csv')


def review_edits(df0, df1, cols):
    for c in cols:
        mask = df0[c].fillna(' ') != df1[c].fillna(' ')
        if mask.sum() > 0:
            print('\n----', c, mask.sum(), 'edits')
            s0 = df0.loc[mask]
            s1 = df1.loc[mask]
            for i in range(s0.shape[0]):
                print(c,
                    "| value_0:", s0[c].iloc[i], 
                    "| value_1:", s1[c].iloc[i], 
                    '| note_0: ', s0.note.iloc[i],
                    '| note_1: ', s1.note.iloc[i])
                

# The original labels by ND are in df0
cols = ['T_pre', 'T', 'N', 'M', 'V', 'R', 'L', 'Pn', 'SM', 'H',
        'G', 'T_pre_min', 'T_min', 'N_min', 'M_min', 'V_min', 'R_min', 'L_min',
        'Pn_min', 'SM_min', 'H_min', 'G_min']
print(df0[cols].head())

# df1 contains labels edited by AT. All of the edits have a comment starting with "AT"
# This was semi-manually done using tnm_check_labels_set1_part1.py
# 17 reports were edited in total:
#   T: null -> 0 due to implicit staging (10 edits)
#   T: correct "is" staging in one report (1)
#   N: correct N1 to N0 (1)
#   Pn: include Pn (1)
#   include historical staging (1)
#   prefer summary staging as the annotated staging (3)
mask = df0[cols].fillna(' ') != df1[cols].fillna(' ')
mask.any(axis=1).sum()
s = df1.loc[mask.any(axis=1)].note
s = s.str.extract('(AT.*)')
s.value_counts()
s.value_counts().sum()

review_edits(df0, df1, cols)

# df2 contains further edits, e.g. ensuring historical and implicit staging is consistently labelled compared to set2
# This was semi-manually done using tnm_check_labels_set1_part2.py
# 100 reports were edited in total
# Some reports were marked for HJ for review
#  T_pre: 85 edits: mostly adding a prefix, e.g. 'p' because it is a pathology report or 'mr' for MRI.
#    This ensures that we add a prefix when it can be inferred from the report, not only when it is explicitly provided.
#  T: 72 edits: mostly assigning an implicit stage to be 'X' or '0'; in some cases removing a stage because it was historical
#    51 edits: null -> x
#    9 edits: null -> 0
#    4 edits: null -> 4
#    2 edits: null -> 2
#    2 edits: 3 -> null
#    1 edit: 0 -> null, clinician comment says cannot assign any stage
#    1 edit: 1 -> 0
#    1 edit: 4 -> 4b, prefer summary
#    1 edit: 4 -> X, 4 was historic staging
#  N: 11 edits
#    4 edits: null -> X (implicit)
#    4 edits: null -> 0 (implicit
#    2 edits: 0 -> null, at least one edit historic
#    1 edit: 1c -> 2, prefer summary
#  M: 17 edits
#    13 edits: null -> x (implicit)
#    2 edits: null -> 0 (implicit)
#    2 edits: 0 -> null (historic in at least one edit)
#  SM: 1 edit
#    2 -> null, it may have been a mistake originally
#  G: 13 edits
#    in all cases, null -> value
#  T_min: prefer summary stage (1)
#  N_min: prefer summary stage (1)
mask = df1[cols].fillna(' ') != df2[cols].fillna(' ')
mask.any(axis=1).sum()

review_edits(df1, df2, cols)

mask = (df1['T'] != df2['T']) & (df1['T'] == ' ')
df2.loc[mask, 'T'].value_counts()
mask.sum() #66

mask = (df1['T'] != df2['T']) & (df2['T'] == ' ')
df1.loc[mask, 'T'].value_counts()
mask.sum() #3

mask = (df1['T'] != df2['T']) & (df2['T'] != ' ') & (df1['T'] != ' ')
mask.sum() #3
df1.loc[mask, 'T']
df2.loc[mask, 'T']


# df3 contains no edits over df2 to the columns or to the note field
# However, the report idx's were likely discussed
mask = df2[cols].fillna(' ') != df3[cols].fillna(' ')
mask.any(axis=1).sum()

mask = df2.note != df3.note
mask.sum()


# df4 contains a few more edits over df3
#  T_pre: 85 edits, mainly about removing an empty space in annotation
#   84 edits: empty space removed, e.g. ' p' -> 'p'
#   1 edit: null -> 'c'
# T: 7 edits
#   4 edits: null -> 'x' (implicit)
#   2 edits: null -> 0 (implicit)
#   1 edit: X -> 4, can infer as per HJ comment
df3.loc[mask, 'T_pre'].iloc[0]
df4.loc[mask, 'T_pre'].iloc[0]
mask = df3[cols].fillna(' ') != df4[cols].fillna(' ')
mask.any(axis=1).sum()

mask = df3.note != df4.note
mask.sum()

review_edits(df3, df4, cols)

mask = df3.T_pre != df4.T_pre
df3.loc[mask, 'T_pre'].iloc[0]
df4.loc[mask, 'T_pre'].iloc[0]
mask = df3.T_pre.str.replace(' ', '') != df4.T_pre.str.replace(' ', '')
mask.sum()
df3.loc[mask, 'T_pre'].iloc[0]
df4.loc[mask, 'T_pre'].iloc[0]

mask = df3.note.str.replace(' ', '') != df4.note.str.replace(' ', '')
s3, s4 = df3.loc[mask, 'note'], df4.loc[mask, 'note']
for i in range(mask.sum()):
    print('\n', i)
    print('note 0', s3.iloc[i])
    print('note 1', s4.iloc[i])

#endregion


# ---- Test data labels ----
#region


# 
files = os.listdir(data_path)
files = [f for f in files if f.startswith('set2_tnm_labelled')]

# Read data
df0 = pd.read_csv(data_path / 'set2_tnm_labelled_HJ_20240203.csv')
df1 = pd.read_csv(data_path / 'set2_tnm_labelled_HJ_AT_20240214.csv')
df2 = pd.read_csv(data_path / 'set2_tnm_labelled_HJ_AT_AT_20241004.csv')

# df0: data labelled by AT, reviewed by HJ in the report viewer

# df0 vs df1: 18 reports edited, based on HJ comments
# T_pre: 11 edits: assigning a prefix based on report type, e.g. p or mr or ct
# T: 6 edits
#  1 edit: null -> 1, inferred
#  1 edit: null -> 4, inferred
#  4 edits: 1 -> 0, 3 -> 0, 4-> 0, 3 -> x the extracted staging was for historic
# N: 1 edit
#  1 -> 0
# M: 1 edit
#  0 -> 1, inferred
# T_min
#  null -> 0, inferred
mask = df0[cols].fillna(' ') != df1[cols].fillna(' ')
mask.any(axis=1).sum()
review_edits(df0, df1, cols)

mask = df0.note != df1.note
note0 = df0.loc[mask, 'note']
note1 = df1.loc[mask, 'note']
for (n0, n1) in zip(note0, note1):
    print('\n---')
    print(n0)
    print(n1)

# df1 vs df2: 3 reports edited
# 3 edits in T stage, 0 -> null, because NOT colorectal sample
mask = df1[cols].fillna(' ') != df2[cols].fillna(' ')
mask.any(axis=1).sum()
review_edits(df1, df2, cols)

#endregion
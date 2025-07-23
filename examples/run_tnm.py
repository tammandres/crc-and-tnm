import numpy as np
import pandas as pd
from textmining.reports import get_crc_reports
from textmining.tnm.clean import add_tumour_tnm
from textmining.tnm.tnm import get_tnm_phrase, get_tnm_values
from textmining.reports import get_crc_reports_par
from textmining.tnm.tnm import get_tnm_phrase_par


# ---- Prepare data ----
#region
# Create dummy reports
reports = ['Metastatic tumour from colorectal primary, T3 N0',
           'T1 N0 MX (colorectal cancer)',
           'pT3/2/1 N0 Mx. Malignant neoplasm ascending colon',
           'pT2a/b N0 Mx (sigmoid tumour)',
           'T4a & b N0 M1 invasive carcinoma, descending colon',
           'T1-weighted image, ... rectal tumour staged as ymrT2',
           'Colorectal tumour. Stage: T4b / T4a / T3 / T2 / T1',
           'Sigmoid adenocarcinoma, ... Summary: pT1 (sigmoid, txt txt txt txt), N3b M0',
           'Colorectal tumour in situ, Tis N0 M0',
           'Clinical information: T1 N1 (sigmoid tumour)'
           ]
df = pd.DataFrame(reports, columns=['report_text_anon'])
df['subject'] = '01'

pd.set_option('display.max_colwidth', 500, 'display.max_rows', 1000, 'display.min_rows', 1000)
print(df)
#endregion

# ---- Step 1. Identify reports that describe current colorectal cancer ----
#region
"""
Main arguments
* `df`: Pandas DataFrame that contains reports (one report per row)
* `col`: name of column in `df` that contains reports
Outputs
* dataframe that contains reports that describe colorectal cancer (a subset of rows of `df`)
* dataframe that contains all matches for colorectal cancer - some of these matches are marked as excluded (`exclusion_indicator = 1`), because they do not correspond to current colorectal cancer
"""

# Find reports that describe current colorectal cancer
df_crc, matches_crc = get_crc_reports(df, col='report_text_anon', add_subj_to_matches=True, subjcol='subject')

# Get included and excluded matches
matches_incl = matches_crc.loc[matches_crc.exclusion_indicator==0]
matches_excl = matches_crc.loc[matches_crc.exclusion_indicator==1]

# Included matches ('row' corresponds to the row of input dataframe)
print('{} matches for tumour keywords were excluded'.format(matches_incl.shape[0]))
matches_incl[['row', 'left', 'target', 'right']]

# Excluded matches
print('{} matches for tumour keywords were excluded'.format(matches_excl.shape[0]))
matches_excl[['row', 'left', 'target', 'right', 'exclusion_indicator', 'exclusion_reason']]

# If some included matches are not correct after review, they can be manually excluded
# In that case, the CRC reports can be identified as
df['row'] = np.arange(df.shape[0])
df['crc_nlp'] = 0
matches_incl = matches_crc.loc[matches_crc.exclusion_indicator==0]
matches_incl_processed = matches_incl # processed matches, add any processing steps
df.loc[df.row.isin(matches_incl_processed.row), 'crc_nlp'] = 1
df_crc = df.loc[df.crc_nlp == 1]
#endregion

# ---- Step 2. Extract TNM phrases from reports ----
#region
"""
I am first running `get_tnm_phrase` to get all TNM sequences (e.g. `T1 N0 M0`) and all phrases with single TNM values (e.g. `stage: T1`). 

I am then running `add_tumour_tnm` to identify tumour keywords that occur near the TNM phrases. This can help decide which tumour the TNM phrase refers to. BUT it is not necessary to run this step.

Main arguments for `get_tnm_phrase`
* `df`  : DataFrame that contains reports
* `col` : column in `df` that contains the report text
* `remove_unusual` : remove unusual TNM phrases from output. For example, if 5 T-values are given in sequence, it is likely a multiple choice option not an actual TNM stage. True by default.
* `remove_historical`: remove TNM phrases that were marked to be historical based on nearby words. False by default, because that part of the code may not be accurate at the moment.
* `remove_falsepos`: remove phrases with single TNM values, if they do not have inclusion keywords or if they have exclusion keywords. For example, `T1-weighted` is removed, as it is not a T-stage. True by default.

Main arguments for `add_tumour_tnm`
* `df`         : dataframe that contains reports
* `matches`    : dataframe that contains matches for TNM phrases - this is the first output of 'get_tnm_phrase()' function
* `col_report` : column in `df` that contains reports
"""

# Extract TNM phrases
#  remove_historical = False, as the detection of historical TNM phrases is likely not accurate atm
matches, check_phrases, check_cleaning, check_rm = get_tnm_phrase(df=df_crc, col='report_text_anon', 
                                                                  remove_unusual=True, 
                                                                  remove_historical=False, 
                                                                  remove_falsepos=True)

# Extract tumour keywords that occur near each TNM phrase
# This can help to later decide which tumour the TNM phrase refers to
matches = add_tumour_tnm(df=df_crc, matches=matches, col_report='report_text_anon', targetcol='target_before_clean')

# View unique values for extracted phrases
check_phrases

# Check cleaning of TNM phrases
check_cleaning

# View all included matches - detailed view
cols =  ['sentence', 'left', 'target_before_split', 'target_before_clean', 'target', 
         'right', 'exclusion_indicator', 'exclusion_reason', 'phrase_with_tumour']
matches[cols]

# View all included matches - simpler view
cols =  ['left', 'target_before_clean', 'target', 'right']
matches[cols]

# View matches marked for exclusion
check_rm

# See if any matches marked for exclusion are among included matches
cols =  ['left', 'target_before_clean', 'target', 'right']
matches.loc[matches.exclusion_indicator==1, cols]
#endregion

# ---- Step 3. Extract TNM values from phrases ----
#region
"""
Arguments for `tnm.get_tnm_values()`:
* `df` : Pandas dataframe that contains reports
* `matches` : TNM phrases that were extracted for each report, output of `tnm.get_tnm_phrases()`
* `col` : name of column in `df` that contains reports
* `pathology_prefix` : if True, the output columns will have 'p' prefix, e.g. 'pT'
"""

# Get TNM values from phrases
df_crc, s = get_tnm_values(df_crc, matches=matches, col='report_text_anon', pathology_prefix=False)

# Column names in df after tnm values were added
print('Columns in df_crc:')
for i, c in enumerate(df_crc.columns):
    print('{}: {}'.format(i,c))

# View subset of output
df_crc[['report_text_anon', 'T_pre', 'T', 'N', 'M', 'T_pre_min', 'T_min', 'N_min', 'M_min']].fillna('')
#endregion

# ---- Optional. Run on multiple cores ----
#region

# Identify CRC reports, dividing the data into 10 chunks which are processed in parallel if there are at least 10 cores
df_crc, matches_crc = get_crc_reports_par(nchunks=10, njobs=-1, df=df, col='report_text_anon')

# Again, can also manually check the matches and identify CRC reports from checked matches
df['row'] = np.arange(df.shape[0])
df['crc_nlp'] = 0
matches_incl = matches_crc.loc[matches_crc.exclusion_indicator==0]
matches_incl_processed = matches_incl # processed matches, add any processing steps
df.loc[df.row.isin(matches_incl_processed.row), 'crc_nlp'] = 1
df_crc = df.loc[df.crc_nlp == 1]

# Extract TNM phrases, dividing the data into 10 chunks which are processed in parallel if there are at least 10 cores
matches, check_phrases, check_cleaning, check_rm = get_tnm_phrase_par(nchunks=10, njobs=-1, 
                                                                      df=df_crc, col='report_text_anon', 
                                                                      remove_unusual=True, 
                                                                      remove_historical=False, 
                                                                      remove_falsepos=True)

#endregion
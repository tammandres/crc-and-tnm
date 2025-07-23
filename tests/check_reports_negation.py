"""Check negation bugfix
"""

from textmining.reports import get_crc_reports
import pandas as pd
import numpy as np


col = 'report_text_anon'
reports = ['site of tumour: rectum\ntumour perforation (pT4): No',
           'tumour perforation (pT4): No']

df = pd.DataFrame(reports, columns=[col])

df_crc, matches = get_crc_reports(df.copy(), col, verbose=False)
print(matches[['row', 'target', 'exclusion_reason']])

df_crc, matches = get_crc_reports(df.copy(), col, verbose=False, negation_bugfix=True)
print(matches[['row', 'target', 'exclusion_reason']])
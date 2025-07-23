import pandas as pd
from textmining.constants import PROJECT_ROOT


col = 'report_text_anon'
reports = ['Metastatic tumour from colorectal primary, T3 N0',
           'T1 N0 MX (colorectal cancer)',
           'pT3/2/1 N0 Mx. Malignant neoplasm ascending colon',
           'pT2a/b N0 Mx (sigmoid tumour)',
           'T4a & b N0 M1 invasive carcinoma, descending colon',
           'T1-weighted image, ... rectal tumour staged as ymrT2',
           'Colorectal tumour. Stage: T4b / T4a / T3 / T2 / T1',
           'Sigmoid adenocarcinoma, ... Summary: pT1 (sigmoid, txt txt txt txt), N3b M0',
           'Colorectal tumour in situ, Tis N0 M0',
           'TO NO MX GX'
           ]
df = pd.DataFrame(reports, columns=[col])
df['subject'] = 'subj-01'

TEST_DIR = PROJECT_ROOT / 'tests' / 'test_cli'
TEST_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(TEST_DIR / 'reports.csv', index=False)


# Then to terminal:
"""
textmining tnmphrase --data ./tests/test_cli/reports.csv --column report_text_anon --remove_historical 0
textmining tnmvalues --data ./tests/test_cli/reports.csv --column report_text_anon --additional_output 0
textmining tnmvalues --data ./tests/test_cli/reports.csv --column report_text_anon --additional_output 1
"""
import pandas as pd
from textmining.constants import PROJECT_ROOT
from textmining.evaluate import evaluate_tnm
from textmining.tnm.tnm import get_tnm_phrase, get_tnm_values


# ==== Dummy data ====
test_dir = PROJECT_ROOT / 'tests' / 'test_evaluate'
test_dir.mkdir(parents=True, exist_ok=True)
TNM_PATH = test_dir / 'tnm.csv'

col = 'report_text_anon'
reports = ['Metastatic tumour from colorectal primary, T3 N0',
           'T1 N0 MX (colorectal cancer)',
           'pT3/2/1 N0 Mx. Malignant neoplasm ascending colon',
           'pT2a/b N0 Mx (sigmoid tumour)',
           'T4a & b N0 M1 invasive carcinoma, descending colon',
           'T1-weighted image, ... rectal tumour staged as ymrT2',
           'Colorectal tumour. Stage: T4b / T4a / T3 / T2 / T1',  # This is removed by default as too many Ts!
           'Sigmoid adenocarcinoma, ... Summary: pT1 (sigmoid, txt txt txt txt), N3b M0',
           'Colorectal tumour in situ, Tis N0 M0',
           'TO NO MX GX'
           ]
df = pd.DataFrame(reports, columns=[col])
df['subject'] = 'subj-01'

matches, __, __, check_rm = get_tnm_phrase(df, col, flex_start=False)
df, submatches = get_tnm_values(df, matches, col)
print(df[['report_text_anon', 'T', 'N', 'M']])

df['labelled'] = 'yes'
df.to_csv(TNM_PATH, index=False)


def test_evaluate_tnm():
    """Tests that code runs"""
    data = evaluate_tnm(truth_path=TNM_PATH, eval_path=TNM_PATH, split='train', brc='ouh')

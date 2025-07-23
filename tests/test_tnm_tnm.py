from textmining.tnm.tnm import get_tnm_phrase, get_tnm_values
import pandas as pd
import numpy as np


def test_get_tnm_phrase():
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
               'stageT1 N0 M0'
               ]
    df = pd.DataFrame(reports, columns=[col])

    for simplicity in [0, 1, 2]:
        print('\n-----')
        matches, __, __, __ = get_tnm_phrase(df, col, simplicity=simplicity)
        print(matches[['target_before_clean', 'target']])

    matches, __, __, __ = get_tnm_phrase(df, col, simplicity=2, remove_historical=False, flex_start=True)
    print(matches[['target_before_clean', 'target']])

    matches, __, __, __ = get_tnm_phrase(df, col, simplicity=2, remove_historical=False, flex_start=False)
    print(matches[['target_before_clean', 'target']])


def test_get_tnm_phrase_simplicity():
    col = 'report_text_anon'
    reports = ['T O N O M O', 'TO NO MO']
    df = pd.DataFrame(reports, columns=[col])

    matches, __, __, __ = get_tnm_phrase(df, col, simplicity=0)
    print(matches['target'])
    assert matches.shape[0] == 2

    matches, __, __, __ = get_tnm_phrase(df, col, simplicity=1)
    print(matches['target'])
    assert matches.shape[0] == 1

    matches, __, __, __ = get_tnm_phrase(df, col, simplicity=2)
    print(matches['target'])
    assert matches.shape[0] == 1


def check_get_tnm_phrase_to():
    col = 'report_text_anon'
    reports = ['TO (TO BE REPORTED) NX MX',
               'T1 TO BE REPORTED NX MX',
               'T1 NX MX']
    df = pd.DataFrame(reports, columns=[col])
    df['subject'] = 'subj-01'

    matches, __, __, __ = get_tnm_phrase(df, col)

    print(matches[['row', 'target']])


def test_get_tnm_phrase_solitary():
    col = 'report_text_anon'
    reports = ["stage T1 ... tumour T2"]
    df = pd.DataFrame(reports, columns=[col])

    matches, __, __, __ = get_tnm_phrase(df, col, extract_solitary=True)
    print(matches[['target', 'exclusion_reason']])
    matches = matches.loc[matches.solitary_indicator == 1]
    assert matches.shape[0] == 2

    col = 'report_text_anon'
    reports = ["tumour perforation (pT4):No",
               "tumour perforation (pT4): No",
               "tumour perforation (pT4) : no"]
    df = pd.DataFrame(reports, columns=[col])

    matches, __, __, check_rm = get_tnm_phrase(df, col, extract_solitary=True)
    print(check_rm)
    assert matches.shape[0] == 0

    col = 'report_text_anon'
    reports = ["T1 flair, tumour staged T2",
               "T1 flair, stage T2"]
    df = pd.DataFrame(reports, columns=[col])

    matches, __, __, check_rm = get_tnm_phrase(df, col, extract_solitary=True)
    print(check_rm[['target', 'exclusion_reason']])
    assert matches.shape[0] == 1

    col = 'report_text_anon'
    reports = ['differentiated (WHO 1990: G1))', 'differentiated (G1) in 1900']
    df = pd.DataFrame(reports, columns=[col])
    matches, __, __, check_rm = get_tnm_phrase(df, col, extract_solitary=True)
    print(check_rm[['target', 'exclusion_reason']])
    assert check_rm.shape[0] == 1

    col = 'report_text_anon'
    reports = ['RM X', 'RMX']
    df = pd.DataFrame(reports, columns=[col])
    matches, __, __, check_rm = get_tnm_phrase(df, col, extract_solitary=True)
    print(check_rm[['target', 'exclusion_reason']])
    assert check_rm.shape[0] == 1


def test_get_tnm_values():

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
               'TO NO MX GX',
               'T1 N0 MX Haggitt level III',
               'stageT4 NX MX'
               ]
    df = pd.DataFrame(reports, columns=[col])

    matches, check_phrases, check_cleaning, check_rm = get_tnm_phrase(df, col, simplicity=2, flex_start=False)

    df, submatches = get_tnm_values(df, matches, col)
    print(df[['report_text_anon', 'T', 'N', 'M', 'H']])

    t = df['T'].to_list()
    assert t == ['3', '1', '3', '2b', '4b', '2', np.nan, '1', 'is', '0', '1', '4']

    df, submatches = get_tnm_values(df, matches, col, additional_output=True)
    print(df.columns)


def test_get_tnm_values_sm_h():

    col = 'report_text_anon'
    reports = ['T1 N0 MX SM1 Haggitt 2',
               'T1 N0 MX kikuchi level III, haggit level: level 2'
               ]
    df = pd.DataFrame(reports, columns=[col])
    df['subject'] = 'subj-01'

    matches, check_phrases, check_cleaning, check_rm = get_tnm_phrase(df, col, simplicity=2, flex_start=False)

    df, submatches = get_tnm_values(df, matches, col)
    print(df[['report_text_anon', 'T', 'N', 'M', 'SM', 'H']])

    t = df['SM'].to_list()
    assert t == ['1', '3']

    t = df['H'].to_list()
    assert t == ['2', '2']

from textmining.tnm.clean import str_split, filter_words, filter_prev, clean_tnm, filter_and_clean, add_tumour_tnm
from textmining.tnm.tnm import get_tnm_phrase
from textmining.utils import extract
import numpy as np
import pandas as pd


def test_str_split():
    reports = ["txt pT1 N0 M0(tumour 1) T2, N0 MX(tumour 2)"]
    df = pd.DataFrame(reports, columns=['target'])
    df['row'] = np.arange(df.shape[0])

    df = str_split(df, 'target', pat=r'p?T')
    assert df.target.iloc[0] == "pT1 N0 M0(tumour 1) "
    assert df.target.iloc[1] == "T2, N0 MX(tumour 2)"


def test_filter_words():
    reports = ["THIS IS AN UPPERCASE TEXT"]
    df = pd.DataFrame(reports, columns=['target'])
    df['row'] = np.arange(df.shape[0])

    df, __ = filter_words(df, max_upper=5, remove=False)
    assert df.exclusion_indicator.iloc[0] == 1

    reports = ["this is lowercase"]
    df = pd.DataFrame(reports, columns=['target'])
    df['row'] = np.arange(df.shape[0])

    df, __ = filter_words(df, max_upper=5, remove=False)
    assert df.exclusion_indicator.iloc[0] == 0


def test_filter_prev():
    reports = ["There is history of condition"]
    df = pd.DataFrame(reports, columns=['report'])
    matches = extract(df, 'report', 'condition')

    matches, __ = filter_prev(matches, remove=False)
    assert matches.exclusion_indicator.iloc[0] == 1
    print(matches[['target', 'exclusion_indicator', 'exclusion_reason']])


def test_clean_tnm():
    reports = ['pT1 (sigmoid, txt txt txt txt), N3b M0']
    df = pd.DataFrame(reports, columns=['target'])
    df = clean_tnm(df, 'target')
    assert df.target.iloc[0] == 'pT1 N3b M0'

    reports = ['pT1a & b (sigmoid, txt txt txt txt), N3b M0']
    df = pd.DataFrame(reports, columns=['target'])
    df = clean_tnm(df, 'target')
    assert df.target.iloc[0] == 'pT1a/1b N3b M0'


def test_filter_and_clean():
    reports = ['history of pT1 (sigmoid, txt txt txt txt), N3b M0']
    df = pd.DataFrame(reports, columns=['report'])
    matches = extract(df, 'report', r'pT1.*N\d[a-d] M\d')
    matches, __, __, __ = filter_and_clean(matches)
    print(matches)


def test_add_tumour_tnm():
    reports = ['pT1a & b (sigmoid adenocarcinoma, txt txt txt txt), N3b M0']
    df = pd.DataFrame(reports, columns=['report'])
    matches, __, __, __ = get_tnm_phrase(df, 'report')

    df = add_tumour_tnm(df, matches, col_report='report')
    print(df.phrase_with_tumour)

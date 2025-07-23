from textmining.tnm.extract import _extract_tnm_sequence, tnm_phrase
from textmining.tnm.pattern import _tnm_sequence, _simple_tnm_sequence, _tnm_info, _build_tnm, _simple_tnm_value
from textmining.utils import mix_case, wrap_pat
import pandas as pd
import time
import regex as re


def get_reports():
    col = 'report_text_anon'
    reports = ['Metastatic tumour from colorectal primary, T3 N0',
               'T1 N0 MX (colorectal cancer)',
               'pT3/2/1 N0 Mx. Malignant neoplasm ascending colon',
               'pT2a/b N0 Mx (sigmoid tumour)',
               'T4a & b N0 M1 invasive carcinoma, descending colon',
               'T1-weighted image, ... rectal tumour staged as ymrT2',
               'Colorectal tumour. Stage: T4b / T4a / T3 / T2 / T1',
               'Sigmoid adenocarcinoma, ... Summary: pT1 (sigmoid, txt txt txt txt), N3b M0',
               'Colorectal tumour in situ, Tis N0 M0'
               ]
    df = pd.DataFrame(reports, columns=[col])
    df['subject'] = 'subj-01'
    return df, col


def test_extract_tnm_sequence():
    df, col = get_reports()

    pat_tnm = _tnm_sequence()
    pat_flex = _tnm_sequence(sequence_type='flexible')
    tic = time.time()
    matches = _extract_tnm_sequence(df, 'report_text_anon', pat_tnm, pat_flex)
    toc = time.time()
    print(toc-tic)
    assert len(matches) == (len(df) - 1)


def test_extract_tnm_sequence_simple():
    df, col = get_reports()

    pat_tnm = _simple_tnm_sequence()
    tic = time.time()
    matches = _extract_tnm_sequence(df, 'report_text_anon', pat_tnm)
    toc = time.time()
    print(toc-tic)

    assert len(matches) == (len(df))


def test_tnm_phrase():

    df, col = get_reports()

    times = []
    for s in [0, 1, 2]:
        tic = time.time()
        m = tnm_phrase(df, col, simplicity=s)
        toc = time.time()
        times.append(toc - tic)
        print(m.target)
    print(times)


def test_tnm_phrase_additional():

    col = 'report_text_anon'
    reports = ['T0 N0 M0 PNI0 V0 R0']
    df = pd.DataFrame(reports, columns=[col])

    m = tnm_phrase(df, col)
    print(m[['target', 'solitary_indicator']])
    assert m.solitary_indicator.item() == 0

    col = 'report_text_anon'
    reports = ['T1a N1']
    df = pd.DataFrame(reports, columns=[col])
    m = tnm_phrase(df, col)
    print(m[['target', 'solitary_indicator']])
    assert m.solitary_indicator.item() == 0


def check_t1a_n1a():

    key = 'T'
    info = _tnm_info(key)

    prefix_values = _tnm_info('pre').values
    prefix = mix_case(wrap_pat(prefix_values))  # Letters in prefix can have mixed case
    prefix = prefix + '{1,5}'  # Prefix can be of length 1 to 5

    letter_set = info.letters
    let_norm = wrap_pat(letter_set)

    value = info.val_pat  # Regex for all possible values
    value = mix_case(value, add_bracket=False)

    norm = _build_tnm(pre=prefix, let=let_norm, gap='', val=value, rep=True, context='none')
    m = re.findall(norm, 'T1aN1a')
    print(m)

    norm = _build_tnm(pre=prefix, let=let_norm, gap='', val=value, rep=True, context='gap_or_value')
    m = re.findall(norm, 'T1aN1a')
    print(m)

    norm = _build_tnm(pre=prefix, let=let_norm, gap='', val=value, rep=True, context='none')
    tnm_pat = _simple_tnm_value(single_pattern=True, capture=False, zero=True)
    norm = norm + tnm_pat
    m = re.findall(norm, 'T1aN1a')
    print(m)


def test_tnm_phrase_solitary_kikuchi():
    col = 'report_text_anon'
    reports = ['Kikuchi SM2', 'SM2']
    df = pd.DataFrame(reports, columns=[col])

    m = tnm_phrase(df, col)
    print(m[['target', 'solitary_indicator', 'exclusion_indicator', 'exclusion_reason']])
    assert m.exclusion_indicator.to_list() == [0, 1]


def test_tnm_phrase_solitary_t():
    col = 'report_text_anon'
    reports = ['T1 weighted', 'stage T1', 'T1 CRC', 'pT4: No', 'pT4 : No']
    df = pd.DataFrame(reports, columns=[col])

    m = tnm_phrase(df, col)
    print(m[['target', 'exclusion_reason']])
    assert m.exclusion_indicator.to_list() == [1, 0, 0, 1, 1]


def test_tnm_phrase_solitary_g():
    col = 'report_text_anon'
    reports = ['G1', 'Differentiated (G1)', 'WELL DIFFERENTIATED XXXXXXXXXXXXXXX TUMOUR (G1)',
               'differentiated (WHO 1900: G1)', 'differentiated (G1) in 1900',
               'TRG1 grade']
    df = pd.DataFrame(reports, columns=[col])

    m = tnm_phrase(df, col)

    print(m[['row', 'target', 'exclusion_reason']])
    assert m.exclusion_indicator.to_list() == [1, 0, 0, 0, 0]


def test_tnm_phrase_solitary_n():
    col = 'report_text_anon'
    reports = ['N0', 'tumour (N0)', 'nodes N0',
               'N1 malignant pelvic sidewall nodes/inguinal nodes :No']
    df = pd.DataFrame(reports, columns=[col])

    m = tnm_phrase(df, col)
    print(m[['target', 'exclusion_reason']])
    assert m.exclusion_indicator.to_list() == [1, 0, 0, 1]


def test_tnm_phrase_solitary_n():
    col = 'report_text_anon'
    reports = ['R0', 'margin (R0)', 'not invade (R0)']
    df = pd.DataFrame(reports, columns=[col])

    m = tnm_phrase(df, col)
    print(m[['target', 'exclusion_reason']])
    assert m.exclusion_indicator.to_list() == [1, 0, 0]

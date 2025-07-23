from textmining.recurrence import _extract_concept, _assign_context, get_recurrence
import pandas as pd


def test_extract_concept():
    col = 'report_text_anon'
    reports = ['Metastatic tumour observed',
               'There is no evidence of recurrence',
               'No definite recurrence seen',
               'Nodes are suspicious for recur.',
               'Potential regrowth observed at TEM',
               'Evidence of mets is clear'
               ]
    df = pd.DataFrame(reports, columns=[col])

    matches = _extract_concept(df, col)
    print(matches['target'])

    t = matches.sort_values(by='row').target.to_list()
    assert t == ['Metastatic', 'recurrence', 'recurrence', 'recur', 'regrowth', 'mets']


def test_assign_context():
    col = 'report_text_anon'
    reports = ['Metastatic tumour observed',
               'There is no evidence of recurrence',
               'No definite recurrence seen',
               'Nodes are suspicious for recur.',
               'Potential regrowth observed at TEM',
               'Evidence of mets is clear'
               ]
    df = pd.DataFrame(reports, columns=[col])

    matches = _extract_concept(df, col)
    matches = _assign_context(matches, verbose=False)
    print(matches['target'])

    matches = matches.sort_values(by='row')

    assert matches.recur_negated.to_list() == [0, 1, 1, 0, 0, 0]
    assert matches.recur_possible.to_list() == [0, 0, 0, 1, 1, 0]


def test_get_recurrence():
    col = 'report_text_anon'
    reports = ['Metastatic tumour observed',
               'There is no evidence of recurrence',
               'No definite recurrence seen',
               'Nodes are suspicious for recur.',
               'Potential regrowth observed at TEM',
               'Evidence of mets is clear'
               ]
    df = pd.DataFrame(reports, columns=[col])

    for u in [True, False]:
        for v in [True, False]:
            __ , matches = get_recurrence(df.copy(), col, conservative_present=u, conservative_locoregional=v)



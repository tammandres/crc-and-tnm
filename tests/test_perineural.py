from textmining.perineural import get_pn
import pandas as pd


def test_get_pn():
    col = 'report_text_anon'
    reports = ['perineural invasion: yes',
               'there is no perineural invasion']
    df = pd.DataFrame(reports, columns=[col])
    df['subject'] = 'subj-01'

    df, matches = get_pn(df, col)
    print(df)

    assert df.Pn.iloc[0] == '1'
    assert df.Pn.iloc[1] == '0'

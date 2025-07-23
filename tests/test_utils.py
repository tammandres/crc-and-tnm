import textmining.utils as ut
import pandas as pd


def test_remove_duplicates():

    reports = ['black magic cat',
               'black cat',
               'magic']
    col = 'report'
    df = pd.DataFrame(reports, columns=[col])

    m0 = ut.extract(df, col, 'black')
    m1 = ut.extract(df.iloc[[1, 2]], col, 'bla')
    m1 = ut.remove_duplicates(m0, m1)

    assert m1.shape[0] == 0

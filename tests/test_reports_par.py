from textmining.reports import get_crc_reports, get_crc_reports_par
import pandas as pd


def test_get_crc_reports():
    col = 'report_text_anon'
    reports = ['T1 N0 MX (colorectal cancer)',
               'Malignant neoplasm ascending colon',
               'pT2a/b N0 Mx (sigmoid tumour)',
               'T4a & b N0 M1 invasive carcinoma, descending colon',
               'T1-weighted image, ... rectal tumour staged as ymrT2',
               'Colorectal tumour. Stage: T4b / T4a / T3 / T2 / T1',
               'Sigmoid adenocarcinoma, ... Summary: pT1 (sigmoid, txt txt txt txt), N3b M0',
               'Colorectal tumour in situ, Tis N0 M0',
               'tumour in ascending colon',
               'there is malignant new rectal neoplasm',
               ]
    reports_ex = ['No colorectal tumour observed',
                  'There is no obvious malignant neoplasm',
                  'Metastatic tumour',
                  'patients with rectal tumours',
                  'resistance in gastrointestinal stromal tumours',
                  'tumour perforation (pTXx): No',
                  'correlates of x x in x x tumors',
                  'txt in the context of colorectal cancer',
                  'colonic adenocarcinomas can show',
                  'Colorectal Cancer : Guideline From',
                  'positivity in intratumoral',
                  'occurring in this tumour',
                  'consistent with metastatic colorectal carcinoma. The tumour is',
                  'There is no  malignant neoplasm',
                  'There is no  (obvious)  malignant neoplasm',
                  'may be associated with xxxxxxxxxx rectal carcinomas',
                  ]

    # Check that relevant reports are detected in the same way
    df = pd.DataFrame(reports, columns=[col])
    df['excl'] = 0
    df_crc, matches = get_crc_reports(df.copy(), col, verbose=False)
    print(matches[['row', 'target', 'exclusion_reason']])
    assert df.shape[0] == df_crc.shape[0]

    df_crc_par, matches_par = get_crc_reports_par(nchunks=3, njobs=3, df=df.copy(), col=col)
    print(matches_par[['row', 'target', 'exclusion_reason']])
    test = matches.fillna('').drop(labels='subrow', axis=1).values == matches_par.fillna('').drop(labels='subrow', axis=1).values
    assert test.all()
    test = df_crc.values == df_crc_par.values
    assert test.all()
    
    # Check that irrelevant reports are excluded in the same way
    df2 = pd.DataFrame(reports_ex, columns=[col])
    df2['excl'] = 1
    df2_crc, matches2 = get_crc_reports(df2.copy(), col, verbose=False)
    assert df2_crc.shape[0] == 0

    df_crc_par2, matches_par2 = get_crc_reports_par(nchunks=3, njobs=3, df=df2.copy(), col=col)
    print(matches_par2[['row', 'target', 'exclusion_reason']])
    test = matches2.fillna('').drop(labels='subrow', axis=1).values == matches_par2.fillna('').drop(labels='subrow', axis=1).values
    assert test.all()
    assert df_crc_par2.shape[0] == 0

    # Check on both relevant and irrelevant
    df3 = pd.concat(objs=[df, df2], axis=0).reset_index(drop=True)
    df_crc, matches = get_crc_reports(df3.copy(), col, verbose=False)
    print(matches[['row', 'target', 'exclusion_reason']])
    assert df.shape[0] == df_crc.shape[0]

    df_crc_par, matches_par = get_crc_reports_par(nchunks=3, njobs=3, df=df3.copy(), col=col)
    print(matches_par[['row', 'target', 'exclusion_reason']])
    test = matches.fillna('').drop(labels='subrow', axis=1).values == matches_par.fillna('').drop(labels='subrow', axis=1).values
    assert test.all()
    test = df_crc.values == df_crc_par.values
    assert test.all()
    
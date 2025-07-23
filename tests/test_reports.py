from textmining.reports import get_crc_reports, pat_from_vocab
from textmining.utils import constrain_distance, add_spell_variation, prep_pat
from textmining.constants import VOCAB_DIR
from textmining.spelling import edits1
import pandas as pd
import regex as re


def testpat_from_vocab_crc():
    v = pd.read_csv(VOCAB_DIR / 'vocab_site_and_tumour.csv')
    vcrc = v.loc[v.cui == 12]
    pats = pat_from_vocab(vcrc, gap=r'\s{1,3}', verbose=False)

    strings = ['colorectal tumour', 'colorectal   carcinoma', 'colo-rectal neoplasm', 'colorectal cancer']
    for string in strings:
        match = re.findall(pats, string)[0]
        print(string, match)
        assert string == match


def testpat_from_vocab_tumour():
    v = pd.read_csv(VOCAB_DIR / 'vocab_site_and_tumour.csv')
    vtum = v.loc[v.cui == 11]
    pats = pat_from_vocab(vtum, gap=r'\s{1,3}', verbose=False)

    strings = ['tumour', 'tumor', 'carcinom', 'carcinoma', 'carcinoid', 'malignant neoplasm',
               'malignant colorectal neoplasm']
    for string in strings:
        match = re.findall(pats, string)[0]
        print(string, match)
        assert string == match


def testpat_from_vocab_site():
    """Test detection of CRC sites"""

    # Targets
    strings = ['caecum', 'caecal', 'cecal', 'cecum', 'right colon', 'right hemicolon',
               'ascending colon', 'right hemicolectomy', 'right hemicolect', 'hepatic flexure',
               'right colic flexure', 'transverse colon', 'splenic flexure', 'left colic flexure',
               'left colon', 'left hemicolon', 'descending colon', 'left hemicolectomy',
               'sigmoid', 'mesorectal', 'anorectal', 'mesorectum', 'mesorectal', 'anal',
               'transanal', 'colon', 'colonic', 'colonoscopy', 'colo-rectal', 'crc', 'dukes',
               'large bowel', 'large intestine', 'bowel wall', 'rectosigmoid', 'recto-sigmoid',
               'recto/sigmoid', 'kikuchi', 'haggitt', 'haggit', 'hagit']

    # Patterns
    v = pd.read_csv(VOCAB_DIR / 'vocab_site_and_tumour.csv')
    vsite = v.loc[v.cui.isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 35, 39])]
    print('\nSites included: {}'.format(vsite.concept.unique()))

    pats_site = pat_from_vocab(vsite, gap=r'\s{1,3}', verbose=False, name='sites')
    site_left = constrain_distance(pats_site, side='left', char='.', distance=10)
    site_right = constrain_distance(pats_site, side='right', char='.', distance=10)

    # Dummy check that site_left and site_right patterns can detect all sites
    for string in strings:
        match = re.findall(site_left, string)[0]
        print(string, match)
        assert string == match

    for string in strings:
        match = re.findall(site_right, string)[0]
        print(string, match)
        assert string == match

    # Dummy check that site_left and site_right patterns can detect all sites at specified distance
    dist_left = [s + ' ' * 20 for s in strings]
    dist_right = [' ' * 20 + s for s in strings]
    near_left = [s + ' ' * 9 for s in strings]
    near_right = [' ' * 9 + s for s in strings]

    for string in dist_left:
        match = re.findall(site_left, string)
        assert not match

    for string in dist_right:
        match = re.findall(site_right, string)
        assert not match

    for site, string in zip(strings, near_left):
        match = re.findall(site_left, string)[0]
        print(site, match)
        assert site == match

    for site, string in zip(strings, near_right):
        match = re.findall(site_right, string)[0]
        print(site, match)
        assert site == match


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
               'there is malignant new rectal neoplasm'
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

    df = pd.DataFrame(reports, columns=[col])
    df['excl'] = 0
    df_crc, matches = get_crc_reports(df.copy(), col, verbose=False)
    print(matches[['row', 'target', 'exclusion_reason']])
    assert df.shape[0] == df_crc.shape[0]

    df2 = pd.DataFrame(reports_ex, columns=[col])
    df2['excl'] = 1
    df2_crc, matches = get_crc_reports(df2.copy(), col, verbose=False)
    assert df2_crc.shape[0] == 0


def check_reports_spellcheck():

    vocab = pd.read_csv(VOCAB_DIR / 'vocab_site_and_tumour.csv')
    pat = vocab.pat.iloc[28]

    gap = r'\s{1,3}'
    pat = prep_pat(pat=pat, pat_type='wordstart', gap=gap, wdelim=r'\W')

    spell = pd.read_csv(VOCAB_DIR / 'spellcheck.csv')

    pat = add_spell_variation(pat, spell)
    print(pat)


def explore_edits():

    print(edits1('colorectal'))


def test_get_crc_reports_spellcheck():
    col = 'report_text_anon'
    reports = ['T1 N0 MX (colorectal cncer)']

    df = pd.DataFrame(reports, columns=[col])
    df_crc, matches = get_crc_reports(df, col, verbose=False, spellcheck=False)
    print(df_crc)
    assert df_crc.shape[0] == 0

    df = pd.DataFrame(reports, columns=[col])
    df_crc, matches = get_crc_reports(df, col, verbose=False, spellcheck=True)
    print(df_crc)
    assert df_crc.shape[0] == 1

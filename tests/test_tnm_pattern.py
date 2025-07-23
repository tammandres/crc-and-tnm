"""Test TNM patterns"""
import regex as re
from textmining.tnm.pattern import _repeated_value, _constrain_context, _prepare_prefix
from textmining.tnm.pattern import _simple_tnm_value, _simple_tnm_sequence
from textmining.tnm.pattern import _tnm_value, _tnm_value_variations
from textmining.tnm.pattern import _tnm_info, _build_tnm, _tnm_sequence
from dataclasses import asdict
from textmining.utils import wrap_pat, mix_case


def check_build_tnm():
    p = _build_tnm('<pre>', let='<T>', val='<[01234]>', gap='', rep=False, context='none')
    print(p)

    p = _build_tnm('<pre>', let='<T>', val='<[01234]>', gap='', rep=False, context='gap')
    print(p)

    p = _build_tnm('<PREFIX>', let='<LETTER>', val='<[01234]>', gap='', rep=False, context='gap_or_value')
    print(p)

    p = _build_tnm('[ypt]', let='T', val='[01234]', gap='', rep=False, context='none')
    print(re.findall(p, 'T1'))
    print(re.findall(p, 'pT2'))

    p = _build_tnm('[ypt]', let='T', val='[01234]', gap='', rep=False, context='gap_or_value')
    print(re.findall(p, 'pT1'))


# ==== Simple TNM values and sequences ====
def test_simple_tnm_value():
    pat = _simple_tnm_value()
    in_out = {'pT1': ['pT1'],
              'N0': ['N0'],
              'T1a N1a M1': ['T1a', 'N1a', 'M1'],
              'T1a/b/c': ['T1a/b/c'],
              'T1/T2': ['T1/T2'],
              'N1a & b': ['N1a & b'],
              'R0': ['R0'],
              'N1 (0/2)': ['N1'],
              'TO': ['TO']
              }
    for inp, out in in_out.items():
        match = re.findall(pat, inp)
        match = [''.join(m) for m in match]
        print(inp, match)
        assert match == out


def test_simple_tnm_value_all():
    """Check that the TNM value pattern is able to detect all valid (normally written) TNM values"""
    pat = _simple_tnm_value()
    keys = ['T', 'N', 'M', 'L', 'V', 'R', 'SM', 'H', 'Pn']
    for key in keys:
        print(key)
        info = _tnm_info(key)
        for let in info.letters:
            for val in info.values:
                inp = let + val
                match = re.findall(pat, inp)
                match = [''.join(m) for m in match]
                print(inp, match)
                assert match == [inp]


def test_simple_tnm_sequence():
    pat = _simple_tnm_sequence()
    in_out = {'ypT1 N0 M0': ['ypT1 N0 M0'],
              'T1a/b (text text text) N0 M1': ['T1a/b (text text text) N0 M1'],
              'ypT1 N0 M0 text text text': ['ypT1 N0 M0'],
              'TO NO MO': ['TO NO MO']
              }
    for inp, out in in_out.items():
        match = [re.search(pat, inp).group(0)]
        print(inp, match)
        assert match == out


# ==== Constrained TNM values and sequences
def test_n_value():
    pat = _tnm_value_variations('N').comb
    in_out = {'pN0': ['pN0'],
              'PN0': [],
              'T1 PN0': ['PN0'],
              'T1 N1 M0 PN0': ['N1'],
              'pn0': ['pn0'],
              'pN X': ['pNX'],  # pN X is full match, but as I concat groups below it is pNX
              'COLON X': []  # LO not considered a preceding TNM value, as it is not preceded by gap or TNM-like str
              }
    for inp, out in in_out.items():
        match = re.findall(pat, inp)
        match = [''.join(m) for m in match]
        print(inp, match)
        assert match == out


def test_pn_value():
    pat = _tnm_value_variations('Pn').comb
    in_out = {'pN0': [],
              'PN0': [],
              'N1 PN0': ['PN0'],
              'pn0': []
              }
    for inp, out in in_out.items():
        match = re.findall(pat, inp)
        match = [''.join(m) for m in match]
        print(inp, match)
        assert match == out


def test_tnm_value():
    pats = asdict(_tnm_value())
    keys = ['T', 'N', 'M', 'L', 'V', 'R', 'SM', 'H', 'Pn', 'G']
    pat = wrap_pat([pats[key]['comb'] for key in keys])
    in_out = {'pT1': ['pT1'],
              'N0': ['N0'],
              'T1a N1a M1': ['T1a', 'N1a', 'M1'],
              'T1a/b/c': ['T1a/b/c'],
              'N1a & b': ['N1a & b'],
              'R0': ['R0'],
              'TO': [],
              'TO N1': ['TO', 'N1'],
              'R0 TO': ['R0', 'TO'],
              'T0 Kikuchi 1': ['T0', 'Kikuchi 1'],
              'T1 text haggitt 2': ['T1', 'haggitt 2'],
              'G1': ['G1'],
              'T1/T2': ['T1/T2'],
              'T2 / 3': ['T2 / 3'],
              'N1 (2/5)': ['N1'],
              'N1 (0/2)': ['N1'],
              'N1 (or 2)': ['N1 (or 2)'],
              'T1 ( or 2 or 3)': ['T1 ( or 2 or 3)'],
              'T1 (or 2/3)': ['T1 (or 2/3)'],
              'T1 (2/3)': ['T1'],
              'T1 (2/3/4)': ['T1 (2/3/4)'],
              'Pn0': ['Pn0'],
              'PNI1': ['PNI1'],
              'pnI0': [],
              'pni0': [],
              'sm1': ['sm1'],
              'sM1': ['sM1'],
              'Sm1': ['Sm1']
              }
    for inp, out in in_out.items():
        match = re.findall(pat, inp)
        match = [''.join(m) for m in match]
        print(inp, match)
        assert match == out


def test_tnm_value_sm_h():
    pats = asdict(_tnm_value())
    keys = ['SM', 'H']
    pat = wrap_pat([pats[key]['comb'] for key in keys])
    in_out = {'sm1': ['sm1'],
              'sM1': ['sM1'],
              'Sm1': ['Sm1'],
              'haggit level IV': ['haggit level IV'],
              'haggit levels IV': ['haggit levels IV'],
              'haggitt levels: level III': ['haggitt levels: level III'],
              'haggit level is at least II': ['haggit level is at least II'],
              'Hagitt level 2/3': ['Hagitt level 2/3'],
              'sm level III': ['sm levelIII'],
              'kikuchi levels III': ['kikuchi levels III'],
              'kikuchi levels: level III': ['kikuchi levels: level III'],
              'kikuchi level is at least II': ['kikuchi level is at least II'],
              'kikuchi level 2/3': ['kikuchi level 2/3'],

              }
    for inp, out in in_out.items():
        match = re.findall(pat, inp)
        match = [''.join(m) for m in match]
        print(inp, match)
        assert match == out


def test_tnm_value_all():
    """Check that the TNM value pattern is able to detect all valid (normally written) TNM values"""
    pats = asdict(_tnm_value())
    keys = ['T', 'N', 'M', 'L', 'V', 'R', 'SM', 'H', 'Pn', 'G']
    pat = wrap_pat([pats[key]['comb'] for key in keys])
    for key in keys:
        info = _tnm_info(key)
        for let in info.letters:
            for val in info.values:
                inp = let + val
                match = re.findall(pat, inp)
                match = ''.join(match[0])
                print(inp, match)
                assert match == inp


def test_tnm_sequence_flex():
    """Note: does not match for single TNM values"""
    pat = _tnm_sequence(sequence_type='flexible')
    in_out = {'ypT1 N0 M0': ['ypT1 N0 M0'],
              'T1a/b (text text text) N0 M1': ['T1a/b (text text text) N0 M1'],
              'ypT1 N0 M0 text text text': ['ypT1 N0 M0'],
              }
    for inp, out in in_out.items():
        match = re.search(pat, inp)
        if match:
            match = [match.group(0)]
        else:
            match = []
        print(inp, match)
        assert match == out


def test_tnm_sequence_constrained():
    """Note: does not match for single TNM values"""
    pat = _tnm_sequence(sequence_type='constrained')
    in_out = {'ypT1 N0 M0': ['ypT1 N0 M0'],
              'TO N0 M0': ['TO N0 M0'],
              'T1a/b (text text text) N0 M1': ['T1a/b (text text text) N0 M1'],
              'ypT1 N0 M0 text text text': ['ypT1 N0 M0'],
              'T1 N0': ['T1 N0'],
              'T1 NO': ['T1 NO'],
              'T1 (text text text text text text) NO': [],
              'T1 (text) N1 M0 HagGitt 2': ['T1 (text) N1 M0 HagGitt 2']
              }
    for inp, out in in_out.items():
        match = re.search(pat, inp)
        if match:
            match = [match.group(0)]
        else:
            match = []
        print(inp, match)
        assert match == out


def test_tnm_value_variations():
    keys = ['T', 'N', 'M', 'L', 'V', 'R', 'SM', 'H', 'Pn', 'G']
    for key in keys:
        print('\n {}'.format(key))
        pat = _tnm_value_variations(key)
        print(pat)


# ==== Lower level functions
def test_prepare_prefix():
    """"""
    pre = r'[acprmy]{1,5}'
    pat = _prepare_prefix(pre, pre_gap=' ?')
    in_out = {'ypT1': ['yp'], 'T1cN1': [], 'T1c N1': [], 'T1 cN1': ['c'], 'c T1': [], 'cT1': ['c'],
              'a T1': [], 'aT1': ['a'], 'ca T1': [], 'caT1': ['ca'], '(yp) T1': ['yp'], '(yp)T1': ['yp']}
    for inp, out in in_out.items():
        match = re.findall(pat, inp)
        print(inp, match)
        assert match == out


def test_constrain_context_gap_or_value():
    pat = 'T[0-4Xx][a-dA-D]?'
    pat = _constrain_context(pat, context='gap_or_value')
    in_out = {'T1/9': [],
              'T1': ['T1'],
              'T1N1': ['T1'],
              'T1aN1a': ['T1a'],
              'T1N5': [],
              'T1 ': ['T1'],
              'N1T1': ['T1'],
              'S1T1': []
              }
    for inp, out in in_out.items():
        match = re.findall(pat, inp)
        print(inp, match)
        assert match == out


def test_constrain_context_gap():
    pat = 'T[0-4Xx][a-dA-D]?'
    pat = _constrain_context(pat, context='gap')
    in_out = {'T1/9': [],
              'T1': ['T1'],
              'T1N1': [],
              'T1N5': [],
              'T1 ': ['T1']
              }
    for inp, out in in_out.items():
        match = re.findall(pat, inp)
        print(inp, match)
        assert match == out


def test_constrain_context_value_before():
    pat = 'T[0-4Xx][a-dA-D]?'
    pat = _constrain_context(pat, context='value_before')
    in_out = {'T1/9': [],
              'T1': [],
              'T1N1': [],
              'N1T1': ['T1'],
              'T1 N1': [],
              'R0 T1': ['T1'],
              'R0 / T1': ['T1'],
              'R0T1 ': ['T1']}
    for inp, out in in_out.items():
        match = re.findall(pat, inp)
        print(inp, match)
        assert match == out


def test_constrain_context_value_after():
    pat = 'T[0-4Xx][a-dA-D]?'
    pat = _constrain_context(pat, context='value_after')
    in_out = {'T1/9': [], 'T1': [], 'T1N1': ['T1'], 'T1 N1': ['T1'], 'T1 / N1': ['T1'], 'R0 T1': [], 'R0T1 ': []}
    for inp, out in in_out.items():
        match = re.findall(pat, inp)
        print(inp, match)
        assert match == out


def test_constrain_context_precededbyn():
    pat = 'T[0-4Xx][a-dA-D]?'
    pat = _constrain_context(pat, context='preceded_by_n')
    in_out = {'N1 T1': ['T1'], 'T1 N1': [], 'N1T1':['T1'], 'N1 / T1':['T1']}
    for inp, out in in_out.items():
        match = re.findall(pat, inp)
        print(inp, match)
        assert match == out


def test_repeated_value():
    val = r'[1-4][A-Da-d]?'
    pat = _repeated_value(val)
    in_out = {'4a': ['4a'], '4a/b': ['4a/b'], '4 a / b': ['4'], '4a & b': ['4a & b'], '4a or b': ['4a or b'],
              '1/9': ['1'], '1/c': ['1/c'], '1a//b': ['1a'], '2a/b/c/d': ['2a/b/c/d']}
    for inp, out in in_out.items():
        match = re.findall(pat, inp)
        print(inp, match)
        assert match == out


def test_repeated_value_iv():
    val = r'IV|I{1,3}'
    pat = _repeated_value(val)
    in_out = {'III': ['III'], 'II/III': ['II/III'], 'III or IV': ['III or IV']}
    for inp, out in in_out.items():
        match = re.findall(pat, inp)
        print(inp, match)
        assert match == out


def test_repeated_value_with_context():
    """Note: repeated_value currently matches for '1/c' which may not be valid,
    but for simplicity I am not correcting that."""
    val = r'[1-4][A-Da-d]?'
    pat = _repeated_value(val)
    pat = _constrain_context(pat, 'gap_or_value')
    in_out = {'1/9': [], '1/c': ['1/c'], '1a//b': []}
    for inp, out in in_out.items():
        match = re.findall(pat, inp)
        print(inp, match)
        assert match == out


def test_mix_case():
    input = 'cat'
    output = '[cC][aA][tT]'
    assert output == mix_case(input)

"""
def text_not_preceded_by():
    exclude = _not_preceded_by('crc')
    targets = ['crcT1', 'cRcT1', 'crc T1']
    outputs = [[], [], ['crc ']]
    pat = r'[cr]{1,5}\s?' + exclude
    for target, out in zip(targets, outputs):
        match = re.findall(pat, target)
        print(match)
        assert out == match
"""

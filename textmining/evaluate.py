"""Compare extracted data vs ground-truth labels"""
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from sklearn.metrics import confusion_matrix
from statsmodels.stats.proportion import proportion_confint
from textmining.constants import RESULTS_DIR
from textmining.reports import get_crc_reports


def evaluate_crc(truth_path: Path, eval_path: Path, split: str, brc: str, ci_method: str = 'wilson',
                 print_errors: bool = False):

    # Columns to be compared
    cols = ['crc_nlp']

    # Load and prepare data for comparison
    df0, df1 = _prepare_data(truth_path, eval_path, cols, replace_n=True)

    print(df0.crc_nlp.unique(), df1.crc_nlp.unique())
    df0.crc_nlp = df0.crc_nlp.replace({'0': 'null'})
    df1.crc_nlp = df1.crc_nlp.replace({'0': 'null'})
    print(df0.crc_nlp.unique(), df1.crc_nlp.unique())

    data = {'msum': pd.DataFrame(), 'msum_print': pd.DataFrame()}
    for rtype in df0.report_type.unique():
        df0sub = df0.loc[df0.report_type == rtype].copy()
        df1sub = df1.loc[df1.report_type == rtype].copy()

        y0 = df0sub['crc_nlp']
        y1 = df1sub['crc_nlp']
        msum, msum_print = diagnostic_metrics_vs_null(y0, y1, index_value=None, index_name=None, ci_method=ci_method)
        for d in [msum, msum_print]:
            d['report_type'] = rtype
        data['msum'] = pd.concat(objs=[data['msum'], msum], axis=0)
        data['msum_print'] = pd.concat(objs=[data['msum_print'], msum_print], axis=0)

    for key, d in data.items():
        d = d.reset_index(drop=True)
        d['split'] = split
        d['brc'] = brc
        d = d[['brc', 'split', 'report_type'] + [c for c in d.columns if c not in ['brc', 'split', 'report_type']]]
        data[key] = d

    # Save
    suf = '_brc-' + brc + '_split-' + split + '.csv'
    fnames = {'msum': 'results-crc_sum' + suf,
              'msum_print': 'results-crc_sum-print' + suf}
    RESULTS_DIR.mkdir(exist_ok=True)
    for key, d in data.items():
        d.to_csv(RESULTS_DIR / fnames[key], index=False)
    
    # For error analysis
    mask = df0.crc_nlp != df1.crc_nlp
    cols = ['brc', 'report_type', 'report_text_anon', 'crc_nlp']
    e0 = df0.loc[mask, cols].rename(columns={'crc_nlp': 'crc_true'})
    e1 = df1.loc[mask, cols].rename(columns={'crc_nlp': 'crc_pred'})
    e = e0.merge(e1, how='left')

    RESULTS_DIR.mkdir(exist_ok=True)
    e.to_csv(RESULTS_DIR / ('results-crc_errors' + suf), index=False)

    __, m = get_crc_reports(e, 'report_text_anon', add_subj_to_matches=False, subjcol='subject')
    m['phrase'] = m.left.str.lower() + m.target.str.upper() + m.right.str.lower()
    m.phrase = m.phrase.str.replace(r'\n|\r', '<n>', regex=True)
    e['row'] = np.arange(e.shape[0])
    e = e.merge(m[['row', 'left', 'target', 'right', 'phrase', 'exclusion_indicator', 'exclusion_reason']])

    RESULTS_DIR.mkdir(exist_ok=True)
    e.to_csv(RESULTS_DIR / ('results-crc_errors-withmatches' + suf), index=False)

    if print_errors:
        print('Total number or errors: {}'.format(e.shape[0]))
        print(e[['crc_true', 'crc_pred']].value_counts())
        for i, row in e.iterrows():
            print('\n-------true {}, pred {}'.format(row.crc_true, row.crc_pred))
            print(row.report_text_anon)
    
        for i, row in e.iterrows():
            print('\n---report: {}, excl reason: {}, crc true: {}, crc pred: {}'.format(row.row, row.exclusion_reason, row.crc_true, row.crc_pred))
            print(row.phrase)


def evaluate_tnm(truth_path: Path, eval_path: Path, split: str, brc: str, ci_method: str = 'wilson', suffix: str = ''):
    print('\nComparing ground-truth TNM values to predicted TNM values...')

    # Columns to be compared
    cols_max = ['T_pre', 'T', 'N', 'M', 'V', 'R', 'L', 'Pn', 'SM', 'H', 'G']
    cols_min = [c + '_min' for c in cols_max]
    cols = cols_max + cols_min

    # Load and prepare data for comparison
    df0, df1 = _prepare_data(truth_path, eval_path, cols)

    # Additionally ensure T_pre is sorted for comparison, e.g. 'py' and 'yp' are equivalent
    df0.T_pre = df0.T_pre.apply(lambda x: ''.join(sorted(x) if x != 'null' else 'null'))
    df1.T_pre = df1.T_pre.apply(lambda x: ''.join(sorted(x) if x != 'null' else 'null'))

    # Compute performance for each TNM category (e.g. T and N), separately for each report type
    data = {'msum': pd.DataFrame(), 'msum_print': pd.DataFrame(), 
            'mcat': pd.DataFrame(), 'mcat_print': pd.DataFrame()}
    for rtype in df0.report_type.unique():
        df0sub = df0.loc[df0.report_type == rtype].copy()
        df1sub = df1.loc[df1.report_type == rtype].copy()
        msum, msum_print, mcat, mcat_print = compute_metrics(df0sub, df1sub, cols, ci_method=ci_method)
        for d in [msum, msum_print, mcat, mcat_print]:
            d['report_type'] = rtype

        data['msum'] = pd.concat(objs=[data['msum'], msum], axis=0)
        data['msum_print'] = pd.concat(objs=[data['msum_print'], msum_print], axis=0)
        data['mcat'] = pd.concat(objs=[data['mcat'], mcat], axis=0)
        data['mcat_print'] = pd.concat(objs=[data['mcat_print'], mcat_print], axis=0)

    for key, d in data.items():
        d['split'] = split
        d['brc'] = brc
        d = d[['brc', 'split', 'report_type'] + [c for c in d.columns if c not in ['brc', 'split', 'report_type']]]
        d = d.loc[~d.tnm_cat.str.endswith('_min')]  # Drop results for detecting min value as there are too little
        data[key] = d
    
    # Error analysis
    cols_compare = cols_max
    cols_add = ['brc', 'report_type', 'report_text_anon']
    cols_add = [c for c in cols_add if c in df0.columns]
    c0, c1 = df0[cols_compare], df1[cols_compare]
    mask = c0 != c1
    mask = mask.any(axis=1)
    c0.columns = [c + '_true' for c in c0.columns]
    c1.columns = [c + '_pred' for c in c1.columns]
    t = pd.concat(objs=[df0[cols_add], c0, c1], axis=1)  # dataframe that contains true and predicted labels
    t['row_num'] = np.arange(t.shape[0])
    err = t.loc[mask]
    
    # For T stage, gemerate outputs similar to confusion matrix
    #  confusion_tstage_sens : for each true T-stage value, what is the distribution of predicted values?
    #  confusion_tstage_ppv : for each predicted T-stage value, what is the distribution of true values?
    confusion_matrix_on_main_cats = True
    if confusion_matrix_on_main_cats:
        t.T_pred = t.T_pred.str.replace(r'(?<=\d)[A-Da-d]', '', regex=True)
        t.T_true = t.T_true.str.replace(r'(?<=\d)[A-Da-d]', '', regex=True)
        print(t.T_pred.unique(), t.T_true.unique())

    confusion_tstage_sens = t.groupby('T_true')['T_pred'].value_counts().rename('count').reset_index()
    confusion_tstage_ppv = t.groupby('T_pred')['T_true'].value_counts().rename('count').reset_index()

    confusion_matrices = {}
    report_types = t.report_type.unique()
    for report_type in report_types:
        tsub = t.loc[t.report_type == report_type]
        labels = pd.concat(objs=[tsub.T_true, tsub.T_pred], axis=0)
        labels = labels.sort_values().drop_duplicates()
        labels = [lab for lab in labels if lab != 'null'] + ['null']
        cmat = confusion_matrix(tsub.T_true, tsub.T_pred, labels=labels)
        cmat = pd.DataFrame(cmat, index=labels, columns=labels).reset_index()
        confusion_matrices[report_type] = cmat

    # Save the main summary tables
    suf = '_brc-' + brc + '_split-' + split + suffix + '.csv'
    fnames = {'msum': 'results-tnm_sum' + suf,
              'msum_print': 'results-tnm_sum-print' + suf,
              'mcat': 'results-tnm_cat' + suf,
              'mcat_print': 'results-tnm_cat-print' + suf}

    RESULTS_DIR.mkdir(exist_ok=True)
    for key, d in data.items():
        d.to_csv(RESULTS_DIR / fnames[key], index=False)

    # Save error analysis tables and confusion matrices
    err.to_csv(RESULTS_DIR / ('results-tnm_errors' + suf), index=False)
    t.to_csv(RESULTS_DIR / ('results-tnm_true-pred' + suf), index=False)
    confusion_tstage_sens.to_csv(RESULTS_DIR / ('results-tnm_confusion-t-sens' + suf), index=False)
    confusion_tstage_ppv.to_csv(RESULTS_DIR / ('results-tnm_confusion-t-ppv' + suf), index=False)
    for key, value in confusion_matrices.items():
        value.to_csv(RESULTS_DIR / ('results-tnm_confusion-t-' + key + suf), index=False)

    return data


def _prepare_data(truth_path, eval_path, cols, reportcol='report_text_anon', replace_n=False, check_label_indicator=True):

    # Read ground truth data (0), and data to be evaluated (1)
    if truth_path.suffix == '.xlsx':
        df0 = pd.read_excel(truth_path)
    elif truth_path.suffix == '.csv':
        df0 = pd.read_csv(truth_path)
    else:
        raise ValueError("file at truth_path is neither xlsx nor csv")
    
    if eval_path.suffix == '.xlsx':
        df1 = pd.read_excel(eval_path)
    elif eval_path.suffix == '.csv':
        df1 = pd.read_csv(eval_path)
    else:
        raise ValueError("file at eval_path is neither xlsx nor csv")

    if replace_n:
        df0[reportcol] = df0[reportcol].str.replace(r' <n> |\r', r'\n', regex=True)
        df1[reportcol] = df1[reportcol].str.replace(r' <n> |\r', r'\n', regex=True)
    
    df0[reportcol] = df0[reportcol].str.strip()
    df1[reportcol] = df1[reportcol].str.strip()

    # Check
    assert np.isin(cols, df0.columns).all(), "Not all columns to be evaluated are present in output dataset"
    assert np.isin(cols, df1.columns).all(), "Not all columns to be evaluated are present in ground truth dataset"
    assert np.all(df0.index == df1.index), "Indices of output and ground truth datasets do not match"
    assert all(df0[reportcol] == df1[reportcol]), "clinical reports do not match exactly in original and labelled data"
    if check_label_indicator:
        assert all(df0.labelled == 'yes'), "dbl check that all examples have been checked"

    # Ensure all columns to be compared are lowercase str
    for d in [df0, df1]:
        d[cols] = d[cols].fillna('null').astype(str)
        d[cols] = d[cols].replace({' ': np.nan}).fillna('null').astype(str)

    def _ensure_str(v):
        v = v.str.lower().copy()
        mask = v.str.contains(r'^\d\.\d$', regex=True)
        if any(mask):
            v[mask] = v[mask].astype(float).astype(int).astype(str)
        return v

    for c in cols:
        df0.loc[:, c] = _ensure_str(df0[c])
        df1.loc[:, c] = _ensure_str(df1[c])

    # Add report_type column if not present
    if 'report_type' not in df0.columns:
        warnings.warn("Adding 'report_type' column to ground truth dataset")
        df0['report_type'] = 'none'
    
    if 'report_type' not in df1.columns:
        warnings.warn("Adding 'report_type' column to outputs dataset")
        df1['report_type'] = 'none'

    return df0, df1


def _reformat_ci(a, a_ci):
    if np.isnan(a):
        return 'na'
    else:
        mu = np.round(a * 100, 1)
        low = np.round(a_ci[0] * 100, 1)
        high = np.round(a_ci[1] * 100, 1)
        out = "{:.1f} ({:.1f}, {:.1f})".format(mu, low, high)
        return out


def _binomial_ci_gauss(p_hat, n_obs):
    se = np.sqrt(p_hat * (1 - p_hat) / n_obs)
    ci = [p_hat - 1.96 * se, p_hat + 1.96 * se]
    return se, ci


def diagnostic_metrics_vs_null(y0: pd.Series, y1: pd.Series, index_value: str = None, index_name: str = None, 
                               ci_method: str = 'wilson'):
    """Compute common diagnostic metrics (PPV, NPV, sensitivity, specificity) against a 'null' category,
    potentially in a multi-class setting.

    args
        y0 : series that contains true values from 2 or more classes, formatted as string;
             at least one class must be labelled as 'null'.
        y1 : series that contains predicted values from 2 or more classes, formatted as string.
        index_value, index_name : value that is set as the index of output dataframe, name of the index
        ci_method : method of computing proportion confidence intervals, see statsmodels.stats.proportion.proportion_confint.
                    if None, standard error based on Gaussian approximation is computed.
    
    The meaning of computed metrics is ...
        PPV : proportion of correct predictions among non-null predictions
        NPV : proportion of null values among values predited to be null
        sensitivity : proportion of non-null values that were correctly detected
        specificity : proportion of null values that were correctly marked as null
    """

    # PPV
    mask_pred_correct = y0 == y1  # where predictions are correct
    mask_pred_notnull = y1 != 'null'  # where predictions are not null
    true_pos = mask_pred_correct[mask_pred_notnull].sum()
    pred_pos = mask_pred_notnull.sum()
    ppv = true_pos / pred_pos
    if ci_method is not None:
        ppv_ci = proportion_confint(count=true_pos, nobs=pred_pos, alpha=0.05, method=ci_method)
        ppv_se = None
    else:
        ppv_se, ppv_ci = _binomial_ci_gauss(ppv, pred_pos)
    ppv_print = _reformat_ci(ppv, ppv_ci)

    # NPV
    mask_null = y0 == 'null'  # where true values are null
    mask_pred_null = y1 == 'null'  # where predictions are null
    true_neg = mask_null[mask_pred_null].sum()
    pred_neg = mask_pred_null.sum()
    npv = true_neg / pred_neg
    if ci_method is not None:
        npv_ci = proportion_confint(count=true_neg, nobs=pred_neg, alpha=0.05, method=ci_method)
        npv_se = None
    else:
        npv_se, npv_ci = _binomial_ci_gauss(npv, pred_neg)
    npv_print = _reformat_ci(npv, npv_ci)

    # Sensitivity
    mask_notnull = y0 != 'null'  # where true values are not null
    pos = mask_notnull.sum()
    sens = true_pos / pos
    if ci_method is not None:
        sens_ci = proportion_confint(count=true_pos, nobs=pos, alpha=0.05, method=ci_method)
        sens_se = None
    else:
        sens_se, sens_ci = _binomial_ci_gauss(sens, pos)
    sens_print = _reformat_ci(sens, sens_ci)

    # Specificity
    neg = mask_null.sum()
    spec = true_neg / neg
    if ci_method is not None:
        spec_ci = proportion_confint(count=true_neg, nobs=neg, alpha=0.05, method=ci_method)
        spec_se = None
    else:
        spec_se, spec_ci = _binomial_ci_gauss(spec, neg)
    spec_print = _reformat_ci(spec, spec_ci)

    # To dataframe 
    r = {'n': [len(y0)], 'n_notnull': [pos], 'n_null': [neg], 
         'pred_correct_notnull': [true_pos], 'pred_notnull': [pred_pos], 'pred_correct_null': [true_neg], 'pred_null': [pred_neg],
         'ppv': [ppv], 'ppv_low': [ppv_ci[0]], 'ppv_high': [ppv_ci[1]], 'ppv_se': [ppv_se],
         'npv': [npv], 'npv_low': [npv_ci[0]], 'npv_high': [npv_ci[1]], 'npv_se': [npv_se],
         'sens': [sens], 'sens_low': [sens_ci[0]], 'sens_high': [sens_ci[1]], 'sens_se': [sens_se],
         'spec': [spec], 'spec_low': [spec_ci[0]], 'spec_high': [spec_ci[1]], 'spec_se': [spec_se]
         }
    r = pd.DataFrame.from_dict(r)
    if index_value is not None:
        r.index = [index_value]
        r.index.name = index_name

    # To reformatted dataframe, e.g. for publication
    p = {'n': [len(y0)], 'n_notnull': [pos], 
         'ppv': [ppv_print], 'npv': [npv_print], 'sens': [sens_print], 'spec': [spec_print]}
    p = pd.DataFrame.from_dict(p)
    if index_value is not None:
        p.index = [index_value]
        p.index.name = index_name

    return r, p


def compute_metrics(df0: pd.DataFrame, df1: pd.DataFrame, cols: list, ci_method: str = 'wilson'):

    # ---- Compute overall metrics ----
    # NB. These are computed for TNM values with subcategories, 
    # e.g. if true value is 2a but model predicts 2, it is counted as wrong
    metrics_sum = pd.DataFrame()
    metrics_sum_print = pd.DataFrame()
    for c in cols:
        y0 = df0[c]
        y1 = df1[c]
        r, p = diagnostic_metrics_vs_null(y0, y1, index_value=c, index_name='tnm_cat', ci_method=ci_method)
        metrics_sum = pd.concat(objs=[metrics_sum, r], axis=0)
        metrics_sum_print = pd.concat(objs=[metrics_sum_print, p], axis=0)
    metrics_sum = metrics_sum.reset_index()
    metrics_sum_print = metrics_sum_print.reset_index()

    # ---- Compute metrics for each TNM value, and balanced metrics as the average over these----
    metrics_cat = pd.DataFrame()
    metrics_cat_print = pd.DataFrame()
    for c in cols:
        y0 = df0[c]
        y1 = df1[c]

        y0_repl = y0.str.replace('(?<=\d)[a-d]', '', regex=True)
        y1_repl = y1.str.replace('(?<=\d)[a-d]', '', regex=True)

        values = y0_repl.loc[y0_repl != 'null'].sort_values().unique()
        if any(values):
            for value in values:

                t0 = y0_repl.copy()
                t0[t0 != value] = 'null'
                t1 = y1_repl.copy()
                t1[t1 != value] = 'null'

                r, p = diagnostic_metrics_vs_null(t0, t1, index_value=c, index_name='tnm_cat', ci_method=ci_method)
                r['tnm_value'] = value
                p['tnm_value'] = value
                metrics_cat = pd.concat(objs=[metrics_cat, r], axis=0)
                metrics_cat_print = pd.concat(objs=[metrics_cat_print, p], axis=0)

    metrics_cat = metrics_cat.reset_index()
    metrics_cat = metrics_cat[['tnm_cat', 'tnm_value'] + [c for c in metrics_cat.columns if c not in ['tnm_cat', 'tnm_value']]]
    metrics_cat_print = metrics_cat_print.reset_index()
    metrics_cat_print = metrics_cat_print[['tnm_cat', 'tnm_value'] + [c for c in metrics_cat_print.columns if c not in ['tnm_cat', 'tnm_value']]]

    metrics_bal = metrics_cat.groupby(['tnm_cat'], sort=False)[['ppv', 'npv', 'sens', 'spec']].mean()
    metrics_bal.columns = [c + '_bal' for c in metrics_bal.columns]
    metrics_bal *= 100
    metrics_bal = metrics_bal.round(1).reset_index()

    metrics_sum = metrics_sum.merge(metrics_bal, how='left')
    metrics_sum_print = metrics_sum_print.merge(metrics_bal.astype(str), how='left').fillna('na')

    return metrics_sum, metrics_sum_print, metrics_cat, metrics_cat_print

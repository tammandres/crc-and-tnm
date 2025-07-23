# Check date ranges of the data for better reporting
# Andres Tamm, 2025-03-27
import pandas as pd
from pathlib import Path
import numpy as np

data_path = Path(r'Z:\Andres\project_textmining\textmining\labelled_data')

# Training data
files = ['reports-all_crc-true_tnm-true_recur-true_crmemvi-true.csv',
         'reports-ouhfuture_crc-true_tnm-true_recur-true_crmemvi-true.csv']
dates0 = pd.DataFrame()
for f in files:
    print(f)
    df = pd.read_csv(data_path / f)
    df.report_date = pd.to_datetime(df.report_date, format='%Y-%m-%d')
    s = df.groupby(['brc', 'report_type'])[['report_date']].agg(['min', 'max'])
    s.columns = ['report_date_min', 'report_date_max']
    s = s.reset_index()
    s['file'] = f
    dates0 = pd.concat(objs=[dates0, s], axis=0)
print(dates0)

# Files selected for evaluation 
files = ['set1_crc.csv', 'set1_tnm.csv', 'set2_crc.csv', 'set2_tnm.csv']
dates = pd.DataFrame()
for f in files:
    print(f)
    df = pd.read_csv(data_path / f)
    df.report_date = pd.to_datetime(df.report_date, format='%Y-%m-%d')
    s = df.groupby('brc')['report_date'].agg(['min', 'max'])
    s.columns = ['report_date_min', 'report_date_max']
    s = s.reset_index()
    s['file'] = f
    dates = pd.concat(objs=[dates, s], axis=0)
print(dates)

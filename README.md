# Extracting colorectal cancer status and explicit TNM staging from free text reports

This code repository accompanies the publication `Supporting cancer research on real-world data: Extracting colorectal cancer status and explicitly written TNM stages from free-text imaging and histopathology reports` by Tamm, Jones, Doshi, Perry, and others.

The purpose is to provide a lightweight regex-based tool to identify clinical reports that discuss current colorectal cancer (CRC) and to extract explicitly given TNM staging scores from the reports.

A license for using the code is not yet included. A permissive license will be included once the manuscript is published.

The release `manuscript` (https://github.com/tammandres/crc-and-tnm/releases/tag/manuscript) is a snapshot of the code that was run when evaluating the algorithms for the manuscript. The current release differs slightly, including a few small updates to the algorithms, such as a few additions to the regex patterns and ability to run the algorithms on multiple cores.

Andres Tamm
2025-07-23


## Installation 
```
conda create -n textmining python=3.9
conda activate textmining
pip install -e .
```


## Usage 

* The function `textmining.reports.get_crc_reports` can be used to identify clinical reports that discuss current colorectal cancer. 
* The functions `textmining.tnm.tnm.get_tnm_phrase` and `textmining.tnm.tnm.get_tnm_values` can be used to extract the TNM scores given in letters and numbers. The `get_tnm_phrase` extracts all phrases that contain TNM scores, and the `get_tnm_values` extracts the maximum and minimum values for each TNM category from these phrases.

See `run_tnm.ipynb` and `run_tnm.py` in `./examples` for a thorough example. 

A brief example code is provided below:
```python
# Find reports that describe current colorectal cancer
df_crc, matches_crc = get_crc_reports(df, col='report_text_anon')

# Extract TNM phrases
matches, check_phrases, check_cleaning, check_rm = get_tnm_phrase(df=df_crc, col='report_text_anon', 
                                                                  remove_unusual=True, 
                                                                  remove_historical=False, 
                                                                  remove_falsepos=True)

# Extract TNM values from phrases
df_crc, s = get_tnm_values(df_crc, matches=matches, col='report_text_anon')
```

The functions can be run from the command line - results will be saved to `./results`.
But the command line version has not been updated to run on multiple cores.
```bash
# Command line help
textmining --help
textmining tnmphrase --help
textmining tnmvalues --help

# Extract TNM phrases (set --remove_previous 1 if want to remove TNM phrases that may be historical)
textmining tnmphrase --data ./tests/test_cli/reports.csv --column report_text_anon --remove_historical 0

# Extract TNM values from phrases (set additional_output 0 if do not want to include less important additional outputs)
textmining tnmvalues --data ./tests/test_cli/reports.csv --column report_text_anon --additional_output 0
```

There is also an R Shiny app for labelling reports for validation, in `./labeller`. This is separate from the
main Python package, and requires additional R libraries to be usable.


## Overview of included files 


./bin


./labeller

* `labeller_crc.R` - R Shiny app used for annotating clinical reports for CRC status

* `labeller_tnm.R` - R Shiny app used for annotating clinical reports for TNM staging


./examples

* `select_reports_for_labelling_set1.ipynb` - code that was used to select clinical reports for evaluation from the training data reports

* `select_reports_for_labelling_set2.ipynb` - code that was used to select clinical reports for evaluation from the future OUH test data reports


./notebooks

* `select_reports_for_labelling_set1.ipynb` - code that was used to select clinical reports for evaluation from the training data reports

* `select_reports_for_labelling_set2.ipynb` - code that was used to select clinical reports for evaluation from the future OUH test data reports


./runs/tnm_paper_20250703/crc - code that was run to evaluate the CRC detection algorithm for the manuscript

* `20250707_crc_check_all_labels_and_fix_genetest.py` - review how reports used for evaluating the CRC algorithm were initially annotated and then the annotations updated; label all supplementary gene testing reports as being colorectal cancer reports (previously some supplementary gene testing reports were missed). 

* `20240817_crc_evaluation_rerun.py` - run the CRC algorithm and analyse errors (produces results in the original manuscript)

* `20250715_crc_evaluation_rerun-with-correct-genetest.py` - run the CRC algorithm on corrected reports. Reports were corrected, such that all supplementary gene testing reports were now identified and labelled as being colorectal cancer reports, and a single report was excluded where the CRC status could not have been ascertained due to redaction.


./runs/tnm_paper_20250703/tnm - code that was run to evaluate the TNM stage extraction algorithm for the manuscript

* `20240214_tnm_evaluation_set2.py` - run the TNM algorithm on annotated subset of test data, and analyse errors

* `20241004_tnm_evaluation_set1_part3.py` - run the TNM algorithm on annotated subset of training data, and analyse errors (produces results in the manuscript)

* `20241004_tnm_evaluation_set2_fixnocrc.py` - run the TNM algorithm again on annotated subset of test data (produces results in the original manuscript). In this run, three reports that contained non-CRC tumours were assigned the staging 'null', which would count the extracted staging as erroneous. This is because the aim of the algorithm was to find TNM staging for CRC tumours. The TNM algorithm was also later evaluated for extracting all TNM stages regardless of whether they were historical or belonging to non-CRC tumours (see a note in the script).


./runs/

* `check_date_ranges.py` - check date ranges of data for the manuscript


./tests - various test to help ensure that the algorithms work as intended


./textmining/tnm - code for extracting TNM staging

* `clean.py` - code for filtering and cleaning extracted TNM phrases

* `extract.py` - code for extracting TNM staging phrases

* `pattern.py` - building block patterns for extracting TNM values and phrases

* `tnm.py` - complete functions for extracting the TNM phrases and TNM values from a clinical report


./textmining/vocab - some patterns and keywords used by the algorithms

* `context_20230206.csv` - patterns for identifying the context of extracted tumour keywords (e.g. negated, historical), based on a 20230206 commit.

* `context.csv` - patterns for identifying the context of extracted tumour keywords (e.g. negated, historical), more recent than the 20230206 commit.

* `context_tnm.csv` - patterns for classifying single values that resemble TNM scores (e.g. whether they are historical or general)

* `NIHR-HIC_Colorectal-Cancer_imaging-types.xlsx` - list of imaging codes that are relevant for investigating CRC, i.e. only clinical imaging reports that are assigned one of these codes would be included in analysis.

* `rules_tnm.csv` - patterns for classifying single values that resemble TNM scores (e.g. whether they are true or false positive)

* `spellcheck.csv` - for each word in column 'repl', the column 'pat' contains some spelling errors. This is used by reports.py to pick up some potential spelling errors for keywords related to tumours and colorectal cancer.

* `vocab_site_and_tumour_20230206.csv` - keywords for extracting colorectal tumours and anatomical sites, based on 20230206 commit. Note that these keywords are transformed into regex patterns based on the information given in the csv, e.g. if the pat_type is 'wordstart' it means that the keyword is meant to be preceded by nonword characters but extended by word characters.

* `vocab_site_and_tumour.csv` - keywords for extracting colorectal tumours and anatomical sites, updated compared to the 20230206 commit


./textmining/

* `cli.py` - command line interface for the TNM stage extraction algorithm

* `constants.py` - assigns file paths

* `evaluate.py` - scripts used for computing performance metrics for the CRC and TNM algorithms

* `perineural.py` - script for extracting perineural invasion not given in letters and numbers

* `reports.py` - code for identifying clinical reports that discuss current CRC

* `spelling.py` - scripts that help with spell checking

* `utils.py` - various helper functions
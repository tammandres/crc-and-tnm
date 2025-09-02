# Extracting colorectal cancer status and explicit TNM staging from free text reports

This code repository accompanies the publication `Supporting cancer research on real-world data: Extracting colorectal cancer status and explicitly written TNM stages from free-text imaging and histopathology reports` by Tamm, Jones, Doshi, Perry, and others.

The purpose is to provide a lightweight regex-based tool to identify clinical reports that discuss current colorectal cancer (CRC) and to extract explicitly given TNM staging scores from the reports.

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

See `run_tnm.ipynb` and `run_tnm.py` in `./examples` for a thorough example, especially https://github.com/tammandres/crc-and-tnm/blob/main/examples/run_tnm.ipynb.

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

See `./files.md` for a more detailed overview of files.

./examples : example scripts for how the code can be run

./labeller : R Shiny apps used for annotating clinical reports

./notebooks : code that was used to select clinical reports for evaluation

./runs : code that was run to evaluate the algorithms for the manuscript

./tests : various test to help ensure that the algorithms work as intended

./textmining/tnm : code for extracting TNM staging

./textmining/vocab : some patterns and keywords used by the algorithms

./textmining : other scripts, including code for identifying reports that describe current CRC

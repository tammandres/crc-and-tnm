"""Note: cli cannot take True/False as input values, hence using 1/0 instead."""
from textmining.constants import RESULTS_DIR
from textmining.tnm.tnm import get_tnm_phrase, get_tnm_values
import pandas as pd
import typer

main = typer.Typer()


@main.command()
def tnmphrase(
    data_path: str = typer.Option(
        ..., "-d", "--data", help="Path to data csv, e.g. './data/reports.csv', must have comma as separator"
    ),
    col: str = typer.Option(
        ..., "-c", "--column", help="Name of column in 'reports.csv' that contains reports"
    ),
    simplicity: int = typer.Option(
        2, "-s", "--simplicity",
        help="Simplicity of regex patterns, in [0, 1, 2]: 2 - least simple, 1 - intermediate, 0 - most simple. See textmining.tnm.tnm.get_tnm_phrase"
    ),
    remove_flex: int = typer.Option(
        1, "-rf", "--remove_flex",
        help="If 1, remove too flexible TNM phrases from output. If 0, don't. Only relevant if flex_start=1"
        ),
    remove_falsepos: int = typer.Option(
        0, "-rs", "--remove_falsepos", help="If 1, remove single TNM phrases that have unwanted keywords nearby. If 0, don't."
        ),
    remove_unusual: int = typer.Option(
        1, "-ru", "--remove_unusual", help="If 1, remove unusual TNM phrases from output. If 0, don't."
    ),
    remove_historical: int = typer.Option(
        0, "-rh", "--remove_historical", help="If 1, remove historical TNM phrases from output. If 0, don't."
        ),
    flex_start: int = typer.Option(
        0, "-f", "--flex_start", help="If 1, extract matches using a flexible regex first. If 0, don't."
    )
):
    """Extract TNM phrases from reports"""

    # Boolean args
    rm_un = True if remove_unusual == 1 else False
    rm_hist = True if remove_historical == 1 else False
    rm_flex = True if remove_flex == 1 else False
    rm_fp = True if remove_falsepos == 1 else False
    f = True if flex_start == 1 else False

    # df = pd.read_csv(data_path)
    df = pd.read_csv(data_path, engine='c', sep=',', lineterminator='\n')
    matches, check_phrases, check_cleaning, check_rm = get_tnm_phrase(df, col, simplicity=simplicity,
                                                                      remove_unusual=rm_un, remove_historical=rm_hist,
                                                                      remove_flex=rm_flex, remove_falsepos=rm_fp,
                                                                      flex_start=f)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    names = ['matches_tnm', 'check_phrases_tnm', 'check_cleaning_tnm', 'check_rm_tnm']
    for dataframe, name in zip([matches, check_phrases, check_cleaning, check_rm], names):
        dataframe.to_csv(RESULTS_DIR / (name + '.csv'), index=False)


@main.command()
def tnmvalues(
    data_path: str = typer.Option(
            ..., '-d', '--data', help="Path to data csv, e.g. './data/reports.csv', must have comma as separator"
        ),
    col: str = typer.Option(
        ..., '-c', '--column', help="Name of column in 'reports.csv' that contains reports"
    ),
    additional_output: int = typer.Option(
        0, "-a", "--additional_output",
        help="If 1, add additional output: indecision indicators, y/r prefix indicators. If 0, don't."
    )
):
    """Extract TNM values from matches. Assumes 'matches_tnm.csv' is in ./results.
    Command tnm_phrase must be run first to generate 'matches_tnm.csv'
    """
    # Boolean args
    a = False if additional_output == 0 else 1

    # df = pd.read_csv(data_path)
    df = pd.read_csv(data_path, engine='c', sep=',', lineterminator='\n')
    matches = pd.read_csv(RESULTS_DIR / 'matches_tnm.csv', engine='c', sep=',', lineterminator='\n')
    df, submatches = get_tnm_values(df, matches, col, additional_output=a)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    names = ['reports_with_tnm', 'check_values_tnm']
    for dataframe, name in zip([df, submatches], names):
        dataframe.to_csv(RESULTS_DIR / (name + '.csv'), index=False)

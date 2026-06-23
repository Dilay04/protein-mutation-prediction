import numpy as np
import pandas as pd

GRANTHAM_MATRIX = {
    ("A", "A"): 0,
    ("A", "R"): 112,
    ("A", "N"): 111,
    ("A", "D"): 126,
    ("A", "C"): 195,
    ("A", "Q"): 91,
    ("A", "E"): 107,
    ("A", "G"): 60,
    ("A", "H"): 86,
    ("A", "I"): 94,
    ("A", "L"): 96,
    ("A", "K"): 106,
    ("A", "M"): 84,
    ("A", "F"): 113,
    ("A", "P"): 27,
    ("A", "S"): 99,
    ("A", "T"): 58,
    ("A", "W"): 148,
    ("A", "Y"): 112,
    ("A", "V"): 64,

    ("R", "R"): 0,
    ("R", "N"): 86,
    ("R", "D"): 96,
    ("R", "C"): 180,
    ("R", "Q"): 43,
    ("R", "E"): 54,
    ("R", "G"): 125,
    ("R", "H"): 29,
    ("R", "I"): 97,
    ("R", "L"): 102,
    ("R", "K"): 26,
    ("R", "M"): 91,
    ("R", "F"): 97,
    ("R", "P"): 103,
    ("R", "S"): 110,
    ("R", "T"): 71,
    ("R", "W"): 101,
    ("R", "Y"): 77,
    ("R", "V"): 96,
}


def get_grantham_score(aa1, aa2):
    if pd.isna(aa1) or pd.isna(aa2):
        return np.nan

    aa1 = str(aa1).upper()
    aa2 = str(aa2).upper()

    return GRANTHAM_MATRIX.get(
        (aa1, aa2),
        GRANTHAM_MATRIX.get((aa2, aa1), np.nan)
    )


def add_grantham_feature(df, aa1_col="AA_1", aa2_col="AA_2"):
    df = df.copy()
    df["Grantham_score"] = df.apply(
        lambda row: get_grantham_score(row[aa1_col], row[aa2_col]),
        axis=1
    )
    return df 

BLOSUM62_MATRIX = {
    ("A", "A"): 4,
    ("A", "R"): -1,
    ("A", "N"): -2,
    ("A", "D"): -2,
    ("A", "C"): 0,
    ("A", "Q"): -1,
    ("A", "E"): -1,
    ("A", "G"): 0,
    ("A", "H"): -2,
    ("A", "I"): -1,
    ("A", "L"): -1,
    ("A", "K"): -1,
    ("A", "M"): -1,
    ("A", "F"): -2,
    ("A", "P"): -1,
    ("A", "S"): 1,
    ("A", "T"): 0,
    ("A", "W"): -3,
    ("A", "Y"): -2,
    ("A", "V"): 0,
}


def get_blosum62_score(aa1, aa2):
    if pd.isna(aa1) or pd.isna(aa2):
        return np.nan

    aa1 = str(aa1).upper()
    aa2 = str(aa2).upper()

    return BLOSUM62_MATRIX.get(
        (aa1, aa2),
        BLOSUM62_MATRIX.get((aa2, aa1), np.nan)
    )


def add_blosum62_feature(df, aa1_col="AA_1", aa2_col="AA_2"):
    df = df.copy()
    df["BLOSUM62_score"] = df.apply(
        lambda row: get_blosum62_score(row[aa1_col], row[aa2_col]),
        axis=1
    )
    return df
def add_aa_features(df, aa1_col="AA_1", aa2_col="AA_2"):
    df = df.copy()

    df["Grantham_score"] = df.apply(
        lambda row: get_grantham_score(row[aa1_col], row[aa2_col]),
        axis=1
    )

    df["BLOSUM62_score"] = df.apply(
        lambda row: get_blosum62_score(row[aa1_col], row[aa2_col]),
        axis=1
    )

    return df

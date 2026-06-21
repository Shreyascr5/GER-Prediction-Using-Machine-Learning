import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import (
    shapiro,
    skew,
    kurtosis,
    probplot,
    pearsonr,
    spearmanr
)

warnings.filterwarnings("ignore")

# ==================================================
# CONFIG
# ==================================================

DATA_PATH = "data/AISHE_GRE_Merged_2018_2022_imputed.csv"

OUTPUT_DIR = "week4_statistics/outputs"

TARGET = "ger_all_total"

# ==================================================
# CREATE FOLDERS
# ==================================================

folders = [
    "distributions/histograms",
    "distributions/kde",
    "distributions/qqplots",
    "normality",
    "hypothesis_tests",
    "reports"
]

for folder in folders:
    os.makedirs(
        os.path.join(OUTPUT_DIR, folder),
        exist_ok=True
    )

print("Output folders created")

# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(DATA_PATH)

print("\nDataset Shape:", df.shape)

# ==================================================
# NUMERICAL FEATURES
# ==================================================

numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

print("Numerical Features:", len(numeric_cols))

# ==================================================
# DISTRIBUTION ANALYSIS
# ==================================================

distribution_results = []

print("\nRunning Distribution Analysis...")

for col in numeric_cols:

    data = df[col].dropna()

    sk = skew(data)
    ku = kurtosis(data)

    # Distribution Classification

    if -0.5 <= sk <= 0.5:
        dist_type = "Approximately Normal"

    elif 0.5 < sk <= 1:
        dist_type = "Moderately Right Skewed"

    elif sk > 1:
        dist_type = "Highly Right Skewed"

    elif -1 <= sk < -0.5:
        dist_type = "Moderately Left Skewed"

    else:
        dist_type = "Highly Left Skewed"

    distribution_results.append([
        col,
        round(sk, 4),
        round(ku, 4),
        dist_type
    ])

    # Histogram

    plt.figure(figsize=(7,4))

    sns.histplot(
        data,
        kde=True
    )

    plt.title(col)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/distributions/histograms/{col}.png"
    )

    plt.close()

    # KDE

    plt.figure(figsize=(7,4))

    sns.kdeplot(
        data,
        fill=True
    )

    plt.title(f"KDE - {col}")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/distributions/kde/{col}.png"
    )

    plt.close()

    # QQ Plot

    plt.figure(figsize=(6,6))

    probplot(
        data,
        dist="norm",
        plot=plt
    )

    plt.title(f"QQ Plot - {col}")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/distributions/qqplots/{col}.png"
    )

    plt.close()

distribution_df = pd.DataFrame(
    distribution_results,
    columns=[
        "Feature",
        "Skewness",
        "Kurtosis",
        "Distribution_Type"
    ]
)

distribution_df.to_csv(
    f"{OUTPUT_DIR}/reports/distribution_summary.csv",
    index=False
)

# ==================================================
# NORMALITY TESTING
# ==================================================

print("\nRunning Shapiro-Wilk Tests...")

normality_results = []

for col in numeric_cols:

    data = df[col].dropna()

    if len(data) > 3:

        stat, p = shapiro(data)

        normality = (
            "Normal"
            if p > 0.05
            else "Not Normal"
        )

        normality_results.append([
            col,
            stat,
            p,
            normality
        ])

normality_df = pd.DataFrame(
    normality_results,
    columns=[
        "Feature",
        "Statistic",
        "P_Value",
        "Normality"
    ]
)

normality_df.to_csv(
    f"{OUTPUT_DIR}/normality/normality_results.csv",
    index=False
)

# ==================================================
# HYPOTHESIS TESTING
# ==================================================

print("\nRunning Hypothesis Tests...")

normality_lookup = dict(
    zip(
        normality_df["Feature"],
        normality_df["Normality"]
    )
)

hypotheses = [

    (
        "Teacher Availability influences GER",
        "teacher_total",
        TARGET
    ),

    (
        "Universities influence GER",
        "total_universities",
        TARGET
    ),

    (
        "Enrollment influences GER",
        "enr_grand_total",
        TARGET
    ),

    (
        "Gender Parity influences GER",
        "gpi_all",
        TARGET
    )
]

hypothesis_results = []

method_results = []

for name, feature, target in hypotheses:

    x = df[feature]
    y = df[target]

    x_normal = (
        normality_lookup.get(feature)
        == "Normal"
    )

    y_normal = (
        normality_lookup.get(target)
        == "Normal"
    )

    if x_normal and y_normal:

        method = "Pearson"

        stat, p = pearsonr(
            x,
            y
        )

    else:

        method = "Spearman"

        stat, p = spearmanr(
            x,
            y
        )

    decision = (
        "Reject H0"
        if p < 0.05
        else "Fail to Reject H0"
    )

    hypothesis_results.append([
        name,
        feature,
        method,
        stat,
        p,
        decision
    ])

    method_results.append([
        feature,
        method
    ])

hypothesis_df = pd.DataFrame(
    hypothesis_results,
    columns=[
        "Hypothesis",
        "Feature",
        "Method",
        "Statistic",
        "P_Value",
        "Decision"
    ]
)

hypothesis_df.to_csv(
    f"{OUTPUT_DIR}/hypothesis_tests/hypothesis_results.csv",
    index=False
)

method_df = pd.DataFrame(
    method_results,
    columns=[
        "Feature",
        "Method_Selected"
    ]
)

method_df.to_csv(
    f"{OUTPUT_DIR}/hypothesis_tests/correlation_method_selection.csv",
    index=False
)

# ==================================================
# SUMMARY
# ==================================================

print("\n" + "="*60)
print("WEEK 4 COMPLETED")
print("="*60)

print("\nGenerated:")

print("Distribution Summary")
print("Normality Results")
print("Hypothesis Results")
print("Correlation Method Selection")

print("\nOutputs saved to:")
print(OUTPUT_DIR)
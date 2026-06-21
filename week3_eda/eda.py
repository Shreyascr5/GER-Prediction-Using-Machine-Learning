import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# =====================================================
# CONFIG
# =====================================================

DATA_PATH = "data/AISHE_GRE_Merged_2018_2022_imputed.csv"

OUTPUT_DIR = "week3_eda/outputs"

TARGET = "ger_all_total"

# =====================================================
# CREATE OUTPUT FOLDERS
# =====================================================

folders = [
    "histograms",
    "boxplots",
    "target_analysis",
    "correlation",
    "reports"
]

for folder in folders:
    os.makedirs(
        os.path.join(OUTPUT_DIR, folder),
        exist_ok=True
    )

print("Output folders ready")

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded")
print("Shape:", df.shape)

# =====================================================
# NUMERICAL FEATURES
# =====================================================

numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

print("\nNumerical Features:", len(numeric_cols))

# =====================================================
# TARGET ANALYSIS
# =====================================================

print("\nGenerating Target Variable Analysis...")

# Histogram

plt.figure(figsize=(8,5))
sns.histplot(
    df[TARGET],
    kde=True
)
plt.title("GER Distribution")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/target_analysis/ger_histogram.png"
)

plt.close()

# KDE

plt.figure(figsize=(8,5))
sns.kdeplot(
    df[TARGET],
    fill=True
)
plt.title("GER KDE Plot")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/target_analysis/ger_kde.png"
)

plt.close()

# Boxplot

plt.figure(figsize=(8,4))
sns.boxplot(
    x=df[TARGET]
)
plt.title("GER Boxplot")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/target_analysis/ger_boxplot.png"
)

plt.close()

# Violin

plt.figure(figsize=(8,4))
sns.violinplot(
    x=df[TARGET]
)
plt.title("GER Violin Plot")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/target_analysis/ger_violinplot.png"
)

plt.close()

# =====================================================
# HISTOGRAMS
# =====================================================

print("\nGenerating Histograms...")

for col in numeric_cols:

    plt.figure(figsize=(7,4))

    sns.histplot(
        df[col],
        kde=True
    )

    plt.title(col)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/histograms/{col}.png"
    )

    plt.close()

# =====================================================
# BOXPLOTS
# =====================================================

print("Generating Boxplots...")

for col in numeric_cols:

    plt.figure(figsize=(7,4))

    sns.boxplot(
        x=df[col]
    )

    plt.title(col)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/boxplots/{col}.png"
    )

    plt.close()

# =====================================================
# CORRELATION MATRIX
# =====================================================

print("\nGenerating Correlation Analysis...")

corr_matrix = df[numeric_cols].corr()

corr_matrix.to_csv(
    f"{OUTPUT_DIR}/correlation/correlation_matrix.csv"
)

plt.figure(figsize=(20,16))

sns.heatmap(
    corr_matrix,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/correlation/correlation_heatmap.png"
)

plt.close()

# =====================================================
# GER CORRELATIONS
# =====================================================

ger_corr = (
    corr_matrix[TARGET]
    .sort_values(
        ascending=False
    )
)

ger_corr_df = pd.DataFrame({
    "Feature": ger_corr.index,
    "Correlation_With_GER": ger_corr.values
})

ger_corr_df.to_csv(
    f"{OUTPUT_DIR}/correlation/ger_correlations.csv",
    index=False
)

# Top Positive

positive = (
    ger_corr_df[
        ger_corr_df["Feature"] != TARGET
    ]
    .head(10)
)

positive.to_csv(
    f"{OUTPUT_DIR}/correlation/top10_positive_ger_predictors.csv",
    index=False
)

# Top Negative

negative = (
    ger_corr_df.sort_values(
        by="Correlation_With_GER"
    )
    .head(10)
)

negative.to_csv(
    f"{OUTPUT_DIR}/correlation/top10_negative_ger_predictors.csv",
    index=False
)

# =====================================================
# OUTLIER ANALYSIS
# =====================================================

print("Detecting Outliers...")

outlier_results = []

for col in numeric_cols:

    q1 = df[col].quantile(0.25)

    q3 = df[col].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr

    upper = q3 + 1.5 * iqr

    count = (
        (
            (df[col] < lower)
            |
            (df[col] > upper)
        )
    ).sum()

    outlier_results.append([
        col,
        count
    ])

outlier_df = pd.DataFrame(
    outlier_results,
    columns=[
        "Feature",
        "Outlier_Count"
    ]
)

outlier_df = outlier_df.sort_values(
    by="Outlier_Count",
    ascending=False
)

outlier_df.to_csv(
    f"{OUTPUT_DIR}/reports/outlier_summary.csv",
    index=False
)

# =====================================================
# SUMMARY
# =====================================================

print("\n" + "="*60)
print("WEEK 3 EDA COMPLETED")
print("="*60)

print("\nOutputs Saved In:")
print(OUTPUT_DIR)

print("\nFiles Generated:")
print("- Histograms")
print("- Boxplots")
print("- Correlation Matrix")
print("- Correlation Heatmap")
print("- GER Correlations")
print("- Top Positive Predictors")
print("- Top Negative Predictors")
print("- Outlier Summary")
print("- Target Variable Analysis")
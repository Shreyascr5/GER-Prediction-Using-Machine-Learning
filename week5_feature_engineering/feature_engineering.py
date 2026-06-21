import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# =====================================================
# CONFIG
# =====================================================

DATA_PATH = "data/AISHE_GRE_Merged_2018_2022_imputed.csv"

OUTPUT_DIR = "week5_feature_engineering/outputs"

TARGET = "ger_all_total"

# =====================================================
# CREATE FOLDERS
# =====================================================

os.makedirs(
    f"{OUTPUT_DIR}/before_after_distributions",
    exist_ok=True
)

os.makedirs(
    f"{OUTPUT_DIR}/reports",
    exist_ok=True
)

print("Output folders ready")

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded")
print(df.shape)

# =====================================================
# REMOVE DATA LEAKAGE
# =====================================================

LEAKAGE_FEATURES = [

    "ger_all_male",
    "ger_all_female",

    "ger_sc_male",
    "ger_sc_female",
    "ger_sc_total",

    "ger_st_male",
    "ger_st_female",
    "ger_st_total"
]

df = df.drop(
    columns=LEAKAGE_FEATURES,
    errors="ignore"
)

print("\nLeakage Features Removed")

# =====================================================
# FEATURE ENGINEERING
# =====================================================

print("\nCreating Engineered Features")

# Log Features

df["log_enr_grand_total"] = np.log1p(
    df["enr_grand_total"]
)

df["log_enr_ug_total"] = np.log1p(
    df["enr_ug_total"]
)

df["log_enr_pg_total"] = np.log1p(
    df["enr_pg_total"]
)

df["log_teacher_total"] = np.log1p(
    df["teacher_total"]
)

df["log_total_universities"] = np.log1p(
    df["total_universities"]
)

# Ratio Features

df["student_teacher_ratio"] = (
    df["enr_grand_total"]
    /
    (df["teacher_total"] + 1)
)

df["students_per_university"] = (
    df["enr_grand_total"]
    /
    (df["total_universities"] + 1)
)

df["teachers_per_university"] = (
    df["teacher_total"]
    /
    (df["total_universities"] + 1)
)

df["female_participation_ratio"] = (
    df["enr_grand_female"]
    /
    (df["enr_grand_total"] + 1)
)

df["phd_ratio"] = (
    df["enr_phd_total"]
    /
    (df["enr_grand_total"] + 1)
)

df["pg_ratio"] = (
    df["enr_pg_total"]
    /
    (df["enr_grand_total"] + 1)
)

# =====================================================
# BEFORE VS AFTER DISTRIBUTION
# =====================================================

comparison_pairs = [

    ("enr_grand_total", "log_enr_grand_total"),

    ("teacher_total", "log_teacher_total"),

    ("total_universities",
     "log_total_universities")
]

for original, transformed in comparison_pairs:

    # Before

    plt.figure(figsize=(7,4))

    sns.histplot(
        df[original],
        kde=True
    )

    plt.title(f"Before: {original}")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/before_after_distributions/{original}_before.png"
    )

    plt.close()

    # After

    plt.figure(figsize=(7,4))

    sns.histplot(
        df[transformed],
        kde=True
    )

    plt.title(f"After: {transformed}")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/before_after_distributions/{original}_after.png"
    )

    plt.close()

# =====================================================
# CORRELATION ANALYSIS
# =====================================================

corr = (
    df.corr(numeric_only=True)[TARGET]
    .sort_values(
        ascending=False
    )
)

corr_df = pd.DataFrame({

    "Feature": corr.index,

    "Correlation_With_GER": corr.values
})

corr_df.to_csv(
    f"{OUTPUT_DIR}/feature_correlation_comparison.csv",
    index=False
)

# =====================================================
# RANDOM FOREST FEATURE IMPORTANCE
# =====================================================

print("\nCalculating Feature Importance")

numeric_df = df.select_dtypes(
    include=np.number
)

X = numeric_df.drop(
    columns=[TARGET],
    errors="ignore"
)

y = numeric_df[TARGET]

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X, y)

importance_df = pd.DataFrame({

    "Feature": X.columns,

    "Importance":
    model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
)

importance_df.to_csv(
    f"{OUTPUT_DIR}/feature_importance.csv",
    index=False
)

# =====================================================
# SAVE ENGINEERED DATASET
# =====================================================

df.to_csv(
    f"{OUTPUT_DIR}/engineered_dataset.csv",
    index=False
)

# =====================================================
# SUMMARY REPORT
# =====================================================

summary = pd.DataFrame({

    "Metric": [

        "Original Features",

        "Features After Engineering",

        "Leakage Features Removed",

        "New Features Created"
    ],

    "Value": [

        57,

        len(df.columns),

        len(LEAKAGE_FEATURES),

        11
    ]
})

summary.to_csv(
    f"{OUTPUT_DIR}/reports/feature_engineering_summary.csv",
    index=False
)

# =====================================================
# DONE
# =====================================================

print("\n" + "="*60)
print("WEEK 5 COMPLETED")
print("="*60)

print("\nGenerated:")

print("Engineered Dataset")
print("Feature Importance")
print("Feature Correlation Comparison")
print("Before vs After Distribution Plots")
print("Feature Engineering Summary")

print("\nOutputs Saved To:")
print(OUTPUT_DIR)
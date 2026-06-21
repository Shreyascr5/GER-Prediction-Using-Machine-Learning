import os
import pandas as pd

# =====================================
# CONFIG
# =====================================

DATA_PATH = "data/AISHE_GRE_Merged_2018_2022_imputed.csv"
OUTPUT_DIR = "week2_data_understanding/outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv(DATA_PATH)

print("\n" + "="*60)
print("DATASET OVERVIEW")
print("="*60)

print(f"\nShape: {df.shape}")

print("\nColumns:")
for col in df.columns:
    print(col)

# =====================================
# DATA TYPES
# =====================================

dtypes_df = pd.DataFrame({
    "Column": df.columns,
    "DataType": df.dtypes.astype(str)
})

dtypes_df.to_csv(
    f"{OUTPUT_DIR}/data_types.csv",
    index=False
)

# =====================================
# MISSING VALUES
# =====================================

missing_df = pd.DataFrame({
    "Column": df.columns,
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage":
        round((df.isnull().sum()/len(df))*100, 2)
})

missing_df.to_csv(
    f"{OUTPUT_DIR}/missing_values.csv",
    index=False
)

# =====================================
# DUPLICATES
# =====================================

duplicate_count = df.duplicated().sum()

duplicate_df = pd.DataFrame({
    "Total_Rows": [len(df)],
    "Duplicate_Rows": [duplicate_count]
})

duplicate_df.to_csv(
    f"{OUTPUT_DIR}/duplicates_summary.csv",
    index=False
)

# =====================================
# NUMERICAL FEATURES
# =====================================

numerical_cols = df.select_dtypes(
    include=["int64", "float64"]
).columns

num_df = pd.DataFrame({
    "Numerical_Features": numerical_cols
})

num_df.to_csv(
    f"{OUTPUT_DIR}/numerical_features.csv",
    index=False
)

# =====================================
# CATEGORICAL FEATURES
# =====================================

categorical_cols = df.select_dtypes(
    include=["object"]
).columns

cat_df = pd.DataFrame({
    "Categorical_Features": categorical_cols
})

cat_df.to_csv(
    f"{OUTPUT_DIR}/categorical_features.csv",
    index=False
)

# =====================================
# DESCRIPTIVE STATISTICS
# =====================================

desc_stats = df.describe().T

desc_stats.to_csv(
    f"{OUTPUT_DIR}/descriptive_statistics.csv"
)

# =====================================
# TARGET ANALYSIS
# =====================================

TARGET = "ger_all_total"

if TARGET in df.columns:

    target_summary = pd.DataFrame({
        "Metric": [
            "Mean",
            "Median",
            "Min",
            "Max",
            "Std"
        ],
        "Value": [
            df[TARGET].mean(),
            df[TARGET].median(),
            df[TARGET].min(),
            df[TARGET].max(),
            df[TARGET].std()
        ]
    })

    target_summary.to_csv(
        f"{OUTPUT_DIR}/target_summary.csv",
        index=False
    )

# =====================================
# DATASET SUMMARY
# =====================================

summary = pd.DataFrame({

    "Metric": [
        "Rows",
        "Columns",
        "Numerical Features",
        "Categorical Features",
        "Duplicate Rows"
    ],

    "Value": [
        df.shape[0],
        df.shape[1],
        len(numerical_cols),
        len(categorical_cols),
        duplicate_count
    ]
})

summary.to_csv(
    f"{OUTPUT_DIR}/dataset_summary.csv",
    index=False
)

print("\n" + "="*60)
print("WEEK 2 COMPLETED")
print("="*60)

print(f"\nOutputs saved in:\n{OUTPUT_DIR}")
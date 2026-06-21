import os
import warnings
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    learning_curve
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.ensemble import (
    RandomForestRegressor
)

from sklearn.metrics import (

    r2_score,

    mean_absolute_error,

    mean_squared_error,

    confusion_matrix,

    classification_report
)

warnings.filterwarnings("ignore")

DATA_PATH = (
    "week5_feature_engineering/outputs/"
    "engineered_dataset.csv"
)

OUTPUT_DIR = (
    "week9_final_evaluation/outputs"
)

PLOT_DIR = (
    f"{OUTPUT_DIR}/plots"
)

REPORT_DIR = (
    f"{OUTPUT_DIR}/reports"
)

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

TARGET = "ger_all_total"

print("Output folders ready")


df = pd.read_csv(DATA_PATH)

df["year"] = df["year"].astype(str)
df["state"] = df["state"].astype(str)

print("\nDataset Loaded")
print(df.shape)

X = df.drop(columns=[TARGET])

y = df[TARGET]


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.15,

    random_state=42
)
categorical_cols = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_cols = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

preprocessor = ColumnTransformer(

    transformers=[

        (
            "num",
            StandardScaler(),
            numerical_cols
        ),

        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_cols
        )
    ]
)

rf = Pipeline([

    ("preprocessor", preprocessor),

    (
        "model",
        RandomForestRegressor(

            n_estimators=300,

            max_depth=10,

            min_samples_split=2,

            min_samples_leaf=1,

            random_state=42
        )
    )
])

rf.fit(
    X_train,
    y_train
)

train_pred = rf.predict(X_train)

test_pred = rf.predict(X_test)

train_r2 = r2_score(
    y_train,
    train_pred
)

test_r2 = r2_score(
    y_test,
    test_pred
)

mae = mean_absolute_error(
    y_test,
    test_pred
)

mse = mean_squared_error(
    y_test,
    test_pred
)

rmse = np.sqrt(mse)

mape = np.mean(
    np.abs(
        (y_test - test_pred) / y_test
    )
) * 100

metrics_df = pd.DataFrame({

    "Metric": [

        "Train_R2",

        "Test_R2",

        "MAE",

        "MSE",

        "RMSE",

        "MAPE"
    ],

    "Value": [

        train_r2,

        test_r2,

        mae,

        mse,

        rmse,

        mape
    ]
})

metrics_df.to_csv(

    f"{REPORT_DIR}/final_test_metrics.csv",

    index=False
)

generalisation_error = (

    train_r2
    -
    test_r2
)

gen_df = pd.DataFrame({

    "Train_R2":
        [train_r2],

    "Test_R2":
        [test_r2],

    "Generalisation_Error":
        [generalisation_error]
})

gen_df.to_csv(

    f"{REPORT_DIR}/generalisation_error.csv",

    index=False
)

train_sizes, train_scores, val_scores = learning_curve(

    rf,

    X,
    y,

    cv=5,

    scoring="r2",

    train_sizes=np.linspace(
        0.1,
        1.0,
        10
    ),

    n_jobs=-1
)

train_mean = np.mean(
    train_scores,
    axis=1
)

val_mean = np.mean(
    val_scores,
    axis=1
)

plt.figure(figsize=(10,6))

plt.plot(

    train_sizes,

    train_mean,

    marker="o",

    label="Training Score"
)

plt.plot(

    train_sizes,

    val_mean,

    marker="o",

    label="Validation Score"
)

plt.legend()

plt.xlabel("Training Size")

plt.ylabel("R² Score")

plt.title("Learning Curve")

plt.tight_layout()

plt.savefig(

    f"{PLOT_DIR}/learning_curve.png"
)

plt.close()

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    test_pred
)

plt.plot(

    [y_test.min(), y_test.max()],

    [y_test.min(), y_test.max()],

    "r--"
)

plt.xlabel("Actual GER")

plt.ylabel("Predicted GER")

plt.title("Actual vs Predicted")

plt.tight_layout()

plt.savefig(

    f"{PLOT_DIR}/actual_vs_predicted_final.png"
)

plt.close()

residuals = y_test - test_pred

plt.figure(figsize=(8,6))

plt.scatter(
    test_pred,
    residuals
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted GER")

plt.ylabel("Residual")

plt.title("Residual Plot")

plt.tight_layout()

plt.savefig(

    f"{PLOT_DIR}/residual_plot.png"
)

plt.close()


def ger_category(x):

    if x < 20:
        return "Low"

    elif x < 35:
        return "Medium"

    else:
        return "High"
    
y_true_cat = y_test.apply(
    ger_category
)

y_pred_cat = pd.Series(
    test_pred
).apply(
    ger_category
)
cm = confusion_matrix(

    y_true_cat,

    y_pred_cat,

    labels=[
        "Low",
        "Medium",
        "High"
    ]
)
plt.figure(figsize=(6,5))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=[
        "Low",
        "Medium",
        "High"
    ],

    yticklabels=[
        "Low",
        "Medium",
        "High"
    ]
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(

    f"{PLOT_DIR}/confusion_matrix.png"
)

plt.close()

report = classification_report(

    y_true_cat,

    y_pred_cat,

    output_dict=True
)

pd.DataFrame(report).transpose().to_csv(

    f"{REPORT_DIR}/classification_report.csv"
)


print("\n" + "="*60)
print("WEEK 9 COMPLETED")
print("="*60)

print("\nGenerated")

print("final_test_metrics.csv")
print("generalisation_error.csv")
print("classification_report.csv")

print("\nPlots")

print("learning_curve.png")
print("actual_vs_predicted_final.png")
print("residual_plot.png")
print("confusion_matrix.png")
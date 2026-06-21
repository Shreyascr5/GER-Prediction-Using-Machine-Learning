import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

warnings.filterwarnings("ignore")

# ==================================================
# CONFIG
# ==================================================

DATA_PATH = (
    "week5_feature_engineering/outputs/"
    "engineered_dataset.csv"
)

OUTPUT_DIR = (
    "week6_model_comparison/outputs"
)

TARGET = "ger_all_total"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("Output folder ready")

# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded")
print(df.shape)

# ==================================================
# FEATURES / TARGET
# ==================================================

X = df.drop(columns=[TARGET])
y = df[TARGET]

categorical_cols = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_cols = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("\nCategorical Columns:")
print(categorical_cols)

print("\nNumerical Features:")
print(len(numerical_cols))

# models
models = {

    "Linear Regression":
        LinearRegression(),

    "SVR":
        SVR(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=300,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            random_state=42
        )
}

# splits

splits = {

    "70-30": 0.30,

    "75-25": 0.25,

    "80-20": 0.20,

    "85-15": 0.15
}

# scalers

scalers = {

    "StandardScaler":
        StandardScaler(),

    "MinMaxScaler":
        MinMaxScaler()
}
# result container
cv_results = []

best_k_results = []

best_split_results = []

final_results = []


# ==================================================
# EXPERIMENT ENGINE
# ==================================================

for model_name, model in models.items():

    print(f"\n{'='*60}")
    print(f"RUNNING: {model_name}")
    print(f"{'='*60}")

    global_best_score = -999

    best_split = None
    best_scaler = None
    best_k = None

    best_pipeline = None

    for split_name, test_size in splits.items():

        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size=test_size,

            random_state=42
        )

        for scaler_name, scaler in scalers.items():

            preprocessor = ColumnTransformer(

                transformers=[

                    (
                        "num",
                        scaler,
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

            pipeline = Pipeline([

                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    model
                )
            ])

            for k in range(2, 8):

                cv = KFold(

                    n_splits=k,

                    shuffle=True,

                    random_state=42
                )

                scores = cross_val_score(

                    pipeline,

                    X_train,

                    y_train,

                    cv=cv,

                    scoring="r2"
                )

                mean_r2 = scores.mean()

                std_r2 = scores.std()

                stability_score = (
                    mean_r2 - std_r2
                )

                cv_results.append([

                    model_name,

                    split_name,

                    scaler_name,

                    k,

                    mean_r2,

                    std_r2,

                    stability_score
                ])

                if stability_score > global_best_score:

                    global_best_score = stability_score

                    best_split = split_name

                    best_scaler = scaler_name

                    best_k = k

                    best_pipeline = pipeline

    # =============================================
    # TRAIN BEST CONFIGURATION
    # =============================================

    print(
        f"Best Split: {best_split}"
    )

    print(
        f"Best Scaler: {best_scaler}"
    )

    print(
        f"Best K: {best_k}"
    )

    best_test_size = splits[
        best_split
    ]

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=best_test_size,

        random_state=42
    )

    best_pipeline.fit(
        X_train,
        y_train
    )

    predictions = best_pipeline.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    mape = np.mean(
        np.abs(
            (
                y_test -
                predictions
            )
            /
            y_test
        )
    ) * 100

    final_results.append([

        model_name,

        best_split,

        best_scaler,

        best_k,

        mae,

        mse,

        rmse,

        r2,

        mape
    ])

    best_k_results.append([

        model_name,

        best_split,

        best_scaler,

        best_k,

        global_best_score
    ])

    best_split_results.append([

        model_name,

        best_split,

        best_scaler,

        best_k
    ])

    # ==================================================
# SAVE OUTPUTS
# ==================================================

cv_df = pd.DataFrame(

    cv_results,

    columns=[

        "Model",

        "Split",

        "Scaler",

        "K",

        "Mean_R2",

        "Std_R2",

        "Stability_Score"
    ]
)

cv_df.to_csv(

    f"{OUTPUT_DIR}/cross_validation_results.csv",

    index=False
)

best_k_df = pd.DataFrame(

    best_k_results,

    columns=[

        "Model",

        "Best_Split",

        "Best_Scaler",

        "Best_K",

        "Best_CV_Score"
    ]
)

best_k_df.to_csv(

    f"{OUTPUT_DIR}/best_k_selection.csv",

    index=False
)

best_split_df = pd.DataFrame(

    best_split_results,

    columns=[

        "Model",

        "Best_Split",

        "Best_Scaler",

        "Best_K"
    ]
)

best_split_df.to_csv(

    f"{OUTPUT_DIR}/best_split_selection.csv",

    index=False
)

model_df = pd.DataFrame(

    final_results,

    columns=[

        "Model",

        "Best_Split",

        "Best_Scaler",

        "Best_K",

        "MAE",

        "MSE",

        "RMSE",

        "R2",

        "MAPE"
    ]
)

model_df = model_df.sort_values(

    by="R2",

    ascending=False
)

model_df.to_csv(

    f"{OUTPUT_DIR}/model_comparison.csv",

    index=False
)

print("\n")
print("="*60)
print("WEEK 6 STAGE 1 COMPLETED")
print("="*60)

print("\nGenerated:")

print("cross_validation_results.csv")
print("best_k_selection.csv")
print("best_split_selection.csv")
print("model_comparison.csv")
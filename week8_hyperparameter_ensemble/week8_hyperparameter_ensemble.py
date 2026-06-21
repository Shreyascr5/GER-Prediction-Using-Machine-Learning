import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    VotingRegressor
)

warnings.filterwarnings("ignore")


DATA_PATH = (
    "week5_feature_engineering/outputs/"
    "engineered_dataset.csv"
)

OUTPUT_DIR = (
    "week8_hyperparameter_ensemble/outputs"
)

PLOT_DIR = (
    f"{OUTPUT_DIR}/plots"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    PLOT_DIR,
    exist_ok=True
)

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

results = []

rf = Pipeline([

    ("preprocessor", preprocessor),

    (
        "model",
        RandomForestRegressor(
            random_state=42
        )
    )
])

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_r2_before = r2_score(
    y_test,
    rf_pred
)

print(
    f"\nRF Baseline R2 = {rf_r2_before:.4f}"
)
rf_grid = {

    "model__n_estimators":
        [100,200,300],

    "model__max_depth":
        [3,5,10,None],

    "model__min_samples_split":
        [2,5,10],

    "model__min_samples_leaf":
        [1,2,4]
}

rf_grid_search = GridSearchCV(

    estimator=rf,

    param_grid=rf_grid,

    cv=5,

    scoring="r2",

    n_jobs=-1
)

rf_grid_search.fit(
    X_train,
    y_train
)

rf_grid_pred = (
    rf_grid_search.best_estimator_
    .predict(X_test)
)

rf_r2_grid = r2_score(
    y_test,
    rf_grid_pred
)

rf_random = RandomizedSearchCV(

    estimator=rf,

    param_distributions=rf_grid,

    n_iter=20,

    cv=5,

    scoring="r2",

    random_state=42,

    n_jobs=-1
)

rf_random.fit(
    X_train,
    y_train
)

rf_random_pred = (
    rf_random.best_estimator_
    .predict(X_test)
)

rf_r2_random = r2_score(
    y_test,
    rf_random_pred
)

gbr = Pipeline([

    ("preprocessor", preprocessor),

    (
        "model",
        GradientBoostingRegressor(
            random_state=42
        )
    )
])

gbr.fit(X_train, y_train)

gbr_pred = gbr.predict(X_test)

gbr_r2_before = r2_score(
    y_test,
    gbr_pred
)

gbr_grid = {

    "model__n_estimators":
        [100,200,300],

    "model__learning_rate":
        [0.01,0.05,0.1],

    "model__max_depth":
        [2,3,4],

    "model__min_samples_split":
        [2,5,10]
}
gbr_grid_search = GridSearchCV(

    estimator=gbr,

    param_grid=gbr_grid,

    cv=5,

    scoring="r2",

    n_jobs=-1
)

gbr_grid_search.fit(
    X_train,
    y_train
)

gbr_grid_pred = (
    gbr_grid_search.best_estimator_
    .predict(X_test)
)

gbr_r2_grid = r2_score(
    y_test,
    gbr_grid_pred
)

gbr_random = RandomizedSearchCV(

    estimator=gbr,

    param_distributions=gbr_grid,

    n_iter=20,

    cv=5,

    scoring="r2",

    random_state=42,

    n_jobs=-1
)

gbr_random.fit(
    X_train,
    y_train
)

gbr_random_pred = (
    gbr_random.best_estimator_
    .predict(X_test)
)

gbr_r2_random = r2_score(
    y_test,
    gbr_random_pred
)

comparison_df = pd.DataFrame({

    "Model": [

        "Random Forest",

        "Gradient Boosting"
    ],

    "Baseline_R2": [

        rf_r2_before,

        gbr_r2_before
    ],

    "GridSearch_R2": [

        rf_r2_grid,

        gbr_r2_grid
    ],

    "RandomSearch_R2": [

        rf_r2_random,

        gbr_r2_random
    ]
})

comparison_df.to_csv(

    f"{OUTPUT_DIR}/before_after_tuning.csv",

    index=False
)

params_df = pd.DataFrame({

    "Model": [

        "Random Forest Grid",

        "Random Forest Random",

        "GBR Grid",

        "GBR Random"
    ],

    "Best_Params": [

        str(
            rf_grid_search.best_params_
        ),

        str(
            rf_random.best_params_
        ),

        str(
            gbr_grid_search.best_params_
        ),

        str(
            gbr_random.best_params_
        )
    ]
})

params_df.to_csv(

    f"{OUTPUT_DIR}/best_hyperparameters.csv",

    index=False
)

comparison_df.set_index(
    "Model"
).plot(

    kind="bar",

    figsize=(10,6)
)

plt.ylabel("R² Score")

plt.title(
    "Before vs After Hyperparameter Tuning"
)

plt.tight_layout()

plt.savefig(

    f"{PLOT_DIR}/before_vs_after_tuning.png"
)

plt.close()

print("\n" + "="*60)
print("WEEK 8 STAGE 1 COMPLETED")
print("="*60)

print("\nGenerated")

print("before_after_tuning.csv")
print("best_hyperparameters.csv")
print("before_vs_after_tuning.png")

# ==================================================
# WEEK 8 STAGE 2
# ENSEMBLE COMPARISON
# ==================================================

ensemble_results = []
dt = Pipeline([

    ("preprocessor", preprocessor),

    (
        "model",
        DecisionTreeRegressor(
            random_state=42
        )
    )
])

dt.fit(
    X_train,
    y_train
)

dt_pred = dt.predict(
    X_test
)

dt_r2 = r2_score(
    y_test,
    dt_pred
)

dt_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        dt_pred
    )
)

dt_mae = mean_absolute_error(
    y_test,
    dt_pred
)

ensemble_results.append([

    "Decision Tree",

    "Base",

    dt_r2,

    dt_rmse,

    dt_mae
])

rf_best = rf_grid_search.best_estimator_

rf_pred = rf_best.predict(
    X_test
)

rf_r2 = r2_score(
    y_test,
    rf_pred
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_pred
    )
)

rf_mae = mean_absolute_error(
    y_test,
    rf_pred
)

ensemble_results.append([

    "Random Forest",

    "Bagging",

    rf_r2,

    rf_rmse,

    rf_mae
])

gbr_best = (
    gbr_grid_search
    .best_estimator_
)

gbr_pred = gbr_best.predict(
    X_test
)

gbr_r2 = r2_score(
    y_test,
    gbr_pred
)

gbr_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        gbr_pred
    )
)

gbr_mae = mean_absolute_error(
    y_test,
    gbr_pred
)

ensemble_results.append([

    "Gradient Boosting",

    "Boosting",

    gbr_r2,

    gbr_rmse,

    gbr_mae
])

X_train_processed = (
    preprocessor.fit_transform(
        X_train
    )
)

X_test_processed = (
    preprocessor.transform(
        X_test
    )
)
voting = VotingRegressor([

    (
        "dt",
        DecisionTreeRegressor(
            random_state=42
        )
    ),

    (
        "rf",
        RandomForestRegressor(
            n_estimators=300,
            random_state=42
        )
    ),

    (
        "gbr",
        GradientBoostingRegressor(
            random_state=42
        )
    )
])

voting.fit(
    X_train_processed,
    y_train
)

voting_pred = voting.predict(
    X_test_processed
)

voting_r2 = r2_score(
    y_test,
    voting_pred
)

voting_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        voting_pred
    )
)

voting_mae = mean_absolute_error(
    y_test,
    voting_pred
)

ensemble_results.append([

    "Voting Regressor",

    "Voting",

    voting_r2,

    voting_rmse,

    voting_mae
])
ensemble_df = pd.DataFrame(

    ensemble_results,

    columns=[

        "Model",

        "Ensemble_Type",

        "R2",

        "RMSE",

        "MAE"
    ]
)

ensemble_df.to_csv(

    f"{OUTPUT_DIR}/ensemble_comparison.csv",

    index=False
)

ranking_df = ensemble_df.sort_values(

    by="R2",

    ascending=False
)

ranking_df.insert(

    0,

    "Rank",

    range(
        1,
        len(ranking_df)+1
    )
)

ranking_df.to_csv(

    f"{OUTPUT_DIR}/final_model_ranking.csv",

    index=False
)

plt.figure(figsize=(10,6))

plt.bar(

    ranking_df["Model"],

    ranking_df["R2"]
)

plt.ylabel("R²")

plt.title(
    "Ensemble Method Comparison"
)

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(

    f"{PLOT_DIR}/ensemble_r2_comparison.png"
)

plt.close()

plt.figure(figsize=(10,6))

plt.bar(

    ranking_df["Model"],

    ranking_df["RMSE"]
)

plt.ylabel("RMSE")

plt.title(
    "Ensemble RMSE Comparison"
)

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(

    f"{PLOT_DIR}/ensemble_rmse_comparison.png"
)

plt.close()

print("\n" + "="*60)
print("WEEK 8 COMPLETED")
print("="*60)

print("\nGenerated")

print("ensemble_comparison.csv")
print("final_model_ranking.csv")
print("ensemble_r2_comparison.png")
print("ensemble_rmse_comparison.png")
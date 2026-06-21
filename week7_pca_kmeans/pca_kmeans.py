import os
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import (
    StandardScaler
)

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ==================================================
# CONFIG
# ==================================================

DATA_PATH = (
    "week5_feature_engineering/outputs/"
    "engineered_dataset.csv"
)

OUTPUT_DIR = (
    "week7_pca_kmeans/outputs"
)

PLOT_DIR = (
    f"{OUTPUT_DIR}/plots"
)

REPORT_DIR = (
    f"{OUTPUT_DIR}/reports"
)

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

print("Output folders ready")

# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded")
print(df.shape)

# ==================================================
# REMOVE TARGET
# ==================================================

TARGET = "ger_all_total"

if TARGET in df.columns:
    df = df.drop(columns=[TARGET])

# ==================================================
# REMOVE LEAKAGE IF PRESENT
# ==================================================

LEAKAGE = [

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
    columns=LEAKAGE,
    errors="ignore"
)

# ==================================================
# ONE HOT ENCODE
# ==================================================

df = pd.get_dummies(
    df,
    drop_first=True
)

print("\nAfter Encoding")
print(df.shape)

# ==================================================
# STANDARD SCALE
# ==================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(df)

# ==================================================
# PCA ANALYSIS
# ==================================================

pca_full = PCA()

X_pca_full = pca_full.fit_transform(
    X_scaled
)

explained_variance = (
    pca_full.explained_variance_ratio_
)

cumulative_variance = np.cumsum(
    explained_variance
)

# Save variance table

variance_df = pd.DataFrame({

    "Component":
        range(
            1,
            len(explained_variance)+1
        ),

    "Explained_Variance":
        explained_variance,

    "Cumulative_Variance":
        cumulative_variance
})

variance_df.to_csv(

    f"{REPORT_DIR}/pca_variance.csv",

    index=False
)

# ==================================================
# FIND 95% VARIANCE COMPONENTS
# ==================================================

optimal_components = np.argmax(
    cumulative_variance >= 0.95
) + 1

optimal_df = pd.DataFrame({

    "Optimal_Components":
        [optimal_components],

    "Variance_Retained":
        [cumulative_variance[
            optimal_components-1
        ]]
})

optimal_df.to_csv(

    f"{REPORT_DIR}/optimal_pca_components.csv",

    index=False
)

print(
    f"\n95% Variance Retained at "
    f"{optimal_components} Components"
)

# ==================================================
# PCA PLOTS
# ==================================================

plt.figure(figsize=(10,6))

plt.plot(
    range(
        1,
        len(explained_variance)+1
    ),
    explained_variance,
    marker="o"
)

plt.xlabel("Principal Component")

plt.ylabel(
    "Explained Variance Ratio"
)

plt.title(
    "Explained Variance by Component"
)

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/explained_variance.png"
)

plt.close()

plt.figure(figsize=(10,6))

plt.plot(
    range(
        1,
        len(cumulative_variance)+1
    ),
    cumulative_variance,
    marker="o"
)

plt.axhline(
    y=0.95,
    linestyle="--"
)

plt.xlabel(
    "Principal Component"
)

plt.ylabel(
    "Cumulative Variance"
)

plt.title(
    "Cumulative Explained Variance"
)

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/cumulative_variance.png"
)

plt.close()

# ==================================================
# PCA REDUCTION
# ==================================================

pca = PCA(
    n_components=optimal_components
)

X_pca = pca.fit_transform(
    X_scaled
)

# ==================================================
# ELBOW METHOD
# ==================================================

wcss = []

for k in range(2, 8):

    kmeans = KMeans(

        n_clusters=k,

        random_state=42,

        n_init=10
    )

    kmeans.fit(X_pca)

    wcss.append(
        kmeans.inertia_
    )

elbow_df = pd.DataFrame({

    "K":
        list(range(2,8)),

    "WCSS":
        wcss
})

elbow_df.to_csv(

    f"{REPORT_DIR}/elbow_results.csv",

    index=False
)

plt.figure(figsize=(8,6))

plt.plot(
    range(2,8),
    wcss,
    marker="o"
)

plt.xlabel("K")

plt.ylabel("WCSS")

plt.title(
    "Elbow Method"
)

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/elbow_plot.png"
)

plt.close()

# ==================================================
# SILHOUETTE ANALYSIS
# ==================================================

silhouette_scores = []

for k in range(2, 8):

    kmeans = KMeans(

        n_clusters=k,

        random_state=42,

        n_init=10
    )

    labels = kmeans.fit_predict(
        X_pca
    )

    score = silhouette_score(
        X_pca,
        labels
    )

    silhouette_scores.append(
        score
    )

silhouette_df = pd.DataFrame({

    "K":
        list(range(2,8)),

    "Silhouette_Score":
        silhouette_scores
})

silhouette_df.to_csv(

    f"{REPORT_DIR}/silhouette_results.csv",

    index=False
)

best_k = (
    silhouette_df
    .loc[
        silhouette_df[
            "Silhouette_Score"
        ].idxmax(),
        "K"
    ]
)

print(
    f"\nBest K Selected = {best_k}"
)

plt.figure(figsize=(8,6))

plt.plot(
    range(2,8),
    silhouette_scores,
    marker="o"
)

plt.xlabel("K")

plt.ylabel(
    "Silhouette Score"
)

plt.title(
    "Silhouette Analysis"
)

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/silhouette_plot.png"
)

plt.close()

# ==================================================
# FINAL KMEANS
# ==================================================

kmeans = KMeans(

    n_clusters=int(best_k),

    random_state=42,

    n_init=10
)

clusters = kmeans.fit_predict(
    X_pca
)

# ==================================================
# CLUSTER VISUALIZATION
# ==================================================

pca_vis = PCA(
    n_components=2
)

X_vis = pca_vis.fit_transform(
    X_scaled
)

plt.figure(figsize=(10,7))

sns.scatterplot(

    x=X_vis[:,0],

    y=X_vis[:,1],

    hue=clusters,

    palette="deep"
)

plt.title(
    f"KMeans Clusters (K={best_k})"
)

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/cluster_visualization.png"
)

plt.close()

# ==================================================
# SAVE CLUSTERS
# ==================================================

cluster_df = pd.DataFrame({

    "PC1":
        X_vis[:,0],

    "PC2":
        X_vis[:,1],

    "Cluster":
        clusters
})

cluster_df.to_csv(

    f"{REPORT_DIR}/cluster_assignments.csv",

    index=False
)

# ==================================================
# COMPLETE
# ==================================================

print("\n" + "="*60)
print("WEEK 7 COMPLETED")
print("="*60)

print("\nGenerated")

print("PCA Variance Report")
print("Optimal PCA Components")
print("Elbow Analysis")
print("Silhouette Analysis")
print("Cluster Visualization")
print("Cluster Assignments")
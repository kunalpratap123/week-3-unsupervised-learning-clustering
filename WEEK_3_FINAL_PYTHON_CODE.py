# Week 3 — Unsupervised Learning and Clustering Analysis
# Dataset: New York City Airbnb Open Data 2019
# Place AB_NYC_2019.csv in this folder before running.

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

DATA = Path("AB_NYC_2019.csv")
OUT = Path("week3_outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

features = ["price","minimum_nights","number_of_reviews",
            "reviews_per_month","calculated_host_listings_count",
            "availability_365"]

work = df[features].copy()

for c in features:
    work[c] = pd.to_numeric(work[c], errors="coerce")

work.loc[work["price"] <= 0, "price"] = np.nan
work.loc[work["minimum_nights"] < 1, "minimum_nights"] = np.nan
work["reviews_per_month"] = work["reviews_per_month"].fillna(0)

for c in features:
    work[c] = work[c].fillna(work[c].median())

for c in ["price","minimum_nights","number_of_reviews",
          "reviews_per_month","calculated_host_listings_count"]:
    work[c] = np.log1p(work[c])

scaler = StandardScaler()
X = scaler.fit_transform(work)

k_values = range(2, 9)
inertias, silhouettes = [], []

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(X)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X, labels))

metrics = pd.DataFrame({
    "k": list(k_values),
    "inertia": inertias,
    "silhouette_score": silhouettes
})
metrics.to_csv(OUT/"cluster_metrics.csv", index=False)

plt.figure(figsize=(8,5))
plt.plot(list(k_values), inertias, marker="o")
plt.title("Elbow Method for Selecting Number of Clusters")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.tight_layout()
plt.savefig(OUT/"elbow_curve.png", dpi=180)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(list(k_values), silhouettes, marker="o")
plt.title("Silhouette Score by Number of Clusters")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.xticks(list(k_values))
plt.tight_layout()
plt.savefig(OUT/"silhouette_scores.png", dpi=180)
plt.close()

best_k = int(metrics.loc[metrics["silhouette_score"].idxmax(), "k"])
print("Selected k:", best_k)
print(metrics)

model = KMeans(n_clusters=best_k, random_state=42, n_init=20)
work["cluster"] = model.fit_predict(X)

profile = work.groupby("cluster")[features].mean()
profile.to_csv(OUT/"cluster_profiles.csv")

result = df.copy()
result["cluster"] = work["cluster"].values
result.to_csv(OUT/"clustered_airbnb.csv", index=False)

summary = result["cluster"].value_counts().sort_index().rename("listing_count").to_frame()
summary["percentage"] = summary["listing_count"] / len(result) * 100
summary.to_csv(OUT/"cluster_summary.csv")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

plot_df = pd.DataFrame({
    "PC1": X_pca[:,0], "PC2": X_pca[:,1],
    "cluster": work["cluster"].astype(str)
})

plt.figure(figsize=(9,6))
sns.scatterplot(data=plot_df.sample(min(12000,len(plot_df)), random_state=42),
                x="PC1", y="PC2", hue="cluster", alpha=.55)
plt.title(f"Airbnb Listing Clusters — PCA Projection (k={best_k})")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig(OUT/"clusters_pca.png", dpi=180)
plt.close()

centroids = pd.DataFrame(model.cluster_centers_, columns=features)
plt.figure(figsize=(10,5))
sns.heatmap(centroids, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Standardized Cluster Centroids")
plt.xlabel("Features")
plt.ylabel("Cluster")
plt.tight_layout()
plt.savefig(OUT/"cluster_centroid_heatmap.png", dpi=180)
plt.close()

print("\nCluster sizes:\n", summary)
print("\nCluster profiles:\n", profile)
print("\nFinal silhouette score:", silhouette_score(X, work["cluster"]))
print("\nCompleted. Outputs saved in:", OUT)

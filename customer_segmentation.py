# customer_segmentation.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path):
    """
    Load dataset from CSV/excel, return DataFrame.
    """
    df = pd.read_csv(path)  # or pd.read_excel(...)
    return df

def preprocess_data(df):
    """
    Clean data, handle missing values, convert data types, remove outliers if any.
    Returns cleaned df.
    """
    # Example:
    # df.dropna(inplace=True)
    # Or fillna, or drop columns etc.
    # Convert date columns, etc.
    return df

def compute_rfm(df, date_column, customer_id_col, monetary_col):
    """
    Compute Recency, Frequency, Monetary metrics.
    date_column: last purchase date / invoice date etc.
    """
    import datetime
    snapshot_date = df[date_column].max() + pd.DateOffset(days=1)
    rfm = df.groupby(customer_id_col).agg({
        date_column: lambda x: (snapshot_date - x.max()).days,
        customer_id_col: 'count',
        monetary_col: 'sum'
    })
    rfm.rename(columns={date_column: 'Recency',
                        customer_id_col: 'Frequency',
                        monetary_col: 'Monetary'}, inplace=True)
    return rfm

def scale_data(rfm_df):
    """
    Standardize / normalize the RFM data prior to clustering.
    """
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df)
    return rfm_scaled

def perform_pca(rfm_scaled, n_components=2):
    """
    Optional: Reduce dimensions for visualization.
    """
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(rfm_scaled)
    return principal_components

def find_optimal_k(rfm_scaled, max_k=10):
    """
    Use Elbow Method or Silhouette Score to find optimal number of clusters.
    """
    sse = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(rfm_scaled)
        sse.append(km.inertia_)
    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(range(2, max_k + 1), sse, marker='o')
    plt.xlabel('Number of clusters k')
    plt.ylabel('SSE (Inertia)')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    return sse

def cluster_customers(rfm_scaled, k):
    """
    Apply KMeans clustering and return cluster labels.
    """
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(rfm_scaled)
    return labels

def visualize_clusters(pca_data, labels, rfm_original):
    """
    Visualize clusters in 2D with PCA, optionally show cluster stats.
    """
    df_viz = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
    df_viz['Cluster'] = labels
    # Merge with RFM original for stats
    df_merge = pd.concat([rfm_original.reset_index(), df_viz], axis=1)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_merge, x='PC1', y='PC2', hue='Cluster', palette='Set2')
    plt.title("Customer Segments Visualized via PCA")
    plt.show()

    # Optional: summary by cluster
    print("Cluster centroids / means:")
    print(df_merge.groupby('Cluster').agg({
        'Recency':'mean',
        'Frequency':'mean',
        'Monetary':'mean'
    }))

def main():
    # Replace with your path
    df = load_data('data/online_retail.csv')
    df_clean = preprocess_data(df)
    rfm = compute_rfm(df_clean, date_column='InvoiceDate', customer_id_col='CustomerID', monetary_col='TotalAmount')
    rfm_scaled = scale_data(rfm)
    pca_data = perform_pca(rfm_scaled, n_components=2)
    _ = find_optimal_k(rfm_scaled, max_k=10)
    labels = cluster_customers(rfm_scaled, k=4)  # say 4 clusters
    visualize_clusters(pca_data, labels, rfm)

if __name__ == "__main__":
    main()

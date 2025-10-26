import datetime
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

os.makedirs("figures", exist_ok=True)


def pad_tensor_sequence(tensors: list[np.ndarray], padding_value=0.0):
    """
    Pads a list of (seq_len, feature_dim) tensors along the token (row) dimension
    to the max length in the batch.

    Returns:
        padded_tensor: np.ndarray of shape (num_samples, max_seq_len, feature_dim)
    """
    max_len = max(t.shape[0] for t in tensors)

    padded = np.stack(
        [
            np.pad(
                t,
                ((0, max_len - t.shape[0]), (0, 0)),
                mode="constant",
                constant_values=padding_value,
            )
            for t in tensors
        ]
    )
    return padded


def train_pca(data):
    logging.info("Training PCA")
    model = PCA(n_components=1024, random_state=42)

    # Collect all token-level embeddings into one big matrix
    token_matrix = np.vstack(
        [sample["tensor_last_layer"] for sample in data]
    )  # shape: (total_tokens, feature_dim)

    model.fit(token_matrix)
    return model


def train_gmm(data, pca_model, n_components=350):
    """
    Train a Gaussian Mixture Model on PCA-transformed token tensors.

    Args:
        data (list[dict]): List of samples with 'tensor' key holding token embeddings.
        pca_model (PCA): Trained PCA model for dimensionality reduction.
        n_components (int): Number of GMM components.

    Returns:
        GaussianMixture: Trained GMM model.
    """
    logging.info("Training GMM model")

    # Stack and project all token tensors
    token_matrix = np.vstack(
        [pca_model.transform(sample["tensor_last_layer"]) for sample in data]
    )

    # Fit GMM
    model = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        random_state=42,
        verbose=True,
        max_iter=200,
        reg_covar=1e-2,
    )
    model.fit(token_matrix)

    return model


def train_kmm(data, pca_model):
    logging.info("Training KMM-model")
    token_matrix = np.vstack(
        [pca_model.transform(sample["tensor_last_layer"]) for sample in data]
    )

    model = KMeans(n_clusters=250, random_state=42)
    model.fit(token_matrix)

    return model


def save_model(model, model_name):
    directory = "feature_abstraction_models"
    os.makedirs(directory, exist_ok=True)

    path = os.path.join(directory, f"{model_name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)

    logging.info(f"Model saved to {path}")


def load_model(model_name):
    path = f"feature_abstraction_models/{model_name}.pkl"

    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def parse_hmm_features(data, pca_model, clustering_model):
    for sample in data:
        tensor = sample["tensor_last_layer"]  # shape: (seq_len, feature_dim)
        reduced = pca_model.transform(tensor)  # shape: (seq_len, pca_dim)
        cluster_ids = clustering_model.predict(reduced)  # shape: (seq_len,)
        sample["pollm_graph_feature"] = (
            cluster_ids  # or store as list(cluster_ids) if needed
        )
    return data


def abstract_hhmm_features(
    benchmark,
    llm,
    data,
    train: bool = True,
    cluster_model_id: str = "gmm",
    experiment: str = "unknown",
):
    train_data, validation_data, test_data = data
    name = f"{llm}_{benchmark}_{experiment}"
    try:
        clustering_model = load_model(f"{cluster_model_id}_{name}")
        pca_model = load_model(f"pca_{name}")
    except FileNotFoundError:
        pca_model = train_pca(train_data)
        save_model(pca_model, model_name=f"pca_{name}")
        if cluster_model_id == "kmm":
            clustering_model = train_kmm(train_data, pca_model)
        if cluster_model_id == "gmm":
            clustering_model = train_gmm(train_data, pca_model)

        save_model(clustering_model, model_name=f"{cluster_model_id}_{name}")

    train_data = parse_hmm_features(train_data, pca_model, clustering_model)
    validation_data = parse_hmm_features(validation_data, pca_model, clustering_model)
    test_data = parse_hmm_features(test_data, pca_model, clustering_model)
    return train_data, validation_data, test_data


def visualize_pca_clusters_by_label(sequences, pca_model):

    all_vectors = []
    all_labels = []

    for sample in sequences:
        tensor = sample["tensor_last_layer"]  # shape: (seq_len, feature_dim)
        reduced = pca_model.transform(tensor)  # shape: (seq_len, pca_dim)
        all_vectors.append(reduced)

        # Extend hallucination label across all tokens
        label = sample["Hallucinating"]
        all_labels.extend([label] * reduced.shape[0])

    all_vectors = np.vstack(all_vectors)
    all_labels = np.array(all_labels)

    # 2D projection for visualization
    reduced_2d = PCA(n_components=2).fit_transform(all_vectors)

    # Plot
    plt.figure(figsize=(8, 6))
    colors = ["blue" if label == 0 else "red" for label in all_labels]
    plt.scatter(reduced_2d[:, 0], reduced_2d[:, 1], c=colors, s=3, alpha=0.5)

    plt.title("PCA Projection Colored by Hallucination Label")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    red_patch = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Hallucinating",
        markerfacecolor="red",
        markersize=8,
    )
    blue_patch = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Not Hallucinating",
        markerfacecolor="blue",
        markersize=8,
    )
    plt.legend(handles=[red_patch, blue_patch])

    # Save figure
    import datetime
    import os

    os.makedirs("figures", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"figures/pca_by_hallucination_{timestamp}.png", dpi=300)
    plt.close()


def visualize_pca_clusters(data, pca, kmeans):
    """
    Visualizes the PCA-reduced token embeddings colored by KMeans cluster ID.
    """
    # Stack all token embeddings into a big matrix (N_tokens, D)
    all_vectors = np.vstack([sample["tensor_last_layer"] for sample in data])

    # Reduce dimensionality
    reduced = pca.transform(all_vectors)  # shape: (N_tokens, 2 or 3)

    # Predict cluster IDs for coloring
    cluster_ids = kmeans.predict(pca.transform(all_vectors))

    # Plot first two PCA dimensions
    plt.figure(figsize=(6, 6))
    plt.scatter(
        reduced[:, 0], reduced[:, 1], c=cluster_ids, cmap="tab20", s=3, alpha=0.6
    )
    plt.title("PCA Projection Colored by KMeans Clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"figures/pca_kmeans_clusters_{timestamp}.png"
    plt.savefig(path, dpi=300)
    plt.close()
    logging.info(f"Saved PCA cluster visualization to {path}")


def plot_pca_variance(pca_model, name="pca_variance_bar"):
    """
    Plots a bar chart showing how much variance each PCA component explains.
    """

    os.makedirs("figures", exist_ok=True)

    explained_var = pca_model.explained_variance_ratio_
    n_components = len(explained_var)

    plt.figure(figsize=(12, 6))
    plt.bar(range(1, n_components + 1), explained_var, alpha=0.8, color="skyblue")
    plt.xlabel("PCA Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Explained Variance per PCA Component")
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"figures/{name}_{timestamp}.png", dpi=300)
    plt.close()


def preprocess_data_for_hhmm_training(data):
    processed_data = [
        (entry["features"].tolist(), entry["Hallucinating"])
        for entry in data
        if "features" in entry and "Hallucinating" in entry
    ]
    return processed_data

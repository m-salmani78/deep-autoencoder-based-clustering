import numpy as np
import torch.utils.data
from sklearn.cluster import KMeans

def k_means_clustering(train_loader, autoencoder, device):
    true_labels = []
    encoded_data = []

    for inputs, label in train_loader.dataset:
        true_labels.append(label)
        inputs = inputs.view(-1).to(device)
        with torch.no_grad():
            encoder = autoencoder.encoder(inputs)
            encoded_data.append(encoder.cpu().numpy())

    # Extract the encoded representations from the data
    encoded_data = np.array(encoded_data)
    # The labels assigned by K-Means to each data point
    true_labels = np.array(true_labels)

    # Perform K-Means clustering on the encoded data
    n_clusters = 10  # number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    predict_labels = kmeans.fit_predict(encoded_data)

    return true_labels, predict_labels

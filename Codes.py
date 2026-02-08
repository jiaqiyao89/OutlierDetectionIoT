import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.cluster import DBSCAN, KMeans
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Global Configuration
class Config:
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Autoencoder parameters
    LATENT_DIM = 16
    HIDDEN_DIM = 128
    AE_EPOCHS = 120
    BATCH_SIZE = 128
    LR = 1e-3
    
    # Contrastive learning parameters
    MARGIN = 1.0
    LAMBDA_CONTRASTIVE = 0.7
    
    # Outlier threshold
    THRESHOLD_METHOD = "gaussian"  # "gaussian" or "percentile"

cfg = Config()

# Reproducibility
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

set_seed(cfg.SEED)

# Dataset Loader
def load_dataset(path):
    data = pd.read_csv(path, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Convert multi-class labels to binary anomaly labels
    majority_class = np.bincount(y).argmax()
    y = (y != majority_class).astype(int)
    
    return X, y

# Preprocessing
def preprocess(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# Train/Validation/Test Split
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=cfg.SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=cfg.SEED, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# Evaluation Metrics
def evaluate(y_true, y_pred, scores):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, scores)
    }

# Baseline Methods
def run_dbscan(X_train, X_test):
    model = DBSCAN(eps=0.7, min_samples=5)
    model.fit(X_train)
    
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X_train)
    distances, _ = nn.kneighbors(X_test)
    scores = distances.mean(axis=1)
    
    tau = np.percentile(scores, 95)
    y_pred = (scores > tau).astype(int)
    return y_pred, scores

def run_kmeans(X_train, X_test, k=6):
    model = KMeans(n_clusters=k, random_state=cfg.SEED)
    model.fit(X_train)
    
    distances = np.min(
        np.linalg.norm(X_test[:, None] - model.cluster_centers_, axis=2),
        axis=1
    )
    tau = np.percentile(distances, 95)
    y_pred = (distances > tau).astype(int)
    return y_pred, distances

def run_ocsvm(X_train, X_test):
    model = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
    model.fit(X_train)
    
    scores = -model.decision_function(X_test)
    tau = np.percentile(scores, 95)
    y_pred = (scores > tau).astype(int)
    return y_pred, scores

def run_bayesian_model(X_train, X_test):
    # Gaussian Mixture Model as probabilistic Bayesian density estimator
    gmm = GaussianMixture(n_components=3, random_state=cfg.SEED)
    gmm.fit(X_train)
    
    scores = -gmm.score_samples(X_test)
    tau = np.percentile(scores, 95)
    y_pred = (scores > tau).astype(int)
    return y_pred, scores

# Autoencoder Architecture
class Encoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(cfg.HIDDEN_DIM, cfg.LATENT_DIM)
        )
    
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.LATENT_DIM, cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(cfg.HIDDEN_DIM, input_dim)
        )
    
    def forward(self, z):
        return self.net(z)

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = Encoder(input_dim)
        self.decoder = Decoder(input_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

# Training Autoencoder (Mini-batch)
def train_autoencoder(X_train):
    model = Autoencoder(X_train.shape[1]).to(cfg.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    loss_fn = nn.MSELoss()
    
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    
    for epoch in range(cfg.AE_EPOCHS):
        total_loss = 0
        for (x_batch,) in loader:
            x_batch = x_batch.to(cfg.DEVICE)
            
            z, x_hat = model(x_batch)
            loss = loss_fn(x_hat, x_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 20 == 0:
            print(f"[AE] Epoch {epoch}, Loss = {total_loss:.6f}")
    
    return model

def reconstruction_error(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(cfg.DEVICE)
        _, x_hat = model(X_tensor)
        errors = torch.mean((X_tensor - x_hat) ** 2, dim=1).cpu().numpy()
    return errors

# Contrastive Learning Module
def build_pairs(X, batch_size):
    idx = np.random.choice(len(X), batch_size, replace=False)
    X_batch = X[idx]
    
    pairs_i, pairs_j, labels = [], [], []
    
    for i in range(batch_size):
        j = np.random.randint(0, batch_size)
        xi, xj = X_batch[i], X_batch[j]
        
        # Self-supervised similarity based on local density
        dist_i = np.linalg.norm(X_batch - xi, axis=1)
        threshold = np.percentile(dist_i, 50)
        label = 1 if np.linalg.norm(xi - xj) < threshold else 0
        
        pairs_i.append(xi)
        pairs_j.append(xj)
        labels.append(label)
    
    return np.array(pairs_i), np.array(pairs_j), np.array(labels)

def contrastive_loss(z1, z2, y):
    dist = torch.norm(z1 - z2, dim=1)
    pos_loss = y * dist**2
    neg_loss = (1 - y) * torch.clamp(cfg.MARGIN - dist, min=0.0)**2
    return torch.mean(pos_loss + neg_loss)

# Training Proposed Model
def train_contrastive_autoencoder(X_train):
    model = Autoencoder(X_train.shape[1]).to(cfg.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    mse = nn.MSELoss()
    
    for epoch in range(cfg.AE_EPOCHS):
        epoch_loss = 0
        
        for _ in range(len(X_train) // cfg.BATCH_SIZE):
            xi, xj, y = build_pairs(X_train, cfg.BATCH_SIZE)
            
            xi = torch.tensor(xi, dtype=torch.float32).to(cfg.DEVICE)
            xj = torch.tensor(xj, dtype=torch.float32).to(cfg.DEVICE)
            y = torch.tensor(y, dtype=torch.float32).to(cfg.DEVICE)
            
            zi, xi_hat = model(xi)
            zj, _ = model(xj)
            
            rec_loss = mse(xi_hat, xi)
            cont_loss = contrastive_loss(zi, zj, y)
            
            loss = rec_loss + cfg.LAMBDA_CONTRASTIVE * cont_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if epoch % 20 == 0:
            print(f"[CAE] Epoch {epoch}, Loss = {epoch_loss:.6f}")
    
    return model

# Threshold Estimation
def compute_threshold(errors):
    if cfg.THRESHOLD_METHOD == "gaussian":
        mu = np.mean(errors)
        sigma = np.std(errors)
        return mu + 1.96 * sigma
    else:
        return np.percentile(errors, 95)

# Main Experiment
def main():
    print("Loading dataset...")
    X, y = load_dataset("statlog_landsat.csv")  # update path if needed
    
    X = preprocess(X)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    results = {}
    
    # DBSCAN
    y_pred, scores = run_dbscan(X_train, X_test)
    results["DBSCAN"] = evaluate(y_test, y_pred, scores)
    
    # k-Means
    y_pred, scores = run_kmeans(X_train, X_test)
    results["k-Means"] = evaluate(y_test, y_pred, scores)
    
    # One-Class SVM
    y_pred, scores = run_ocsvm(X_train, X_test)
    results["One-Class SVM"] = evaluate(y_test, y_pred, scores)
    
    # Bayesian (GMM)
    y_pred, scores = run_bayesian_model(X_train, X_test)
    results["Bayesian Model"] = evaluate(y_test, y_pred, scores)
    
    # Autoencoder
    ae_model = train_autoencoder(X_train)
    val_err = reconstruction_error(ae_model, X_val)
    tau = compute_threshold(val_err)
    test_err = reconstruction_error(ae_model, X_test)
    y_pred = (test_err > tau).astype(int)
    results["Autoencoder"] = evaluate(y_test, y_pred, test_err)
    
    # Proposed Method
    cae_model = train_contrastive_autoencoder(X_train)
    val_err = reconstruction_error(cae_model, X_val)
    tau = compute_threshold(val_err)
    test_err = reconstruction_error(cae_model, X_test)
    y_pred = (test_err > tau).astype(int)
    results["Proposed"] = evaluate(y_test, y_pred, test_err)
    
    print("\n================ FINAL RESULTS ================")
    for method, metrics in results.items():
        print(f"\n{method}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()

"""
Evaluation metrics for HyperGraph-MAE.
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from typing import Dict, List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class MetricCalculator:
    """Calculate various metrics for model evaluation."""

    def __init__(self, task_type: str = 'reconstruction'):
        """
        Initialize metric calculator.

        Args:
            task_type: Type of task ('reconstruction', 'classification', 'regression')
        """
        self.task_type = task_type
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.predictions = []
        self.targets = []
        self.losses = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor,
               loss: Optional[float] = None):
        """
        Update metrics with new predictions.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            loss: Optional loss value
        """
        # Move to CPU and convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu()

        self.predictions.append(predictions)
        self.targets.append(targets)

        if loss is not None:
            self.losses.append(loss)

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            Dictionary of computed metrics
        """
        if not self.predictions:
            return {}

        # Concatenate all predictions and targets
        all_preds = torch.cat(self.predictions, dim=0).numpy()
        all_targets = torch.cat(self.targets, dim=0).numpy()

        metrics = {}

        # Average loss
        if self.losses:
            metrics['loss'] = np.mean(self.losses)

        # Task-specific metrics
        if self.task_type == 'reconstruction':
            metrics.update(self._compute_reconstruction_metrics(all_preds, all_targets))
        elif self.task_type == 'classification':
            metrics.update(self._compute_classification_metrics(all_preds, all_targets))
        elif self.task_type == 'regression':
            metrics.update(self._compute_regression_metrics(all_preds, all_targets))

        return metrics

    def _compute_reconstruction_metrics(self, predictions: np.ndarray,
                                        targets: np.ndarray) -> Dict[str, float]:
        """Compute reconstruction metrics."""
        metrics = {}

        # Mean Squared Error
        metrics['mse'] = mean_squared_error(targets.flatten(), predictions.flatten())

        # Mean Absolute Error
        metrics['mae'] = mean_absolute_error(targets.flatten(), predictions.flatten())

        # R-squared
        metrics['r2'] = r2_score(targets.flatten(), predictions.flatten())

        # Cosine similarity
        pred_norm = predictions / (np.linalg.norm(predictions, axis=-1, keepdims=True) + 1e-8)
        target_norm = targets / (np.linalg.norm(targets, axis=-1, keepdims=True) + 1e-8)
        cosine_sim = np.sum(pred_norm * target_norm, axis=-1)
        metrics['cosine_similarity'] = np.mean(cosine_sim)

        # Pearson correlation
        if len(predictions.shape) == 2:
            correlations = []
            for i in range(predictions.shape[0]):
                corr = np.corrcoef(predictions[i], targets[i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            if correlations:
                metrics['pearson_correlation'] = np.mean(correlations)

        return metrics

    def _compute_classification_metrics(self, predictions: np.ndarray,
                                        targets: np.ndarray) -> Dict[str, float]:
        """Compute classification metrics."""
        metrics = {}

        # Binary classification
        if predictions.shape[-1] == 1 or len(predictions.shape) == 1:
            # Ensure predictions are probabilities
            if predictions.min() < 0 or predictions.max() > 1:
                predictions = 1 / (1 + np.exp(-predictions))  # Sigmoid

            # Accuracy (with 0.5 threshold)
            pred_labels = (predictions > 0.5).astype(int)
            metrics['accuracy'] = np.mean(pred_labels == targets)

            # ROC-AUC
            try:
                metrics['roc_auc'] = roc_auc_score(targets, predictions)
            except:
                pass

            # Average Precision
            try:
                metrics['avg_precision'] = average_precision_score(targets, predictions)
            except:
                pass

        # Multi-class classification
        else:
            # Accuracy
            pred_labels = np.argmax(predictions, axis=-1)
            true_labels = np.argmax(targets, axis=-1) if len(targets.shape) > 1 else targets
            metrics['accuracy'] = np.mean(pred_labels == true_labels)

            # Per-class metrics
            num_classes = predictions.shape[-1]
            for i in range(num_classes):
                try:
                    class_targets = (true_labels == i).astype(int)
                    class_preds = predictions[:, i]
                    metrics[f'class_{i}_roc_auc'] = roc_auc_score(class_targets, class_preds)
                except:
                    pass

        return metrics

    def _compute_regression_metrics(self, predictions: np.ndarray,
                                    targets: np.ndarray) -> Dict[str, float]:
        """Compute regression metrics."""
        metrics = {}

        # Flatten if needed
        predictions = predictions.flatten()
        targets = targets.flatten()

        # MSE and RMSE
        mse = mean_squared_error(targets, predictions)
        metrics['mse'] = mse
        metrics['rmse'] = np.sqrt(mse)

        # MAE
        metrics['mae'] = mean_absolute_error(targets, predictions)

        # R-squared
        metrics['r2'] = r2_score(targets, predictions)

        # Explained variance
        metrics['explained_variance'] = 1 - np.var(targets - predictions) / np.var(targets)

        # Pearson correlation
        corr = np.corrcoef(predictions, targets)[0, 1]
        if not np.isnan(corr):
            metrics['pearson_r'] = corr

        # Spearman correlation
        from scipy.stats import spearmanr
        spearman_corr, _ = spearmanr(predictions, targets)
        if not np.isnan(spearman_corr):
            metrics['spearman_r'] = spearman_corr

        return metrics


def compute_embedding_quality_metrics(embeddings: torch.Tensor,
                                      labels: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Compute quality metrics for learned embeddings.

    Args:
        embeddings: Embedding tensor [num_samples, embedding_dim]
        labels: Optional labels for supervised metrics

    Returns:
        Dictionary of embedding quality metrics
    """
    metrics = {}

    # Move to numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    # Basic statistics
    metrics['embedding_mean'] = float(np.mean(embeddings))
    metrics['embedding_std'] = float(np.std(embeddings))
    metrics['embedding_min'] = float(np.min(embeddings))
    metrics['embedding_max'] = float(np.max(embeddings))

    # Norm statistics
    norms = np.linalg.norm(embeddings, axis=1)
    metrics['norm_mean'] = float(np.mean(norms))
    metrics['norm_std'] = float(np.std(norms))

    # Cosine similarity distribution
    normalized = embeddings / (norms[:, np.newaxis] + 1e-8)
    similarity_matrix = np.dot(normalized, normalized.T)

    # Remove diagonal
    mask = np.ones_like(similarity_matrix, dtype=bool)
    np.fill_diagonal(mask, False)
    similarities = similarity_matrix[mask]

    metrics['cosine_sim_mean'] = float(np.mean(similarities))
    metrics['cosine_sim_std'] = float(np.std(similarities))

    # If labels are provided, compute clustering metrics
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        # Silhouette score
        from sklearn.metrics import silhouette_score
        try:
            metrics['silhouette_score'] = float(silhouette_score(embeddings, labels))
        except:
            pass

        # Calinski-Harabasz score
        from sklearn.metrics import calinski_harabasz_score
        try:
            metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(embeddings, labels))
        except:
            pass

        # Davies-Bouldin score
        from sklearn.metrics import davies_bouldin_score
        try:
            metrics['davies_bouldin_score'] = float(davies_bouldin_score(embeddings, labels))
        except:
            pass

    return metrics


def compute_retrieval_metrics(query_embeddings: torch.Tensor,
                              gallery_embeddings: torch.Tensor,
                              query_labels: torch.Tensor,
                              gallery_labels: torch.Tensor,
                              k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Compute retrieval metrics.

    Args:
        query_embeddings: Query embeddings
        gallery_embeddings: Gallery embeddings
        query_labels: Query labels
        gallery_labels: Gallery labels
        k_values: Values of k for top-k metrics

    Returns:
        Dictionary of retrieval metrics
    """
    metrics = {}

    # Normalize embeddings
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    gallery_embeddings = torch.nn.functional.normalize(gallery_embeddings, p=2, dim=1)

    # Compute similarity matrix
    similarity = torch.mm(query_embeddings, gallery_embeddings.t())

    # Get top-k predictions
    for k in k_values:
        _, topk_indices = similarity.topk(k, dim=1)
        topk_labels = gallery_labels[topk_indices]

        # Compute accuracy@k
        correct = (topk_labels == query_labels.unsqueeze(1)).any(dim=1)
        accuracy = correct.float().mean().item()
        metrics[f'accuracy@{k}'] = accuracy

    # Mean Average Precision
    maps = []
    for i in range(len(query_labels)):
        query_label = query_labels[i]
        similarities = similarity[i]

        # Sort by similarity
        sorted_indices = similarities.argsort(descending=True)
        sorted_labels = gallery_labels[sorted_indices]

        # Compute AP
        relevant = (sorted_labels == query_label)
        if relevant.sum() > 0:
            cumsum = relevant.float().cumsum(0)
            precision = cumsum / torch.arange(1, len(relevant) + 1, device=relevant.device)
            ap = (precision * relevant).sum() / relevant.sum()
            maps.append(ap.item())

    if maps:
        metrics['mAP'] = np.mean(maps)

    return metrics


def masked_mae(predictions: torch.Tensor,
               targets: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute Mean Absolute Error on masked elements.

    Args:
        predictions: Predicted tensor
        targets: Target tensor (same shape as predictions)
        mask: Optional boolean mask selecting elements to evaluate

    Returns:
        Scalar MAE value (float)
    """
    if predictions.numel() == 0 or targets.numel() == 0:
        return 0.0

    # Align sizes conservatively
    valid_size = min(predictions.size(0), targets.size(0))
    preds = predictions[:valid_size]
    targs = targets[:valid_size]

    if mask is not None:
        mask = mask[:valid_size].bool()
        if mask.dim() == 1 and preds.dim() > 1:
            mask = mask.unsqueeze(-1).expand_as(preds)
        if mask.numel() > 0 and mask.any():
            preds = preds[mask]
            targs = targs[mask]
        else:
            # Empty mask: compute MAE on all data rather than returning 0
            logger.warning("Empty mask in masked_mae, computing MAE on all data")

    return torch.nn.functional.l1_loss(preds, targs).item()


def compute_embedding_health_metrics(embeddings: torch.Tensor,
                                     variance_threshold: float = 1e-5) -> Dict[str, float]:
    """
    Assess representation health to detect collapse.

    Metrics:
    - var_mean: mean per-dimension variance
    - var_min: minimum per-dimension variance
    - frac_below_thr: fraction of dimensions with variance below threshold
    - cov_offdiag_mean: mean absolute off-diagonal covariance magnitude (lower is better)
    """
    if isinstance(embeddings, torch.Tensor):
        E = embeddings.detach()
    else:
        E = torch.tensor(embeddings)

    if E.dim() != 2 or E.size(0) < 2:
        return {
            'var_mean': 0.0,
            'var_min': 0.0,
            'frac_below_thr': 1.0,
            'cov_offdiag_mean': 0.0,
        }

    # Center
    E_centered = E - E.mean(dim=0, keepdim=True)

    # Variance per dimension
    var = E_centered.var(dim=0, unbiased=False)
    var_mean = var.mean().item()
    var_min = var.min().item()
    frac_below = (var < variance_threshold).float().mean().item()

    # Covariance matrix
    cov = (E_centered.t() @ E_centered) / max(1, E_centered.size(0))
    offdiag = cov - torch.diag(torch.diag(cov))
    cov_offdiag_mean = offdiag.abs().mean().item() if offdiag.numel() > 0 else 0.0

    return {
        'var_mean': var_mean,
        'var_min': var_min,
        'frac_below_thr': frac_below,
        'cov_offdiag_mean': cov_offdiag_mean,
    }


_MORGAN_GEN = None  # global singleton to avoid per-sample construction


def compute_neighbor_consistency_spearman(model: torch.nn.Module,
                                          dataset,
                                          device: torch.device,
                                          sample_size: int = 128,
                                          fp_bits: int = 2048,
                                          fp_radius: int = 2,
                                          pool: str = 'mean') -> Dict[str, float]:
    """
    Compute rank correlation between latent similarities and RDKit Tanimoto(MorganFP).

    Notes:
    - Accesses dataset items directly to obtain `smiles` strings stored in Data.
    - Samples up to `sample_size` graphs for efficiency.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit import DataStructs
    except Exception:
        logger.warning("RDKit not available; skipping neighbor consistency metric.")
        return {'neighbor_spearman': 0.0}

    import random
    from scipy.stats import spearmanr

    n = len(dataset)
    if n == 0:
        return {'neighbor_spearman': 0.0}

    indices = list(range(n))
    random.shuffle(indices)
    indices = indices[: min(sample_size, n)]

    graph_embeds: List[torch.Tensor] = []
    fps = []

    model.eval()
    with torch.no_grad():
        for idx in indices:
            data = dataset[idx]
            smiles = getattr(data, 'smiles', None)
            if smiles is None:
                continue

            # Morgan fingerprint via new Generator API (singleton)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            global _MORGAN_GEN
            if _MORGAN_GEN is None:
                _MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_bits)
            fps.append(_MORGAN_GEN.GetFingerprint(mol))

            # Move to device and get graph embedding (mean pool of nodes)
            x = data.x.to(device)
            he_idx = data.hyperedge_index.to(device)
            he_attr = data.hyperedge_attr.to(device) if getattr(data, 'hyperedge_attr', None) is not None else None

            node_emb = model.get_embedding(x, he_idx, he_attr)
            if pool == 'mean':
                graph_emb = node_emb.mean(dim=0)
            elif pool == 'max':
                graph_emb, _ = node_emb.max(dim=0)
            else:
                graph_emb = node_emb.mean(dim=0)
            graph_embeds.append(graph_emb.detach().cpu())

    if len(graph_embeds) < 3:
        return {'neighbor_spearman': 0.0}

    E = torch.stack(graph_embeds, dim=0)
    E = torch.nn.functional.normalize(E, p=2, dim=1)
    sim_latent = (E @ E.t()).numpy()

    # Tanimoto similarities
    m = len(fps)
    sim_tani = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        # RDKit bulk similarity
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        sim_tani[i, :] = np.array(sims, dtype=np.float32)

    # Upper triangle (exclude diagonal)
    iu = np.triu_indices(min(sim_latent.shape[0], sim_tani.shape[0]), k=1)
    a = sim_latent[iu]
    b = sim_tani[iu]

    if a.size == 0 or b.size == 0:
        return {'neighbor_spearman': 0.0}

    rho, _ = spearmanr(a, b)
    if np.isnan(rho):
        rho = 0.0
    return {'neighbor_spearman': float(rho)}

class MovingAverage:
    """
    Compute moving average of metrics.
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize moving average.

        Args:
            window_size: Size of the moving window
        """
        self.window_size = window_size
        self.values = []

    def update(self, value: float):
        """Update with new value."""
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)

    def get(self) -> float:
        """Get current moving average."""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    def reset(self):
        """Reset values."""
        self.values = []


class MetricTracker:
    """
    Track multiple metrics over time.
    """

    def __init__(self, metrics: List[str], window_size: int = 100):
        """
        Initialize metric tracker.

        Args:
            metrics: List of metric names to track
            window_size: Window size for moving average
        """
        self.metrics = {name: MovingAverage(window_size) for name in metrics}
        self.history = {name: [] for name in metrics}

    def update(self, metric_dict: Dict[str, float]):
        """Update metrics."""
        for name, value in metric_dict.items():
            if name in self.metrics:
                self.metrics[name].update(value)
                self.history[name].append(value)

    def get_current(self) -> Dict[str, float]:
        """Get current moving averages."""
        return {name: ma.get() for name, ma in self.metrics.items()}

    def get_history(self) -> Dict[str, List[float]]:
        """Get full history."""
        return self.history

    def reset(self):
        """Reset all metrics."""
        for ma in self.metrics.values():
            ma.reset()
        self.history = {name: [] for name in self.metrics}

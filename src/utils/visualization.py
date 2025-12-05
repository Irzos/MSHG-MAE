"""
Visualization utilities for HyperGraph-MAE.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def setup_plotting_style():
    """Set up consistent plotting style."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10


def plot_training_history(history: Dict[str, List[float]],
                          save_path: Optional[Path] = None,
                          show: bool = True) -> plt.Figure:
    """
    Plot training history with losses and metrics.

    Args:
        history: Dictionary containing training history
        save_path: Optional path to save the figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    setup_plotting_style()

    # Determine number of subplots
    metrics = list(history.keys())
    n_metrics = len(metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (metric_name, values) in enumerate(history.items()):
        ax = axes[idx]

        # Plot metric
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, 'b-', label=metric_name)

        # Add trend line
        if len(values) > 5:
            z = np.polyfit(epochs, values, 1)
            p = np.poly1d(z)
            ax.plot(epochs, p(epochs), "r--", alpha=0.5, label='Trend')

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_title(f'{metric_name.replace("_", " ").title()} Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Remove empty subplots
    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_embeddings_2d(embeddings: Union[torch.Tensor, np.ndarray],
                       labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
                       method: str = 'pca',
                       save_path: Optional[Path] = None,
                       show: bool = True,
                       title: str = 'Embedding Visualization') -> plt.Figure:
    """
    Visualize embeddings in 2D using dimensionality reduction.

    Args:
        embeddings: Embeddings to visualize
        labels: Optional labels for coloring
        method: Dimensionality reduction method ('pca', 'tsne', 'umap')
        save_path: Optional path to save the figure
        show: Whether to display the plot
        title: Plot title

    Returns:
        Matplotlib figure
    """
    setup_plotting_style()

    # Convert to numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Reduce dimensions
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = reducer.fit_transform(embeddings)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    if labels is None:
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                             alpha=0.6, s=30)
    else:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

        for idx, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c=[colors[idx]], label=f'Class {label}',
                       alpha=0.6, s=30)

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_attention_weights(attention_weights: torch.Tensor,
                           node_labels: Optional[List[str]] = None,
                           save_path: Optional[Path] = None,
                           show: bool = True) -> plt.Figure:
    """
    Visualize attention weights as a heatmap.

    Args:
        attention_weights: Attention weight matrix
        node_labels: Optional labels for nodes
        save_path: Optional path to save the figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    setup_plotting_style()

    # Convert to numpy
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(attention_weights,
                cmap='YlOrRd',
                cbar_kws={'label': 'Attention Weight'},
                xticklabels=node_labels,
                yticklabels=node_labels,
                ax=ax)

    ax.set_title('Attention Weights Visualization')
    ax.set_xlabel('Target Nodes')
    ax.set_ylabel('Source Nodes')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def create_interactive_embedding_plot(embeddings: Union[torch.Tensor, np.ndarray],
                                      labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
                                      hover_data: Optional[pd.DataFrame] = None,
                                      method: str = 'pca',
                                      save_path: Optional[Path] = None) -> go.Figure:
    """
    Create interactive embedding visualization using Plotly.

    Args:
        embeddings: Embeddings to visualize
        labels: Optional labels for coloring
        hover_data: Optional DataFrame with hover information
        method: Dimensionality reduction method
        save_path: Optional path to save the figure

    Returns:
        Plotly figure
    """
    # Convert to numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Reduce dimensions
    if embeddings.shape[1] > 3:
        if method == 'pca':
            reducer = PCA(n_components=3, random_state=42)
            embeddings_3d = reducer.fit_transform(embeddings)
        elif method == 'tsne':
            reducer = TSNE(n_components=3, random_state=42)
            embeddings_3d = reducer.fit_transform(embeddings)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=3, random_state=42)
            embeddings_3d = reducer.fit_transform(embeddings)
    else:
        embeddings_3d = embeddings

    # Create DataFrame for plotting
    df = pd.DataFrame(embeddings_3d[:, :3], columns=['X', 'Y', 'Z'])

    if labels is not None:
        df['Label'] = labels
        color_col = 'Label'
    else:
        color_col = None

    if hover_data is not None:
        df = pd.concat([df, hover_data], axis=1)

    # Create 3D scatter plot
    fig = px.scatter_3d(df, x='X', y='Y', z='Z',
                        color=color_col,
                        hover_data=df.columns,
                        title=f'3D Embedding Visualization ({method.upper()})')

    fig.update_traces(marker=dict(size=5, opacity=0.8))

    fig.update_layout(
        scene=dict(
            xaxis_title=f'{method.upper()} 1',
            yaxis_title=f'{method.upper()} 2',
            zaxis_title=f'{method.upper()} 3'
        ),
        width=900,
        height=700
    )

    if save_path:
        fig.write_html(str(save_path))

    return fig


def plot_hypergraph_statistics(dataset_stats: Dict[str, Union[List, float]],
                               save_path: Optional[Path] = None,
                               show: bool = True) -> plt.Figure:
    """
    Plot dataset statistics for hypergraphs.

    Args:
        dataset_stats: Dictionary containing dataset statistics
        save_path: Optional path to save the figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    setup_plotting_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Plot 1: Node distribution
    if 'num_nodes' in dataset_stats:
        ax = axes[0]
        ax.hist(dataset_stats['num_nodes'], bins=50, alpha=0.7, color='blue')
        ax.axvline(np.mean(dataset_stats['num_nodes']), color='red',
                   linestyle='--', label=f'Mean: {np.mean(dataset_stats["num_nodes"]):.1f}')
        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Count')
        ax.set_title('Node Distribution')
        ax.legend()

    # Plot 2: Edge distribution
    if 'num_edges' in dataset_stats:
        ax = axes[1]
        ax.hist(dataset_stats['num_edges'], bins=50, alpha=0.7, color='green')
        ax.axvline(np.mean(dataset_stats['num_edges']), color='red',
                   linestyle='--', label=f'Mean: {np.mean(dataset_stats["num_edges"]):.1f}')
        ax.set_xlabel('Number of Hyperedges')
        ax.set_ylabel('Count')
        ax.set_title('Hyperedge Distribution')
        ax.legend()

    # Plot 3: Node-Edge ratio
    if 'num_nodes' in dataset_stats and 'num_edges' in dataset_stats:
        ax = axes[2]
        ratios = [n / e if e > 0 else 0 for n, e in
                  zip(dataset_stats['num_nodes'], dataset_stats['num_edges'])]
        ax.hist(ratios, bins=50, alpha=0.7, color='orange')
        ax.set_xlabel('Node/Edge Ratio')
        ax.set_ylabel('Count')
        ax.set_title('Node-to-Hyperedge Ratio Distribution')

    # Plot 4: Summary statistics
    ax = axes[3]
    ax.axis('off')

    summary_text = "Dataset Summary\n" + "=" * 30 + "\n"
    summary_text += f"Total graphs: {dataset_stats.get('num_graphs', 'N/A')}\n"

    if 'num_nodes' in dataset_stats:
        summary_text += f"Avg nodes: {np.mean(dataset_stats['num_nodes']):.1f} ± {np.std(dataset_stats['num_nodes']):.1f}\n"
        summary_text += f"Min/Max nodes: {min(dataset_stats['num_nodes'])}/{max(dataset_stats['num_nodes'])}\n"

    if 'num_edges' in dataset_stats:
        summary_text += f"Avg edges: {np.mean(dataset_stats['num_edges']):.1f} ± {np.std(dataset_stats['num_edges']):.1f}\n"
        summary_text += f"Min/Max edges: {min(dataset_stats['num_edges'])}/{max(dataset_stats['num_edges'])}\n"

    if 'feature_dim' in dataset_stats:
        summary_text += f"Feature dimension: {dataset_stats['feature_dim']}\n"

    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_loss_components(history: Dict[str, List[float]],
                         save_path: Optional[Path] = None,
                         show: bool = True) -> plt.Figure:
    """
    Plot different loss components over training.

    Args:
        history: Dictionary containing loss components
        save_path: Optional path to save the figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    setup_plotting_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each loss component
    loss_components = ['recon_loss', 'edge_loss', 'total_loss']
    colors = ['blue', 'green', 'red']

    for loss_name, color in zip(loss_components, colors):
        if loss_name in history:
            epochs = range(1, len(history[loss_name]) + 1)
            ax.plot(epochs, history[loss_name], label=loss_name.replace('_', ' ').title(),
                    color=color, linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log scale option
    ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def create_training_dashboard(history: Dict[str, List[float]],
                              save_path: Optional[Path] = None) -> go.Figure:
    """
    Create an interactive training dashboard using Plotly.

    Args:
        history: Training history dictionary
        save_path: Optional path to save the dashboard

    Returns:
        Plotly figure
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training Loss', 'Validation Loss',
                        'Learning Rate', 'Loss Components'),
        specs=[[{'secondary_y': False}, {'secondary_y': False}],
               [{'secondary_y': False}, {'secondary_y': False}]]
    )

    epochs = list(range(1, len(history.get('train_loss', [])) + 1))

    # Training loss
    if 'train_loss' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['train_loss'],
                       mode='lines', name='Train Loss'),
            row=1, col=1
        )

    # Validation loss
    if 'val_loss' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_loss'],
                       mode='lines', name='Val Loss'),
            row=1, col=2
        )

    # Learning rate
    if 'learning_rate' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['learning_rate'],
                       mode='lines', name='Learning Rate'),
            row=2, col=1
        )

    # Loss components
    loss_components = ['recon_loss', 'edge_loss']
    for component in loss_components:
        if component in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history[component],
                           mode='lines', name=component.replace('_', ' ').title()),
                row=2, col=2
            )

    # Update layout
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss", row=1)
    fig.update_yaxes(title_text="Learning Rate", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=2, col=2)

    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Training Dashboard"
    )

    if save_path:
        fig.write_html(str(save_path))

    return fig
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

# Helper for denormalization

def denormalize(tensor, mean, std):
    """Denormalize tensor using mean and std."""
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean

def plot_class_transitions(x0, xk_tensor, deltas_tensor, mean, std, class_names=None, n_points=10, figsize=(16, 4)):
    """
    Plot the transition of the input image to each class along the geodesic, with 10 equally spaced images per class.
    Below each image, plot the corresponding delta (step direction).
    Args:
        x0: original input tensor [1, C, H, W] or [C, H, W]
        xk_tensor: tensor [num_target_classes, C, H, W] (final point for each class trajectory)
        deltas_tensor: tensor [num_target_classes, max_steps, C, H, W] (delta tensors per class, per step)
        mean, std: normalization params for denormalization
        class_names: list of class names (optional)
        n_points: number of images to show per class
        figsize: tuple for figure size
    """
    n_classes = xk_tensor.shape[0]
    x0 = x0.squeeze().detach().cpu()
    for class_idx in range(n_classes):
        xk = xk_tensor[class_idx]
        deltas = deltas_tensor[class_idx]
        # Stack trajectory: reconstruct all intermediate points
        x_traj = [x0]
        # Filter out zero deltas (padding)
        non_zero_deltas = []
        for i in range(deltas.shape[0]):
            if deltas[i].abs().sum() > 1e-8:  # Check if delta is non-zero
                non_zero_deltas.append(deltas[i])
        
        for delta in non_zero_deltas:
            x_traj.append(x_traj[-1] + delta)
        x_traj = torch.stack(x_traj)
        # Select n_points equally spaced indices
        idxs = np.linspace(0, len(x_traj)-1, n_points, dtype=int)
        fig, axes = plt.subplots(2, n_points, figsize=(figsize[0], figsize[1]*2))
        for i, idx in enumerate(idxs):
            # Image
            img = denormalize(x_traj[idx], mean, std).clamp(0, 1)
            img_np = img.numpy().transpose(1, 2, 0)
            axes[0, i].imshow(img_np)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Start')
            elif i == n_points-1:
                axes[0, i].set_title('End')
            else:
                axes[0, i].set_title(f'Step {idx}')
            # Delta
            if idx > 0:
                delta = x_traj[idx] - x_traj[idx-1]
                delta_np = delta.numpy().transpose(1, 2, 0)
                # Normalize delta for visualization
                delta_vis = np.abs(delta_np)
                delta_vis = delta_vis / (delta_vis.max() + 1e-8)
                axes[1, i].imshow(delta_vis)
            else:
                axes[1, i].imshow(np.zeros_like(img_np))
            axes[1, i].axis('off')
        class_label = class_names[class_idx] if class_names is not None else f'Class {class_idx}'
        plt.suptitle(f'Trajectory to {class_label}')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def plot_cumulative_distance_evolution(alphas_list, class_names=None, figsize=(8, 5)):
    """
    Plot the cumulative geodesic distance evolution for each class trajectory.
    Args:
        alphas_list: list of lists of alpha floats (per class, per step)
        class_names: list of class names (optional)
        figsize: tuple for figure size
    """
    plt.figure(figsize=figsize)
    for class_idx, alphas in enumerate(alphas_list):
        alphas_np = np.array(alphas)  # alphas are already floats
        cum_dist = np.cumsum(alphas_np)
        label = class_names[class_idx] if class_names is not None else f'Class {class_idx}'
        plt.plot(cum_dist, label=label)
    plt.xlabel('Step')
    plt.ylabel('Cumulative Geodesic Distance')
    plt.title('Cumulative Geodesic Distance Evolution')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_fisher_margins(margins, class_names=None, figsize=(8, 5)):
    """
    Plot the minimal Fisher-geodesic margin for each class.
    Args:
        margins: list of floats (per class)
        class_names: list of class names (optional)
        figsize: tuple for figure size
    """
    margins_np = np.array(margins)  # margins are already floats
    x = np.arange(len(margins_np))
    plt.figure(figsize=figsize)
    plt.bar(x, margins_np, tick_label=class_names if class_names is not None else x)
    plt.xlabel('Class')
    plt.ylabel('Minimal Fisher-Geodesic Margin')
    plt.title('Minimal Fisher Margin per Class')
    plt.tight_layout()
    plt.show()

def plot_class_probability_trajectories(probs_seq, orig_label, target_label, class_names=None, reached_label=None, pred_k_seq=None, alphas=None, figsize=(8, 7)):
    """
    Plot class-probability trajectories along a geodesic flow for the top-3 relevant classes and the top-5 contenders at the first step.
    Annotate the crossing point where the prediction changes.
    Plot the raw Fisher-norms (alphas) as a secondary subplot.
    Always show both step-index and cumulative geodesic distance as x-axes (requires alphas).
    Args:
        probs_seq: list of 1D tensors/arrays (length K), probabilities at each step
        orig_label: int, original class index
        target_label: int, target class index
        class_names: list of class names (optional)
        reached_label: int, class index actually reached at the end (optional)
        pred_k_seq: list of predicted class indices at each step (optional)
        alphas: list of raw Fisher-norms (required)
        figsize: tuple for figure size
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Convert to numpy arrays
    probs_seq_np = [p.numpy() if hasattr(p, 'numpy') else np.array(p) for p in probs_seq]
    n_steps = len(probs_seq_np)

    assert len(alphas) == n_steps - 1, "alphas should have length n_steps-1 (steps between points)"

    # Determine reached_label if not provided
    if reached_label is None:
        reached_label = int(np.argmax(probs_seq_np[-1]))

    # Extract probability trajectories for main classes
    p_orig = [float(p[orig_label]) for p in probs_seq_np]
    p_target = [float(p[target_label]) for p in probs_seq_np]
    p_reached = [float(p[reached_label]) for p in probs_seq_np]

    # Prepare labels
    def class_label(idx):
        if class_names is not None:
            return f"{class_names[idx]} ({idx})"
        else:
            return f"Class {idx}"

    # X-axis: step index
    xvals_steps = np.arange(n_steps)
    # X-axis: cumulative geodesic distance
    cum_dist = np.cumsum([float(a) for a in alphas])
    xvals_dist = np.concatenate([[0.0], cum_dist])

    def plot_panel(ax1, ax2, xvals, x_label, panel_title):
        # Main curves
        ax1.plot(xvals, p_orig, label=f'Original: {class_label(orig_label)}', linewidth=2)
        ax1.plot(xvals, p_target, label=f'Target: {class_label(target_label)}', linewidth=2)
        if reached_label != orig_label and reached_label != target_label:
            ax1.plot(xvals, p_reached, label=f'Reached: {class_label(reached_label)}', linestyle='--', linewidth=2)
        # Top-5 contenders at k=0 (excluding already plotted classes)
        first_probs = probs_seq_np[0]
        all_main = {orig_label, target_label, reached_label}
        topk = np.argsort(first_probs)[-5:][::-1]
        for idx in topk:
            if idx in all_main:
                continue
            p_curve = [float(p[idx]) for p in probs_seq_np]
            ax1.plot(xvals, p_curve, label=f'Top contender: {class_label(idx)}', linestyle=':', linewidth=1)
        # Annotate crossing point if pred_k_seq is provided
        if pred_k_seq is not None:
            try:
                cross_k = next(i for i, p in enumerate(pred_k_seq) if p != orig_label)
                cross_x = xvals[cross_k]
                ax1.axvline(cross_x, color='grey', linestyle='--', label='Decision flip')
                ax1.scatter(cross_x, p_reached[cross_k], marker='o', color='red', zorder=10, label='Crossing point')
            except StopIteration:
                pass  # No crossing found
        ax1.set_ylabel('Class probability')
        ax1.set_title(panel_title)
        ax1.legend()
        ax1.grid(True)
        # Plot alphas
        alphas_np = [float(a) for a in alphas]
        ax2.plot(xvals[1:], alphas_np, label=r'Raw Fisher-norm $\alpha_k$', color='tab:purple')
        ax2.set_ylabel(r'Raw Fisher-norm $\alpha_k$')
        ax2.set_xlabel(x_label)
        ax2.legend()
        ax2.grid(True)

    # Plot with respect to step index
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    plot_panel(ax1, ax2, xvals_steps, 'Step index (k)', 'Class-Probability Trajectories (vs. Step Index)')
    plt.tight_layout()
    plt.show()

    # Plot with respect to cumulative geodesic distance
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    plot_panel(ax1, ax2, xvals_dist, 'Cumulative geodesic distance $s_k$', 'Class-Probability Trajectories (vs. Geodesic Distance)')
    plt.tight_layout()
    plt.show()
    # plt.savefig('class_probability_trajectories.png')


def plot_class_probabilities_twin_axes(probs_seq_list, pred_k_list, alphas_list, 
                                     margin_losses_list, orig_label, target_classes, 
                                     class_names=None, figsize=(25, 10), output_dir=None):
    """
    Create figure 1: Class probability trajectories with twin x-axis (step + distance) and twin y-axis (probs + margin loss)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    n_targets = len(target_classes)
    n_cols = 5
    n_rows = 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, target_label in enumerate(target_classes):
        if i >= n_targets:
            axes[i].set_visible(False)
            continue
            
        # Get data for this class
        probs_seq = probs_seq_list[i]
        pred_k_seq = pred_k_list[i]
        alphas = alphas_list[i]
        margin_losses = margin_losses_list[i]
        
        # Convert to numpy arrays
        probs_seq_np = [p.numpy() if hasattr(p, 'numpy') else np.array(p) for p in probs_seq]
        n_steps = len(probs_seq_np)
        
        # Determine reached_label
        reached_label = int(np.argmax(probs_seq_np[-1]))
        
        # Extract probability trajectories
        p_orig = [float(p[orig_label]) for p in probs_seq_np]
        p_target = [float(p[target_label]) for p in probs_seq_np]
        p_reached = [float(p[reached_label]) for p in probs_seq_np]
        
        # Top-5 contenders at k=0 (excluding already plotted classes)
        first_probs = probs_seq_np[0]
        all_main = {orig_label, target_label, reached_label}
        topk = np.argsort(first_probs)[-5:][::-1]
        
        # X-axes
        xvals_steps = np.arange(n_steps)
        cum_dist = np.cumsum([float(a) for a in alphas])
        xvals_dist = np.concatenate([[0.0], cum_dist])
        
        # Plot class probability trajectories with twin axes
        ax = axes[i]
        
        # Primary x-axis (step index) and y-axis (probabilities)
        ax.plot(xvals_steps, p_orig, label=f'Orig ({orig_label})', linewidth=2, color='blue')
        ax.plot(xvals_steps, p_target, label=f'Target ({target_label})', linewidth=2, color='orange')
        if reached_label != orig_label and reached_label != target_label:
            ax.plot(xvals_steps, p_reached, label=f'Reached ({reached_label})', linestyle='--', linewidth=2, color='green')
        for idx in topk:
            if idx in all_main:
                continue
            p_curve = [float(p[idx]) for p in probs_seq_np]
            ax.plot(xvals_steps, p_curve, label=f'Top {idx}', linestyle=':', linewidth=1, alpha=0.7)
        
        # Secondary x-axis (geodesic distance)
        ax_twinx = ax.twinx()
        ax_twinx.set_xlim(ax.get_xlim())
        ax_twinx.set_xticks(xvals_steps)
        ax_twinx.set_xticklabels([f'{d:.2f}' for d in xvals_dist], rotation=45, fontsize=8)
        ax_twinx.set_xlabel('Geodesic Distance', color='red')
        ax_twinx.tick_params(axis='x', labelcolor='red')
        
        # Secondary y-axis (margin loss)
        ax_twiny = ax.twinx()
        ax_twiny.spines['right'].set_position(('outward', 60))
        ax_twiny.plot(xvals_steps, margin_losses, color='red', label='Margin loss', marker='o', markersize=3, linewidth=1.5, alpha=0.7)
        ax_twiny.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Decision boundary')
        ax_twiny.set_ylabel('Margin Loss', color='red')
        ax_twiny.tick_params(axis='y', labelcolor='red')
        
        # Annotate crossing point
        try:
            cross_k = next(j for j, p in enumerate(pred_k_seq) if p != orig_label)
            ax.axvline(cross_k, color='grey', linestyle='--', alpha=0.7, label='Decision flip')
            ax.scatter(cross_k, p_reached[cross_k], marker='o', color='red', zorder=10, s=50, label='Crossing')
        except StopIteration:
            pass
        
        if class_names is not None:
            title = f"{class_names[target_label]} ({target_label})"
        else:
            title = f"Class {target_label}"
        ax.set_title(title)
        ax.set_xlabel('Step Index')
        ax.set_ylabel('Class Probability', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        if i == 0:
            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_twiny.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Class Probability Trajectories with Twin Axes (Original: {orig_label})', fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f'class_probabilities_twin_axes_orig_{orig_label}.png'), dpi=300, bbox_inches='tight')
    else:
        fig.show()
        fig.savefig(f'class_probabilities_twin_axes_orig_{orig_label}.png', dpi=300, bbox_inches='tight')
    
    return fig


def plot_margin_loss_twin_x(margin_losses_list, pred_k_list, alphas_list,
                           orig_label, target_classes, class_names=None, figsize=(25, 10), output_dir=None):
    """
    Create figure 2: Margin loss evolution with twin x-axis (step + distance)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    n_targets = len(target_classes)
    n_cols = 5
    n_rows = 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, target_label in enumerate(target_classes):
        if i >= n_targets:
            axes[i].set_visible(False)
            continue
            
        # Get data for this class
        pred_k_seq = pred_k_list[i]
        alphas = alphas_list[i]
        margin_losses = margin_losses_list[i]
        
        # X-axes
        xvals_steps = np.arange(len(margin_losses))
        cum_dist = np.cumsum([float(a) for a in alphas])
        xvals_dist = np.concatenate([[0.0], cum_dist])
        
        # Plot margin loss evolution with twin x-axis
        ax = axes[i]
        
        # Primary x-axis (step index) and y-axis (margin loss)
        ax.plot(xvals_steps, margin_losses, color='red', label='Margin loss', marker='o', markersize=4, linewidth=2)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Decision boundary')
        
        # Secondary x-axis (geodesic distance)
        ax_twinx = ax.twinx()
        ax_twinx.set_xlim(ax.get_xlim())
        ax_twinx.set_xticks(xvals_steps)
        ax_twinx.set_xticklabels([f'{d:.2f}' for d in xvals_dist], rotation=45, fontsize=8)
        ax_twinx.set_xlabel('Geodesic Distance', color='blue')
        ax_twinx.tick_params(axis='x', labelcolor='blue')
        
        # Annotate crossing point
        try:
            cross_k = next(j for j, p in enumerate(pred_k_seq) if p != orig_label)
            ax.axvline(cross_k, color='grey', linestyle='--', alpha=0.7, label='Decision flip')
            ax.scatter(cross_k, margin_losses[cross_k], marker='o', color='red', zorder=10, s=50, label='Crossing')
        except StopIteration:
            pass
        
        if class_names is not None:
            title = f"{class_names[target_label]} ({target_label})"
        else:
            title = f"Class {target_label}"
        ax.set_title(title)
        ax.set_xlabel('Step Index')
        ax.set_ylabel('Margin Loss', color='red')
        ax.tick_params(axis='y', labelcolor='red')
        if i == 0:
            ax.legend(loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Margin Loss Evolution with Twin X-Axis (Original: {orig_label})', fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f'margin_loss_twin_x_orig_{orig_label}.png'), dpi=300, bbox_inches='tight')
    else:
        fig.show()
        fig.savefig(f'margin_loss_twin_x_orig_{orig_label}.png', dpi=300, bbox_inches='tight')
    
    return fig


def plot_distance_matrix_summary(margins, probs_seq_list, orig_label, target_classes, 
                               class_names=None, figsize=(20, 8), output_dir=None):
    """
    Create figure 3: Distance matrix/heatmap showing geodesic distances and reached classes
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Prepare data for distance matrix
    distances = [float(m) for m in margins]
    reached_classes = []
    for i, target_label in enumerate(target_classes):
        probs_seq = probs_seq_list[i]
        probs_seq_np = [p.numpy() if hasattr(p, 'numpy') else np.array(p) for p in probs_seq]
        reached_label = int(np.argmax(probs_seq_np[-1]))
        reached_classes.append(reached_label)
    
    # Create labels
    if class_names is not None:
        target_labels = [f"{class_names[t]} ({t})" for t in target_classes]
        reached_labels = [f"{class_names[r]} ({r})" for r in reached_classes]
    else:
        target_labels = [f"Class {t}" for t in target_classes]
        reached_labels = [f"Class {r}" for r in reached_classes]
    
    # Bar chart of distances with simple color coding
    bar_colors = []
    for i, (dist, reached, target) in enumerate(zip(distances, reached_classes, target_classes)):
        # Color based on success
        if reached == target:
            bar_colors.append('green')
        else:
            bar_colors.append('red')
    
    bars = ax1.bar(range(len(distances)), distances, 
                   color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for i, (bar, dist, reached, target) in enumerate(zip(bars, distances, reached_classes, target_classes)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(distances)*0.01,
                 f'{dist:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        # Add reached class annotation
        if reached != target:
            ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                     f'→{reached}', ha='center', va='center', fontsize=8, 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    ax1.set_xlabel('Target Class')
    ax1.set_ylabel('Geodesic Distance')
    ax1.set_title(f'Geodesic Distances to Target Classes (Original: {orig_label})')
    ax1.set_xticks(range(len(target_classes)))
    ax1.set_xticklabels([f'{t}' for t in target_classes], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add simple legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Target reached'),
        Patch(facecolor='red', alpha=0.7, label='Different class reached'),
        Patch(facecolor='yellow', alpha=0.7, label='Reached class annotation')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Heatmap of distances
    heatmap_data = np.zeros((len(target_classes), 2))
    heatmap_data[:, 0] = distances  # Distance column
    heatmap_data[:, 1] = reached_classes  # Reached class column
    
    im = ax2.imshow(heatmap_data, cmap='viridis', aspect='auto')
    
    # Add text annotations
    for i in range(len(target_classes)):
        for j in range(2):
            if j == 0:  # Distance
                text = f'{distances[i]:.3f}'
                color = 'white' if distances[i] > np.mean(distances) else 'black'
            else:  # Reached class
                text = f'{reached_classes[i]}'
                color = 'red' if reached_classes[i] != target_classes[i] else 'green'
            ax2.text(j, i, text, ha="center", va="center", color=color, fontweight='bold')
    
    ax2.set_xlabel('Metric')
    ax2.set_ylabel('Target Class')
    ax2.set_title('Distance Matrix Heatmap')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Distance', 'Reached Class'])
    ax2.set_yticks(range(len(target_classes)))
    ax2.set_yticklabels([f'{t}' for t in target_classes])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Value')
    
    fig.suptitle(f'Distance Matrix Summary (Original: {orig_label})', fontsize=16)
    fig.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f'distance_matrix_summary_orig_{orig_label}.png'), dpi=300, bbox_inches='tight')
    else:
        fig.show()
        fig.savefig(f'distance_matrix_summary_orig_{orig_label}.png', dpi=300, bbox_inches='tight')
    
    return fig


def plot_distance_distribution(margins, orig_label, target_classes, class_names=None, figsize=(20, 8), output_dir=None):
    """
    Create figure 4: Distribution of Geodesic Distances (bar chart and box plot)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    distances = [float(m) for m in margins]
    
    # Bar chart in rank order
    sorted_distances = sorted(distances, reverse=True)
    sorted_targets = [target_classes[i] for i in np.argsort(distances)[::-1]]
    
    bars = ax1.bar(range(len(sorted_distances)), sorted_distances, 
                   color='skyblue', alpha=0.7, edgecolor='navy', linewidth=1)
    
    # Add value labels
    for i, (bar, dist) in enumerate(zip(bars, sorted_distances)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(sorted_distances)*0.01,
                 f'{dist:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        # Add rank labels
        ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                 f'#{i+1}', ha='center', va='center', fontsize=8, 
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    ax1.set_xlabel('Target Class (Ranked by Distance)')
    ax1.set_ylabel('Geodesic Distance')
    ax1.set_title('Geodesic Distances Ranked (Largest to Smallest)')
    ax1.set_xticks(range(len(sorted_targets)))
    ax1.set_xticklabels([f'{t}' for t in sorted_targets], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(distances, patch_artist=True, 
                boxprops=dict(facecolor='lightgreen', alpha=0.7),
                medianprops=dict(color='red', linewidth=2),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=6))
    
    # Add statistics
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)
    std_dist = np.std(distances)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    stats_text = f'Mean: {mean_dist:.3f}\nMedian: {median_dist:.3f}\nStd: {std_dist:.3f}\nMin: {min_dist:.3f}\nMax: {max_dist:.3f}'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    ax2.set_ylabel('Geodesic Distance')
    ax2.set_title('Distribution of Geodesic Distances')
    ax2.set_xticklabels(['All Targets'])
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'Distance Distribution Analysis (Original: {orig_label})', fontsize=16)
    fig.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f'distance_distribution_orig_{orig_label}.png'), dpi=300, bbox_inches='tight')
    else:
        fig.show()
        fig.savefig(f'distance_distribution_orig_{orig_label}.png', dpi=300, bbox_inches='tight')
    
    return fig


def plot_fisher_length_analysis(alphas_list, orig_label, target_classes, class_names=None, figsize=(20, 8), output_dir=None):
    """
    Create figure 5: Histogram of Per-Step Fisher-Lengths (alpha values)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Pool all alpha values
    all_alphas = []
    for alphas in alphas_list:
        all_alphas.extend([float(a) for a in alphas])
    
    # Overall histogram
    ax1.hist(all_alphas, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax1.set_xlabel('Fisher-Length (α)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of All Fisher-Lengths')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mean_alpha = np.mean(all_alphas)
    median_alpha = np.median(all_alphas)
    std_alpha = np.std(all_alphas)
    stats_text = f'Mean: {mean_alpha:.3f}\nMedian: {median_alpha:.3f}\nStd: {std_alpha:.3f}\nMin: {min(all_alphas):.3f}\nMax: {max(all_alphas):.3f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Per-class alpha distributions
    for i, (target_label, alphas) in enumerate(zip(target_classes, alphas_list)):
        alpha_vals = [float(a) for a in alphas]
        ax2.hist(alpha_vals, bins=15, alpha=0.5, label=f'Class {target_label}', density=True)
    
    ax2.set_xlabel('Fisher-Length (α)')
    ax2.set_ylabel('Density')
    ax2.set_title('Fisher-Length Distribution by Target Class')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'Fisher-Length Analysis (Original: {orig_label})', fontsize=16)
    fig.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f'fisher_length_analysis_orig_{orig_label}.png'), dpi=300, bbox_inches='tight')
    else:
        fig.show()
        fig.savefig(f'fisher_length_analysis_orig_{orig_label}.png', dpi=300, bbox_inches='tight')
    
    return fig


def plot_angular_change_analysis(deltas_tensor, orig_label, target_classes, class_names=None, figsize=(20, 8), output_dir=None):
    """
    Create figure 6: Angular Change Between Successive Steps
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Compute angles for each target class
    all_angles = []
    angle_data = []
    
    for j, target_label in enumerate(target_classes):
        deltas = deltas_tensor[j]
        # Filter out zero deltas (padding)
        non_zero_deltas = []
        for i in range(deltas.shape[0]):
            if deltas[i].abs().sum() > 1e-8:  # Check if delta is non-zero
                non_zero_deltas.append(deltas[i])
        
        if len(non_zero_deltas) < 2:  # Need at least 2 steps to compute angles
            continue
            
        # Convert deltas to flattened vectors
        vecs = np.array([d.numpy().flatten() for d in non_zero_deltas])
        
        # Compute angles between successive steps
        cosines = np.sum(vecs[:-1] * vecs[1:], axis=1) / (
                   np.linalg.norm(vecs[:-1], axis=1) * np.linalg.norm(vecs[1:], axis=1))
        angles = np.arccos(np.clip(cosines, -1, 1))
        angles_deg = np.degrees(angles)
        
        all_angles.extend(angles_deg)
        angle_data.append((target_label, angles_deg))
        
        # Plot angles vs step index for this class
        steps = range(1, len(angles_deg) + 1)
        ax1.plot(steps, angles_deg, marker='o', label=f'Class {target_label}', linewidth=2, markersize=4)
    
    ax1.set_xlabel('Step Index')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('Angular Change Between Successive Steps')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add horizontal line at 90 degrees (perpendicular)
    ax1.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Perpendicular (90°)')
    
    # Overall angle distribution
    ax2.hist(all_angles, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Angular Changes')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    mean_angle = np.mean(all_angles)
    median_angle = np.median(all_angles)
    std_angle = np.std(all_angles)
    stats_text = f'Mean: {mean_angle:.1f}°\nMedian: {median_angle:.1f}°\nStd: {std_angle:.1f}°\nMin: {min(all_angles):.1f}°\nMax: {max(all_angles):.1f}°'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Add vertical line at mean
    ax2.axvline(mean_angle, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_angle:.1f}°')
    ax2.legend()
    
    fig.suptitle(f'Angular Change Analysis (Original: {orig_label})', fontsize=16)
    fig.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f'angular_change_analysis_orig_{orig_label}.png'), dpi=300, bbox_inches='tight')
    else:
        fig.show()
        fig.savefig(f'angular_change_analysis_orig_{orig_label}.png', dpi=300, bbox_inches='tight')
    
    return fig


def plot_fisher_margin_figures(
    margins,
    xk_tensor,
    deltas_tensor,
    alphas_list,
    probs_seq_list,
    pred_k_list,
    margin_losses_list,
    orig_label,
    target_classes,
    class_names=None,
    figsize=(25, 10),
    output_dir=None
):
    """
    Main function that creates all six figures for geodesic analysis.
    This function calls the individual plotting functions for each figure type.
    
    Args:
        output_dir: Directory to save all plots. If None, plots are shown and saved in current directory.
    """
    # Create all figures
    fig1 = plot_class_probabilities_twin_axes(
        probs_seq_list, pred_k_list, alphas_list, 
        margin_losses_list, orig_label, target_classes, class_names, figsize, output_dir
    )
    
    fig2 = plot_margin_loss_twin_x(
        margin_losses_list, pred_k_list, alphas_list,
        orig_label, target_classes, class_names, figsize, output_dir
    )
    
    fig3 = plot_distance_matrix_summary(
        margins, probs_seq_list, orig_label, target_classes, class_names, figsize, output_dir
    )
    
    fig4 = plot_distance_distribution(
        margins, orig_label, target_classes, class_names, figsize, output_dir
    )
    
    fig5 = plot_fisher_length_analysis(
        alphas_list, orig_label, target_classes, class_names, figsize, output_dir
    )
    
    fig6 = plot_angular_change_analysis(
        deltas_tensor, orig_label, target_classes, class_names, figsize, output_dir
    )
    
    if output_dir is None:
        plt.show()
    
    return fig1, fig2, fig3, fig4, fig5, fig6
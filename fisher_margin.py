import torch
import torch.nn.functional as F
import os
import logging
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from tqdm import tqdm

from linalg import CG
from visualization import plot_fisher_margin_figures
from config import ExperimentConfig

# Set up logging
def setup_logger(name: str = 'fisher_margin', level: int = logging.INFO, log_file: str = None) -> logging.Logger:
    """
    Set up a logger with file handler only (no console output).
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # File handler (required)
    if log_file:
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create logger instance with default file logging
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/fisher_margin_{timestamp}.log"
logger = setup_logger('fisher_margin', logging.DEBUG, log_file)

class FisherMargin:
    def __init__(self, model, config: ExperimentConfig):
        """
        Args:
            model (torch.nn.Module): Target classifier.
            config (ExperimentConfig): Configuration object with all hyperparameters.
        """
        self.model = model.eval()
        self.config = config
        logger.info(f"FisherMargin initialized with config: {config}")

    def _fisher_vector_product(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        v: torch.Tensor,
        sample_size: int
    ) -> torch.Tensor:
        """
        Monte Carlo estimate of G(x) · v, where G(x) is the Fisher Information Matrix:
            G(x) = E_y [ ∇_x log p(y|x) ⊗ ∇_x log p(y|x) ].

        Args:
            model: classification network.
            x:     input tensor, shape [C,H,W], requires_grad=True.
            v:     same shape as x.
            sample_size: number of label samples for MC.

        Returns:
            Tensor same shape as x: G(x) · v.
        """
        logger.debug(f"Computing Fisher vector product with sample_size={sample_size}")
        
        # forward + probabilities
        logits = model(x.unsqueeze(0))              # x.unsqueeze(0) shape: [1, C, H, W] -> logits shape: [1, K]
        probs  = F.softmax(logits, dim=1).squeeze(0) # probs shape: [K]
        logp   = F.log_softmax(logits, dim=1).squeeze(0)  # logp shape: [K]

        # Fix numerical instability: ensure probabilities are valid
        probs = torch.clamp(probs, min=1e-10)  # Ensure no negative values
        probs = probs / probs.sum()  # Renormalize to sum to 1

        # sample labels
        y_samples = torch.multinomial(probs, num_samples=sample_size, replacement=True)  # y_samples shape: [sample_size]

        gv = torch.zeros_like(x)  # gv shape: [C, H, W]
        for y in y_samples:
            # compute loss_j = -log p(y|x)
            loss_j = -logp[y]  # loss_j shape: scalar
            # gradient ∇_x loss_j
            grads  = torch.autograd.grad(loss_j, x, retain_graph=True)[0]  # grads shape: [C, H, W]
            # inner product <grads, v>
            coeff  = (grads * v).view(-1).sum()  # coeff shape: scalar
            gv    += coeff * grads  # gv shape: [C, H, W]

        result = gv.div_(float(sample_size))  # gv shape: [C, H, W]
        logger.info(f"Fisher vector product computed, result norm: {result.norm().item():.6f}")
        return result
    
    def _fisher_vector_product_mc(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,                   # [C,H,W], requires_grad=True
        v: torch.Tensor,                   # [C,H,W]
        sample_size: int
    ) -> torch.Tensor:
        """
        Vectorized Monte Carlo estimate of F(x)·v where
            F(x) = E_y [ ∇_x log p(y|x) ⊗ ∇_x log p(y|x) ].
        """
        # 1) forward + probabilities
        logits = model(x.unsqueeze(0))               # [1, K]
        probs  = F.softmax(logits, dim=1).squeeze(0)  # [K]
        K = logits.size(1)

        # Fix numerical instability: ensure probabilities are valid
        probs = torch.clamp(probs, min=1e-10)  # Ensure no negative values
        probs = probs / probs.sum()  # Renormalize to sum to 1

        # 2) sample labels
        y_samples = torch.multinomial(probs, sample_size, replacement=True)  # [S]
        freq = torch.bincount(y_samples, minlength=K).float()                # [K]

        # 3) compute per-class gradients G_k = ∇_x log p(y=k|x)
        def single_logp(inp: torch.Tensor) -> torch.Tensor:
            # returns [K] vector of log-probabilities
            return F.log_softmax(model(inp.unsqueeze(0)), dim=1).squeeze(0)

        # jacobian: [K, C, H, W]
        G = torch.autograd.functional.jacobian(single_logp, x)  

        # 4) inner products α_k = ⟨G_k, v⟩
        #    reshape to [K, -1] then dot
        G_flat = G.view(K, -1)             # [K, C*H*W]
        v_flat = v.view(-1)                # [C*H*W]
        alpha = G_flat.matmul(v_flat)      # [K]

        # 5) assemble final vector:  sum_k freq_k * α_k * G_k  / S
        #    we broadcast (freq*alpha) over (C,H,W)
        weights = freq * alpha             # [K]
        # reshape to [K,1,1,1] to broadcast:
        w = weights.view(K, 1, 1, 1)       
        gv = (w * G).sum(dim=0).div_(float(sample_size))  # [C,H,W]

        return gv

    def _get_model_prediction_and_probs(self, model: torch.nn.Module, x: torch.Tensor) -> Tuple[int, float, torch.Tensor]:
        """
        Get model prediction, confidence, and full probability distribution in a single forward pass.
        
        Args:
            model: Neural network model
            x: Input tensor [C, H, W]
            
        Returns:
            predicted_class: Predicted class index
            confidence: Confidence score for the predicted class
            probs: Full probability distribution tensor [num_classes]
        """
        with torch.no_grad():
            logits = model(x)  # logits shape: [B, K]
            probs = F.softmax(logits, dim=1)  # probs shape: [B, K]
            predicted_class = torch.argmax(probs, dim=1).item()  # predicted_class shape: [B]
            confidence = probs[:, predicted_class].item()  # confidence shape: [B]
        
        logger.debug(f"Model prediction: class={predicted_class}, confidence={confidence:.4f}")
        return predicted_class, confidence, probs
    
    def _margin_loss(self, x, model, true_label, target_label):
        """
        Compute the margin loss L = -log p_target + log p_true.
        x: input tensor [C,H,W] or [B,C,H,W]
        model: model to use for logits
        true_label: int or tensor
        target_label: int or tensor
        Returns: margin loss (same shape as batch, or scalar)
        """
        logits = model(x)
        logp = F.log_softmax(logits, dim=1)
        margin_loss = -logp[:, target_label] + logp[:, true_label]
        logger.debug(f"Margin loss: true_label={true_label}, target_label={target_label}, loss={margin_loss.item():.6f}")
        return margin_loss

    def _natural_gradient_unit_step(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        true_label: int,
        target_label: int,
        lambda_reg: float = 0.01,
        cg_tol: float = 1e-3,
        cg_max_iter: int = 30,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one unit‐Fisher‐length natural‐gradient step 
        aimed at pushing towards target_label.

        Args:
            model: classification network.
            x:     input tensor [C,H,W], requires_grad=True.
            true_label:   correct class index.
            target_label: adversarial class index.
            lambda_reg:   regularization parameter.
            cg_tol:       tolerance for CG.
            cg_max_iter:  maximum number of CG iterations.
        Returns:
            delta: tensor [C,H,W], a step of unit Fisher‐length.
            alpha: tensor scalar, the Fisher‐length of the raw step (before normalization).
        """
        logger.debug(f"Computing natural gradient step: true_label={true_label}, target_label={target_label}, lambda_reg={lambda_reg}")
        
        # 1) compute scalar margin loss L = -log p_target + log p_true
        L      = self._margin_loss(x, model, true_label, target_label)
        # 2) gradient g = ∇_x L
        g      = torch.autograd.grad(L, x, create_graph=False)[0]  # g shape: [C, H, W]

        logger.info(f"Margin loss: {L.item():.6f}, gradient norm: {g.norm().item():.6f}")

        # 3) solve G v = g using CG
        g_flat = g.contiguous().view(-1)  # g_flat shape: [C*H*W]
        
        def A_reg(v_flat):
            v = v_flat.view_as(x)  # v shape: [C, H, W]
            Av_regularized = self._fisher_vector_product_mc(model, x.squeeze(0), v, self.config.sample_size) + lambda_reg * v
            return Av_regularized.contiguous().view(-1)
        
        logger.debug(f"Starting CG solve: tol={cg_tol}, max_iter={cg_max_iter}")
        v_flat, cg_info = CG(A_reg, g_flat, tol=cg_tol, max_iter=cg_max_iter)  # v_flat shape: [C*H*W]
        v = v_flat.view_as(x)  # v shape: [C, H, W]
        logger.info(f"Velocity vector norm: {v.norm().item():.6f}")
        
        # CG returns (solution, (iterations, err_history))
        iterations = cg_info[0] if isinstance(cg_info, tuple) and len(cg_info) > 0 else 'unknown'
        logger.info("iterations: " + str(iterations))
        
        # 4) compute Riemannian speed α = sqrt(g·v)
        gv_dot  = (g * v).view(-1).sum()  # gv_dot shape: scalar
        # Ensure gv_dot is never negative due to numerical instability
        gv_dot = torch.clamp(gv_dot, min=1e-8)
        alpha = torch.sqrt(gv_dot)  # alpha shape: scalar
        alpha = torch.clamp(alpha, min=1e-3)

        # 5) unit‐length step δx = v / α
        delta   = v.div(alpha)  # delta shape: [C, H, W]

        logger.info(f"Natural gradient step computed: alpha={alpha}, delta_norm={delta.norm()}")
        return delta, alpha

    def compute_min_geodesic_margin(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        cg_tol: float = 1e-3,
        cg_max_iter: int = 30,
        max_steps: int = 100,
        lambda_reg: float = 0.01
    ) -> Tuple[list, torch.Tensor, torch.Tensor, list, list, list, list, list, list]:
        """
        Approximate the minimal Fisher-geodesic margin d_min to the nearest
        adversarial class via parallel natural‐gradient geodesic shooting.

        Args:
            model: classification network.
            x:     input tensor [C,H,W].
            cg_tol: tolerance for CG solves.
            max_steps: maximum steps per class to avoid infinite loops.
            lambda_reg: regularization parameter.
        Returns:
            margins: list of scalar Tensors, geodesic length for each target class
            xk_tensor: tensor of final xk tensors for each target class [num_target_classes, C, H, W]
            deltas_tensor: tensor of delta tensors for each target class and step [num_target_classes, max_steps, C, H, W]
            alphas_list: list of lists of alpha tensors (per class, per step)
            probs_seq_list: list of lists of probs_1d (per class, per step)
            pred_k_list: list of lists of predicted class indices (per class, per step)
            margin_losses_list: list of lists of margin loss values (per class, per step)
            target_classes: list of target class indices corresponding to each result
            num_steps_list: list of number of steps taken for each target class
        """
        logger.info(f"Starting geodesic margin computation with parameters: cg_tol={cg_tol}, cg_max_iter={cg_max_iter}, max_steps={max_steps}, lambda_reg={lambda_reg}")
        
        # prepare
        x0           = x.detach()  # x0 shape: [C, H, W]
        pred_label, confidence , probs = self._get_model_prediction_and_probs(model, x0)  # pred_label shape: [B], confidence shape: [B], probs shape: [B, K]
        num_classes  = probs.shape[1]  # num_classes shape: scalar

        logger.info(f"Original prediction: class={pred_label}, confidence={confidence:.4f}, num_classes={num_classes}")

        margins = []
        xk_tensors = []
        deltas_tensors = []
        alphas_list = []
        probs_seq_list = []
        pred_k_list = []
        margin_losses_list = []
        target_classes = []
        num_steps_list = []
        
        for tgt in tqdm(range(num_classes), desc="Target classes", unit="class"):
            if tgt == pred_label:
                logger.debug(f"Skipping target class {tgt} (same as predicted class)")
                continue
                
            logger.info(f"Processing target class {tgt}")
            
            # initialize
            xk           = x0.clone().requires_grad_(True)  # xk shape: [C, H, W]
            total_length = torch.tensor(0.0, device=x.device)  # total_length shape: scalar
            deltas = []
            alphas = []
            probs_seq = []
            pred_k_seq = []
            margin_losses = []
                
            # Compute initial margin loss
            initial_margin_loss = self._margin_loss(xk, model, pred_label, tgt)
            margin_losses.append(initial_margin_loss.detach().cpu().item())
            logger.info(f"Initial margin loss for class {tgt}: {initial_margin_loss.item():.6f}")

            # Get initial prediction
            _, _, probs = self._get_model_prediction_and_probs(model, xk)
            pred_k = np.argmax(probs.squeeze(0).detach().cpu().tolist())
            
            # Create progress bar for steps with detailed information
            step_pbar = tqdm(range(max_steps), desc=f"Class {tgt} (pred: {pred_k})", unit="step", leave=False)
            
            for step in step_pbar:
                logger.debug(f"Step {step + 1}/{max_steps} for target class {tgt}")
                
                delta, alpha = self._natural_gradient_unit_step(
                    model, xk, pred_label, tgt, cg_tol=cg_tol, cg_max_iter=cg_max_iter, lambda_reg=lambda_reg
                )  # delta shape: [C, H, W], alpha shape: scalar
                deltas.append(delta.detach().cpu())
                alphas.append(alpha.detach().item())

                # Take full step (no line search)
                xk_new = (xk + delta).detach().requires_grad_(True)  # xk_new shape: [C, H, W]
                xk = xk_new

                total_length += alpha  # total_length shape: scalar
                
                logger.info(f"Step {step + 1}: alpha={alpha.item():.6f}, total_length={total_length.item():.6f}")
                
                # Compute margin loss at new point
                current_margin_loss = self._margin_loss(xk, model, pred_label, tgt)
                margin_losses.append(current_margin_loss.detach().cpu().item())

                # check crossing: stop when ANY other class beats the original
                _, _, probs = self._get_model_prediction_and_probs(model, xk)  # probs shape: [1, num_classes]
                probs_1d = probs.squeeze(0).detach().cpu().tolist()  # shape: [num_classes]
                probs_seq.append(probs_1d)
                pred_k = np.argmax(probs_1d)
                pred_k_seq.append(pred_k)
                
                # Update progress bar with current information
                step_pbar.set_description(f"Class {tgt} (pred: {pred_k}, α: {alpha.item():.3f}, len: {total_length.item():.3f})")
                
                logger.info(f"Step {step + 1}: predicted_class={pred_k}, margin_loss={current_margin_loss.item():.6f}")
                
                if pred_k != pred_label:
                    logger.info(f"Class boundary crossed at step {step + 1} for target class {tgt}: prediction changed from {pred_label} to {pred_k}")
                    break

            margins.append(total_length.detach().cpu().item())  # margins list contains scalar tensors
            xk_tensors.append(xk.detach().cpu())
            deltas_tensors.append(deltas)
            alphas_list.append(alphas)
            probs_seq_list.append(probs_seq)
            pred_k_list.append(pred_k_seq)
            margin_losses_list.append(margin_losses)
            target_classes.append(tgt)
            num_steps_list.append(step + 1)  # step is 0-indexed, so add 1 for actual number of steps
            
            logger.info(f"Completed target class {tgt}: margin={total_length.item():.6f}, steps={step + 1}")

        # Convert lists to tensors
        if xk_tensors:
            xk_tensor = torch.stack(xk_tensors, dim=0)  # Shape: [num_target_classes, C, H, W]
        else:
            xk_tensor = torch.empty(0, *x.shape, device='cpu')
            
        # Convert deltas to tensor - pad with zeros for variable length sequences
        if deltas_tensors:
            max_steps_actual = max(len(deltas) for deltas in deltas_tensors)
            deltas_tensor = torch.zeros(len(deltas_tensors), max_steps_actual, *x.shape, device='cpu')
            for i, deltas in enumerate(deltas_tensors):
                for j, delta in enumerate(deltas):
                    deltas_tensor[i, j] = delta
        else:
            deltas_tensor = torch.empty(0, 0, *x.shape, device='cpu')

        logger.info(f"Geodesic margin computation completed. Found {len(margins)} target classes with margins: {[f'{m:.6f}' for m in margins]}")
        
        # return all margins, xk, deltas, alphas, probs_seq, pred_k_seq, margin_losses_list, target_classes, num_steps_list
        return margins, xk_tensor, deltas_tensor, alphas_list, probs_seq_list, pred_k_list, margin_losses_list, target_classes, num_steps_list

if __name__ == "__main__":
    # Use the existing logger for main execution
    logger.info("Starting Fisher Margin computation (main execution)")
    print("Starting Fisher Margin computation (main execution)")
    
    # Load config
    config = ExperimentConfig()
    logger.info(f"Loaded experiment config: {config}")
    print(f"Loaded experiment config: {config}")
    
    # Image path
    image_path = 'data/dog.png'
    
    # Check if image exists
    if not os.path.exists(image_path):
        logger.error(f"Image file {image_path} not found!")
        print(f"ERROR: Image file {image_path} not found!")
        exit()

    # Preprocess image
    logger.info("Preprocessing image...")
    print("Preprocessing image...")
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).double()
    
    # Add batch dimension
    x = image_tensor.unsqueeze(0)  # Shape: (1, 3, 32, 32)
    logger.info(f"Image loaded and preprocessed: shape={x.shape}")
    print(f"Image loaded and preprocessed: shape={x.shape}")
    
    # Load model (assuming you have the data_loader module)
    try:
        from data_loader import load_model
        model = load_model('resnet18')
        model.eval()
        logger.info("Model loaded successfully!")
        print("Model loaded successfully!")
    except ImportError:
        logger.error("Could not import data_loader. Please ensure the model is loaded manually.")
        logger.error("Example: model = load_model('resnet18')")
        print("ERROR: Could not import data_loader. Please ensure the model is loaded manually.")
        exit()
    
    # Get prediction for original image
    with torch.no_grad():
        logits_orig = model(x)
        probs_orig = F.softmax(logits_orig, dim=1)
        pred_orig = torch.argmax(logits_orig, dim=1)
        confidence_orig = probs_orig.max(dim=1)[0]
    
    logger.info(f"Original prediction: Class {pred_orig.item()}, Confidence: {confidence_orig.item():.4f}")
    print(f"Original prediction: Class {pred_orig.item()}, Confidence: {confidence_orig.item():.4f}")
    
    fisher_margin = FisherMargin(
        model=model,
        config=config
    )

    # Compute minimal geodesic margin
    logger.info("Starting minimal geodesic margin computation...")
    print("Starting minimal geodesic margin computation...")
    margins, xk_tensor, deltas_tensor, alphas_list, probs_seq_list, pred_k_list, margin_losses_list, target_classes, num_steps_list = fisher_margin.compute_min_geodesic_margin(model, x)
    
    # CIFAR-10 class names
    cifar10_class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    logger.info("Generating visualization plots...")
    print("Generating visualization plots...")
    plot_fisher_margin_figures(margins, xk_tensor, deltas_tensor, alphas_list, probs_seq_list, pred_k_list, margin_losses_list, pred_orig.item(), target_classes, class_names=cifar10_class_names, output_dir='plots/')
    
    logger.info("Fisher Margin computation completed successfully!")
    print("Fisher Margin computation completed successfully!")
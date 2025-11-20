import torch
import torch.nn.functional as F
import numpy as np

def bim_step(image_clean, image, data_grad, alpha, epsilon, max_v=1):
    """
    Basic Iterative Method (BIM) step.
    Mathematically similar to PGD, but typically assumes alpha = epsilon / iterations 
    and often doesn't include the random start step of standard PGD.
    """
    sign_data_grad = data_grad.sign()
    
    # Update image
    perturbed_image = image + alpha * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, max_v)
    
    # Projection (Clipping)
    clipped_perturb = torch.clamp(image_clean - perturbed_image, min=-epsilon, max=epsilon)
    perturbed_image = image_clean - clipped_perturb
    
    return perturbed_image

def to_tanh_space(x, box_min=0, box_max=1):
    """
    Convert image to tanh space for C&W attack to ensure box constraints.
    x = 0.5 * (tanh(w) + 1)
    w = arctanh(2x - 1)
    """
    # Add a small constant to avoid division by zero or infinity
    _x = x.clamp(box_min + 1e-6, box_max - 1e-6)
    w = torch.arctanh((_x - box_min) / (box_max - box_min) * 2 - 1)
    return w

def from_tanh_space(w, box_min=0, box_max=1):
    """
    Convert from tanh space back to image space.
    """
    return 0.5 * (torch.tanh(w) + 1) * (box_max - box_min) + box_min

def cw_loss(logits, target, kappa=0, target_is_gt=True):
    """
    Carlini & Wagner Loss (f-function).
    
    Args:
        logits: Model output (before softmax).
        target: The target class index (or GT index).
        kappa: Confidence margin.
        target_is_gt: If True, we want to DECREASE the prob of 'target' (Untargeted attack).
                      If False, we want to INCREASE the prob of 'target' (Targeted attack).
    """
    # Create one-hot encoding
    one_hot_target = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1)
    
    # Get the logit of the target class
    real = (logits * one_hot_target).sum(1)
    
    # Get the max logit of the OTHER classes
    # We subtract a large value from the target index so it's not picked as max
    other = (logits - one_hot_target * 1e4).max(1)[0]
    
    if target_is_gt:
        # Untargeted: We want real < other (GT probability goes down)
        # Loss = max(real - other, -kappa)
        # We want to minimize this loss
        loss = torch.clamp(real - other, min=-kappa)
    else:
        # Targeted: We want other < real
        loss = torch.clamp(other - real, min=-kappa)
        
    return loss.sum()
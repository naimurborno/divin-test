"""Utility functions for DivIn pipeline."""

import torch
import math
import os
import torch.nn as nn
import torchvision.transforms as transforms
from itertools import combinations


def arnoldi_iteration_jvp(b, n: int, jvp_func):
    """Compute a basis of the (n + 1)-Krylov subspace using JVP.

    This is the space spanned by the vectors {b, Ab, ..., A^n b}.

    Parameters
    ----------
    b : torch.Tensor
        Initial vector (length m).
    n : int
        One less than the dimension of the Krylov subspace.

    Returns
    -------
    h : torch.Tensor
        An n x n upper Hessenberg matrix.
    """
    eps = 1e-12
    h = torch.zeros((n + 1, n), dtype=torch.float64, device=b.device)
    Q = torch.zeros((b.numel(), n + 1), dtype=torch.complex128, device=b.device)

    Q[:, 0] = b / torch.norm(b, 2)

    for k in range(1, n + 1):
        vec = Q[:, k - 1]
        v = jvp_func(vec)
        for j in range(k):
            h[j, k - 1] = torch.real(Q[:, j].conj().dot(v))
            v = v - h[j, k - 1] * Q[:, j]

        h[k, k - 1] = torch.norm(v, 2)

        if h[k, k - 1] > eps:
            Q[:, k] = v / h[k, k - 1]
        else:
            return Q[:, :k], h[:k + 1, :k]

    h = h[:-1, :]
    return h


def gaussian_samples_with_sample_corr(n, d, rho, device=None, dtype=torch.float16):
    """Generate n samples of dimension d with pairwise correlation rho.

    Returns X in R^{n x d}: n samples, each of dimension d.
    Each row ~ N(0, I_d). Across rows, pairwise correlation = rho (equicorrelated).
    Requires rho in (-1/(n-1), 1).
    """
    if rho <= -1 / (n - 1) or rho >= 1:
        raise ValueError(f"rho must be in (-1/(n-1), 1); got {rho}")

    device = device or 'cpu'

    n_vec = torch.ones(n, 1, device=device, dtype=dtype) / torch.sqrt(
        torch.tensor(float(n), device=device, dtype=dtype)
    )

    P1 = n_vec @ n_vec.T
    P0 = torch.eye(n, device=device, dtype=dtype) - P1

    lam1 = 1 + (n - 1) * rho
    lam0 = 1 - rho

    R_half = math.sqrt(lam0) * P0 + math.sqrt(lam1) * P1

    Z = torch.randn(n, d, device=device, dtype=dtype)
    X = R_half @ Z

    return X


def compute_div_per_sample(latent):
    """Compute pairwise diversity metric per sample using RBF kernel."""
    bsz = len(latent)
    lat_flat = latent.view(bsz, -1)
    xx = (lat_flat * lat_flat).sum(dim=1, keepdim=True)

    dist2 = xx + xx.t() - 2.0 * (lat_flat @ lat_flat.t())
    dist2 = dist2.clamp_min(0.0)
    distance = torch.sqrt(dist2 + 1e-12)

    mask = torch.eye(bsz, dtype=torch.bool, device=latent.device)
    distance_offdiag = distance[~mask]
    median_r = distance_offdiag.median()
    tau2 = (median_r ** 2) / max(1.0, math.log(max(2, bsz - 1)))
    tau2 = float(tau2.clamp_min(1e-12))

    dist2 = dist2.masked_fill(mask, float('inf'))
    K = torch.exp(-dist2 / (tau2 + 1e-12))
    div_per_sample = K.sum(dim=1) / max(1, (bsz - 1))
    return dist2, div_per_sample


def measure_CLIP_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    """Measure CLIP cosine similarity between images and a text prompt."""
    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return (image_features @ text_features.T).mean(-1)


def measure_SSCD_similarity(gt_images, images, model, device):
    """Measure SSCD (Self-Supervised Copy Detection) similarity."""
    ret_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    gt_images = torch.stack([ret_transform(x.convert("RGB")) for x in gt_images]).to(device)
    images = torch.stack([ret_transform(x.convert("RGB")) for x in images]).to(device)

    gt_images = gt_images.type(torch.float32)
    images = images.type(torch.float32)
    with torch.no_grad():
        feat_1 = model(gt_images).clone()
        feat_1 = nn.functional.normalize(feat_1, dim=1, p=2)

        feat_2 = model(images).clone()
        feat_2 = nn.functional.normalize(feat_2, dim=1, p=2)

        return torch.max(torch.mm(feat_1, feat_2.T), dim=0).values

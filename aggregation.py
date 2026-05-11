"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.

Strategy (V5 — Synergistic Structural-Semantic Hybrid)
------------------------------------------------------
Three feature blocks are extracted and concatenated:

  Block 1 — Geometric features (50 dims):
    Smart-masked response-token statistics from layers 10-19.
      1. L2 norm mean (response tokens, per-layer)        10
      2. L2 norm std  (response tokens, per-layer)        10
      3. Inter-layer cosine similarity (last token)         9
      4. Inter-layer norm drift (last token)                9
      5. Spherical Geodesic Index — SGI (per-layer)        10
      6. Response fraction (n_resp / n_total)                1
      7. Log response length — log(n_resp + 1)              1

  Block 2 — Semantic context (896 dims):
    Mean-pooled hidden states across layers 10-19 and response tokens.
    Compressed by PCA(32) in ``probe.py``.

  Block 3 — Lexical anomaly (1792 dims):
    Max-pooled response tokens for layers 12 and 13.
    Compressed by PCA(32) in ``probe.py``.

Total output: 50 + 896 + 1792 = 2738 dims per sample.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ── Validated configuration ───────────────────────────────────────────
LAYER_START = 10
LAYER_END = 20             # exclusive → layers 10-19
N_GEO_FEATURES = 50       # first N dims are geometric
N_SEM_FEATURES = 896      # next N dims are mean-pooled semantic
ASSISTANT_TOKEN_OFFSET = 3
# ──────────────────────────────────────────────────────────────────────


def _find_response_start(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> int:
    """Locate ``<|im_start|>assistant`` boundary via embedding similarity.

    Finds the last occurrence of the ``<|im_start|>`` token by comparing
    embedding-layer cosine similarity to the very first token (which is
    always ``<|im_start|>`` in ChatML format).

    Args:
        hidden_states:  Tensor ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor ``(seq_len,)``; 1 = real, 0 = pad.

    Returns:
        Integer index (relative to real tokens) where the response begins.

    Note:
        Validated: 50/50 exact matches on diagnostic test.
    """
    emb_layer = hidden_states[0]
    real_mask = attention_mask.bool()
    real_positions = real_mask.nonzero(as_tuple=False).squeeze(-1)

    if len(real_positions) < 4:
        return 0

    ref_vec = emb_layer[real_positions[0]]
    real_embeddings = emb_layer[real_mask]
    sims = F.cosine_similarity(real_embeddings, ref_vec.unsqueeze(0), dim=-1)
    matches = (sims > 0.999).nonzero(as_tuple=False).squeeze(-1)

    if len(matches) >= 3:
        last_im = matches[-1].item()
        return min(last_im + ASSISTANT_TOKEN_OFFSET, len(real_positions) - 1)

    return int(len(real_positions) * 0.65)


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(2738,)`` containing three blocks:
        geometric (50) + semantic (896) + lexical anomaly (1792).
    """
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    dev = hidden_states.device
    layers = hidden_states[LAYER_START:LAYER_END]  # (10, seq, dim)
    n_layers = layers.shape[0]
    eps = 1e-7

    real_mask = attention_mask.bool()
    real_positions = real_mask.nonzero(as_tuple=False).squeeze(-1)
    last_pos = int(real_positions[-1].item())
    n_real = int(real_mask.sum().item())

    # ── Smart masking — isolate response tokens ──────────────────────
    resp_start = _find_response_start(hidden_states, attention_mask)
    real_tokens = layers[:, real_mask, :]           # (10, n_real, dim)
    prompt_tokens = real_tokens[:, :resp_start, :]
    response_tokens = real_tokens[:, resp_start:, :]
    n_resp = response_tokens.shape[1]

    last_token_vecs = layers[:, last_pos, :]        # (10, dim)
    embedding_vec = hidden_states[0, last_pos, :]

    parts: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Block 1: Geometric features (50 dims)
    # ------------------------------------------------------------------

    # 1. L2 norm mean — response tokens (10) ──────────────────────────
    if n_resp > 0:
        resp_l2 = torch.norm(response_tokens, dim=-1)  # (10, n_resp)
        parts.append(resp_l2.mean(dim=-1))
    else:
        parts.append(torch.zeros(n_layers, device=dev))

    # 2. L2 norm std — response tokens (10) — time-variance proxy ─────
    if n_resp > 1:
        parts.append(resp_l2.std(dim=-1))
    else:
        parts.append(torch.zeros(n_layers, device=dev))

    # 3. Inter-layer cosine at last token (9) ─────────────────────────
    cos_list = []
    for i in range(n_layers - 1):
        cos_list.append(F.cosine_similarity(
            last_token_vecs[i].unsqueeze(0),
            last_token_vecs[i + 1].unsqueeze(0)))
    parts.append(torch.cat(cos_list))

    # 4. Inter-layer norm drift at last token (9) ─────────────────────
    lt_norms = torch.norm(last_token_vecs, dim=-1)
    parts.append(torch.abs(lt_norms[1:] - lt_norms[:-1]))

    # 5. SGI per layer (10) ───────────────────────────────────────────
    sgi_list = []
    for i in range(n_layers):
        rv = last_token_vecs[i]
        pc = (prompt_tokens[i].mean(dim=0)
              if prompt_tokens.shape[1] > 0
              else real_tokens[i, 0, :])

        cos_p = F.cosine_similarity(
            rv.unsqueeze(0), pc.unsqueeze(0)).clamp(-1 + eps, 1 - eps)
        cos_e = F.cosine_similarity(
            rv.unsqueeze(0), embedding_vec.unsqueeze(0)).clamp(-1 + eps, 1 - eps)
        sgi_list.append(torch.arccos(cos_p) / (torch.arccos(cos_e) + eps))
    parts.append(torch.cat(sgi_list))

    # 6. Response fraction (1) ────────────────────────────────────────
    parts.append(torch.tensor(
        [n_resp / max(n_real, 1)], dtype=torch.float32, device=dev))

    # 7. Log response length (1) ──────────────────────────────────────
    parts.append(torch.tensor(
        [math.log(n_resp + 1)], dtype=torch.float32, device=dev))

    # ------------------------------------------------------------------
    # Block 2: Mean-pooled semantic vector (896 dims)
    # ------------------------------------------------------------------
    if n_resp > 0:
        mean_per_layer = response_tokens.mean(dim=1)   # (10, 896)
        mean_pooled = mean_per_layer.mean(dim=0)       # (896,)
    else:
        mean_pooled = real_tokens.mean(dim=(0, 1))     # (896,)
    parts.append(mean_pooled)

    # ------------------------------------------------------------------
    # Block 3: Max-pooled lexical anomaly — L12 + L13 (1792 dims)
    # ------------------------------------------------------------------
    resp_mask = attention_mask.bool()
    resp_offset = _find_response_start(hidden_states, attention_mask)

    for layer_idx in (12, 13):
        layer_resp = hidden_states[layer_idx, resp_mask, :][resp_offset:, :]
        if layer_resp.size(0) > 0:
            parts.append(layer_resp.max(dim=0).values)
        else:
            parts.append(hidden_states[layer_idx].max(dim=0).values)

    # ------------------------------------------------------------------
    return torch.cat(parts, dim=0).float()


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Note:
        All features are already computed inside ``aggregate()``.
        This stub is kept for API compatibility with ``solution.py``.
    """
    return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.py`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.py``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = 2738`` (or larger when geometric features are
        enabled).
    """
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features

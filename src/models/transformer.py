"""
FT-Transformer (Feature Tokenizer + Transformer) for tabular data, in PyTorch.

How it works, step by step:
  1. Tokenise: every feature becomes one d-dimensional token.
       numerical feature j  ->  x_j * W_j + b_j     (its own linear embedding)
       categorical feature  ->  lookup in its own embedding table
  2. Prepend a learnable [CLS] token (like BERT).
  3. Pass the token sequence through several Transformer encoder layers
     (multi-head self-attention + feed-forward, pre-layer-norm, dropout).
     Self-attention lets every feature "look at" every other feature.
  4. The final [CLS] vector summarises the patient; a small head maps it to
     the logit of Alzheimer's.

The attention weights of every layer are kept so that we can visualise
which features the [CLS] token attends to.

`TransformerClassifier` wraps the network with an sklearn-like API:
fit / predict_proba / embed / attention, and handles feature scaling
(StandardScaler fitted on the training rows only).
"""
import copy
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src import config

DEVICE = torch.device("cpu")  # small model and data: CPU is faster than MPS here
# PyTorch and XGBoost each bundle their own OpenMP runtime on macOS; letting both
# start thread pools in one process can deadlock. Inference on this tiny model is
# fast single-threaded, so that is the default (src/train.py raises it to 4).
torch.set_num_threads(1)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class FeatureTokenizer(nn.Module):
    """Turn each feature of a patient into its own token."""

    def __init__(self, n_num: int, cat_cardinalities: list[int], d: int):
        super().__init__()
        self.num_weight = nn.Parameter(torch.empty(n_num, d))
        self.num_bias = nn.Parameter(torch.empty(n_num, d))
        for p in (self.num_weight, self.num_bias):
            nn.init.uniform_(p, -1 / math.sqrt(d), 1 / math.sqrt(d))
        self.cat_embeddings = nn.ModuleList([nn.Embedding(c, d) for c in cat_cardinalities])

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        # (batch, n_num, 1) * (n_num, d) -> (batch, n_num, d)
        tokens = [x_num.unsqueeze(-1) * self.num_weight + self.num_bias]
        if len(self.cat_embeddings):
            tokens.append(torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1))
        return torch.cat(tokens, dim=1)


class EncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer that also returns attention weights."""

    def __init__(self, d: int, n_heads: int, ff_mult: float, attn_dropout: float, ff_dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        hidden = int(d * ff_mult)
        self.ff = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(ff_dropout), nn.Linear(hidden, d))
        self.drop = nn.Dropout(ff_dropout)

    def forward(self, x):
        h = self.norm1(x)
        a, w = self.attn(h, h, h, need_weights=True, average_attn_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x, w  # w: (batch, heads, tokens, tokens)


class FTTransformer(nn.Module):
    def __init__(self, n_num: int, cat_cardinalities: list[int], d: int = 32, n_layers: int = 3,
                 n_heads: int = 4, ff_mult: float = 2.0, attn_dropout: float = 0.1,
                 ff_dropout: float = 0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_num, cat_cardinalities, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls, std=0.02)
        self.layers = nn.ModuleList(
            [EncoderLayer(d, n_heads, ff_mult, attn_dropout, ff_dropout) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(d, 1))

    def forward(self, x_num, x_cat, return_all: bool = False):
        tokens = self.tokenizer(x_num, x_cat)
        x = torch.cat([self.cls.expand(len(tokens), -1, -1), tokens], dim=1)
        attentions = []
        for layer in self.layers:
            x, w = layer(x)
            attentions.append(w)
        cls_emb = self.final_norm(x[:, 0])
        logit = self.head(cls_emb).squeeze(-1)
        if return_all:
            return logit, cls_emb, attentions
        return logit


# ---------------------------------------------------------------------------
# sklearn-like wrapper
# ---------------------------------------------------------------------------
class TransformerClassifier:
    """Train / predict / embed with an FT-Transformer on a pandas DataFrame."""

    def __init__(self, cat_cardinality: dict[str, int], d: int = 32, n_layers: int = 3,
                 n_heads: int = 4, ff_mult: float = 2.0, attn_dropout: float = 0.1,
                 ff_dropout: float = 0.1, lr: float = 1e-3, weight_decay: float = 1e-4,
                 batch_size: int = 64, class_weight: bool = False,
                 max_epochs: int = config.TRANSFORMER_MAX_EPOCHS,
                 patience: int = config.TRANSFORMER_PATIENCE, seed: int = config.SEED):
        self.cat_cardinality = cat_cardinality
        self.params = dict(d=d, n_layers=n_layers, n_heads=n_heads, ff_mult=ff_mult,
                           attn_dropout=attn_dropout, ff_dropout=ff_dropout)
        self.lr, self.weight_decay, self.batch_size = lr, weight_decay, batch_size
        self.class_weight, self.max_epochs, self.patience, self.seed = class_weight, max_epochs, patience, seed
        self.history = {"train_loss": [], "val_loss": []}

    # -- data handling ------------------------------------------------------
    def _split_columns(self, X: pd.DataFrame):
        self.feature_names_ = list(X.columns)
        self.cat_cols_ = [c for c in X.columns if c in self.cat_cardinality]
        self.num_cols_ = [c for c in X.columns if c not in self.cat_cardinality]

    def _tensors(self, X: pd.DataFrame):
        x_num = torch.tensor(self.scaler_.transform(X[self.num_cols_].values), dtype=torch.float32)
        x_cat = torch.tensor(X[self.cat_cols_].values.astype(np.int64), dtype=torch.long) \
            if self.cat_cols_ else torch.zeros((len(X), 0), dtype=torch.long)
        return x_num, x_cat

    @property
    def token_names(self) -> list[str]:
        """Token order inside the network: [CLS], numerical..., categorical..."""
        return ["[CLS]"] + self.num_cols_ + self.cat_cols_

    # -- training --------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y, X_val: pd.DataFrame, y_val):
        """Train with AdamW + cosine LR schedule; early stopping on validation loss."""
        set_seed(self.seed)
        self._split_columns(X)
        self.scaler_ = StandardScaler().fit(X[self.num_cols_].values)  # training rows only
        self.model_ = FTTransformer(len(self.num_cols_),
                                    [self.cat_cardinality[c] for c in self.cat_cols_],
                                    **self.params).to(DEVICE)
        xn, xc = self._tensors(X)
        vn, vc = self._tensors(X_val)
        yt = torch.tensor(np.asarray(y), dtype=torch.float32)
        yv = torch.tensor(np.asarray(y_val), dtype=torch.float32)

        pos_weight = None
        if self.class_weight:  # up-weight the minority (Alzheimer's) class
            pos_weight = (len(yt) - yt.sum()) / yt.sum()
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        val_loss_fn = nn.BCEWithLogitsLoss()
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)

        gen = torch.Generator().manual_seed(self.seed)
        best, best_state, bad_epochs = float("inf"), None, 0
        self.history = {"train_loss": [], "val_loss": []}
        for _epoch in range(self.max_epochs):
            self.model_.train()
            perm = torch.randperm(len(yt), generator=gen)
            total = 0.0
            for i in range(0, len(perm), self.batch_size):
                b = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.model_(xn[b], xc[b]), yt[b])
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            sched.step()
            self.model_.eval()
            with torch.no_grad():
                vloss = val_loss_fn(self.model_(vn, vc), yv).item()
            self.history["train_loss"].append(total / len(yt))
            self.history["val_loss"].append(vloss)
            if vloss < best - 1e-5:
                best, best_state, bad_epochs = vloss, copy.deepcopy(self.model_.state_dict()), 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        self.model_.load_state_dict(best_state)
        self.best_epoch_ = int(np.argmin(self.history["val_loss"])) + 1
        self.best_val_loss_ = best
        return self

    # -- inference ---------------------------------------------------------------
    @torch.no_grad()
    def _forward(self, X: pd.DataFrame):
        self.model_.eval()
        xn, xc = self._tensors(X[self.feature_names_])
        return self.model_(xn, xc, return_all=True)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        logit, _, _ = self._forward(X)
        return torch.sigmoid(logit).numpy()

    def embed(self, X: pd.DataFrame) -> np.ndarray:
        """[CLS] embedding (one d-dimensional vector per patient)."""
        _, emb, _ = self._forward(X)
        return emb.numpy()

    def attention(self, X: pd.DataFrame) -> np.ndarray:
        """Attention averaged over layers and heads: (patients, tokens, tokens)."""
        _, _, att = self._forward(X)
        return torch.stack(att).mean(dim=(0, 2)).numpy()

    def cls_attention(self, X: pd.DataFrame) -> pd.DataFrame:
        """Attention from [CLS] to each feature token, one row per patient."""
        att = self.attention(X)[:, 0, 1:]
        return pd.DataFrame(att, columns=self.token_names[1:])

    # -- persistence ---------------------------------------------------------
    def save(self, path) -> None:
        torch.save({"state_dict": self.model_.state_dict(), "params": self.params,
                    "cat_cardinality": self.cat_cardinality, "feature_names": self.feature_names_,
                    "num_cols": self.num_cols_, "cat_cols": self.cat_cols_,
                    "scaler_mean": self.scaler_.mean_, "scaler_scale": self.scaler_.scale_,
                    "history": self.history, "best_epoch": self.best_epoch_,
                    "train_cfg": dict(lr=self.lr, weight_decay=self.weight_decay,
                                      batch_size=self.batch_size, class_weight=self.class_weight)},
                   path)

    @classmethod
    def load(cls, path) -> "TransformerClassifier":
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        obj = cls(ck["cat_cardinality"], **ck["params"], **ck["train_cfg"])
        obj.feature_names_, obj.num_cols_, obj.cat_cols_ = ck["feature_names"], ck["num_cols"], ck["cat_cols"]
        obj.scaler_ = StandardScaler()
        obj.scaler_.mean_, obj.scaler_.scale_ = ck["scaler_mean"], ck["scaler_scale"]
        obj.scaler_.var_ = obj.scaler_.scale_ ** 2
        obj.scaler_.n_features_in_ = len(obj.num_cols_)
        obj.model_ = FTTransformer(len(obj.num_cols_), [obj.cat_cardinality[c] for c in obj.cat_cols_],
                                   **obj.params).to(DEVICE)
        obj.model_.load_state_dict(ck["state_dict"])
        obj.model_.eval()
        obj.history, obj.best_epoch_ = ck["history"], ck["best_epoch"]
        return obj

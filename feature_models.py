import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def pairwise_vector_features(vectors, pairs):
    first, second = pairs
    unit = vectors[:, :, :32]
    cosine = (unit[:, first] * unit[:, second]).sum(dim=-1).clamp(-1, 1)
    norms = vectors[:, :, 32].expm1()
    present = vectors[:, first, 33] * vectors[:, second, 33]
    dot = cosine * norms[:, first] * norms[:, second]
    distance = (
        norms[:, first].square() + norms[:, second].square() - 2 * dot
    ).clamp_min(0).sqrt()
    return torch.cat([
        cosine * present,
        dot.sign() * dot.abs().log1p().clamp_max(20) * present,
        distance.log1p().clamp_max(20) * present,
        present,
    ], dim=1)


class FeatureEncoder(nn.Module):
    def __init__(
        self, n_numeric, n_dense, n_vectors, cardinalities, continuous_indices,
        embedding_dimensions, numerical_encoding="none", bins=None,
        vector_projection=False, vector_dropout=0.0, vector_crosses=False,
        summary_groups=None, indicator_groups=None, numeric_dropout=0.0,
    ):
        super().__init__()
        self.n_dense, self.n_vectors = n_dense, n_vectors
        self.vector_dropout, self.vector_crosses = vector_dropout, vector_crosses
        self.numeric_dropout = numeric_dropout
        self.summary_groups = summary_groups or [[] for _ in range(n_vectors)]
        self.indicator_groups = indicator_groups or [[] for _ in range(n_vectors)]
        self.register_buffer("continuous", torch.tensor(continuous_indices, dtype=torch.int64))
        remaining = [i for i in range(n_numeric) if i not in set(continuous_indices)]
        self.register_buffer("remaining", torch.tensor(remaining, dtype=torch.int64))
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, dimension, padding_idx=0)
            for cardinality, dimension in zip(cardinalities, embedding_dimensions)
        )
        for layer in self.embeddings:
            nn.init.normal_(layer.weight, std=0.02)
            with torch.no_grad():
                layer.weight[0].zero_()
        self.numerical = None
        dimension = n_numeric + sum(embedding_dimensions)
        if numerical_encoding != "none":
            from rtdl_num_embeddings import PeriodicEmbeddings, PiecewiseLinearEmbeddings

            if not continuous_indices:
                raise ValueError("Numerical embeddings require continuous scalar features")
            if numerical_encoding == "ple":
                if bins is None or len(bins) != len(continuous_indices):
                    raise ValueError("PLE requires one train-fitted bin array per feature")
                self.numerical = PiecewiseLinearEmbeddings(
                    [torch.tensor(values, dtype=torch.float32) for values in bins],
                    8, activation=True, version="B",
                )
            elif numerical_encoding == "plr":
                self.numerical = PeriodicEmbeddings(
                    len(continuous_indices), 8, n_frequencies=8,
                    frequency_init_scale=0.01, activation=True, lite=True,
                )
            else:
                raise ValueError(f"Unknown numerical encoding: {numerical_encoding}")
            dimension += len(continuous_indices) * 7
        self.vector_weight = None
        if vector_projection:
            self.vector_weight = nn.Parameter(torch.empty(n_vectors, 32, 32))
            for matrix in self.vector_weight:
                nn.init.xavier_uniform_(matrix)
            self.vector_bias = nn.Parameter(torch.zeros(n_vectors, 32))
            self.vector_norm = nn.LayerNorm(32)
        pairs = torch.triu_indices(n_vectors, n_vectors, offset=1)
        self.register_buffer("pairs", pairs)
        if vector_crosses:
            dimension += pairs.shape[1] * 4
        self.output_dim = dimension

    def forward(self, numeric, categorical):
        if self.training and self.numeric_dropout:
            numeric = numeric.clone()
            numeric[:, self.continuous] = F.dropout(
                numeric[:, self.continuous], p=self.numeric_dropout, training=True
            )
        vectors = numeric[:, self.n_dense :].reshape(-1, self.n_vectors, 34)
        if self.training and self.vector_dropout:
            keep = torch.rand(
                vectors.shape[:2], device=vectors.device
            ) >= self.vector_dropout
            numeric = numeric.clone()
            vectors = vectors * keep.unsqueeze(-1)
            for index in range(self.n_vectors):
                if self.summary_groups[index]:
                    numeric[:, self.summary_groups[index]] *= keep[:, index : index + 1]
                if self.indicator_groups[index]:
                    columns = self.indicator_groups[index]
                    numeric[:, columns] = torch.where(
                        keep[:, index : index + 1],
                        numeric[:, columns], torch.ones_like(numeric[:, columns]),
                    )
        crossed = None
        if self.vector_crosses:
            crossed = pairwise_vector_features(vectors, self.pairs)
        if self.vector_weight is not None:
            directions = F.silu(self.vector_norm(
                torch.einsum("bfi,fij->bfj", vectors[:, :, :32], self.vector_weight)
                + self.vector_bias
            )) * vectors[:, :, 33:]
            vectors = torch.cat([directions, vectors[:, :, 32:]], dim=-1)
        if self.vector_weight is not None or (self.training and self.vector_dropout):
            numeric = torch.cat([numeric[:, : self.n_dense], vectors.flatten(1)], dim=1)
        if self.numerical is not None:
            numeric = torch.cat([
                numeric[:, self.remaining],
                self.numerical(numeric[:, self.continuous]).flatten(1),
            ], dim=1)
        if self.training:
            categorical = categorical.masked_fill(
                torch.rand_like(categorical, dtype=torch.float32) < 0.3, 0
            )
        parts = [
            numeric,
            *[layer(categorical[:, index]) for index, layer in enumerate(self.embeddings)],
        ]
        if crossed is not None:
            parts.append(crossed)
        return torch.cat(parts, dim=1)


class LowRankCross(nn.Module):
    def __init__(self, dimension, rank=32, experts=1):
        super().__init__()
        self.experts = experts
        self.down = nn.ModuleList(nn.Linear(dimension, rank, bias=False) for _ in range(experts))
        self.up = nn.ModuleList(nn.Linear(rank, dimension, bias=False) for _ in range(experts))
        self.bias = nn.Parameter(torch.zeros(experts, dimension))
        if experts > 1:
            self.middle = nn.ModuleList(nn.Linear(rank, rank, bias=False) for _ in range(experts))
            self.gate = nn.Linear(dimension, experts, bias=False)

    def forward(self, original, current):
        if self.experts == 1:
            return current + original * (self.up[0](self.down[0](current)) + self.bias[0])
        weights = self.gate(current).softmax(dim=-1)
        transformed = []
        for index in range(self.experts):
            low = self.down[index](current).tanh()
            low = self.middle[index](low).tanh()
            transformed.append(
                original * (self.up[index](low) + self.bias[index])
            )
        return current + (
            torch.stack(transformed, dim=1) * weights.unsqueeze(-1)
        ).sum(dim=1)


class DCNv2Backbone(nn.Module):
    def __init__(self, dimension, width=128, dropout=0.2, rank=32, experts=1):
        super().__init__()
        self.crosses = nn.ModuleList(LowRankCross(dimension, rank, experts) for _ in range(2))
        self.deep = nn.Sequential(
            nn.Linear(dimension, width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout),
        )
        self.output = nn.Linear(dimension + width, 1)

    def forward(self, inputs):
        crossed = inputs
        for layer in self.crosses:
            crossed = layer(inputs, crossed)
        return self.output(torch.cat([crossed, self.deep(inputs)], dim=1))


class FinalMLPBackbone(nn.Module):
    def __init__(self, dimension, width=128, dropout=0.2, heads=4, gating=True):
        super().__init__()
        if width % heads:
            raise ValueError("FinalMLP width must be divisible by head count")
        self.width, self.heads, self.gating = width, heads, gating
        self.streams = nn.ModuleList(
            nn.Sequential(
                nn.Linear(dimension, width), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout),
            ) for _ in range(2)
        )
        if gating:
            self.gates = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(dimension, 64), nn.ReLU(),
                    nn.Linear(64, dimension), nn.Sigmoid(),
                ) for _ in range(2)
            )
        self.linear = nn.ModuleList(nn.Linear(width, 1, bias=False) for _ in range(2))
        self.bilinear = nn.Parameter(torch.empty(heads, width // heads, width // heads))
        for matrix in self.bilinear:
            nn.init.xavier_uniform_(matrix)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        streams = [
            stream(inputs * (2 * self.gates[index](inputs) if self.gating else 1))
            for index, stream in enumerate(self.streams)
        ]
        first, second = [stream.reshape(-1, self.heads, self.width // self.heads) for stream in streams]
        interaction = torch.einsum("bhd,hde,bhe->b", first, self.bilinear, second).unsqueeze(1)
        return self.linear[0](streams[0]) + self.linear[1](streams[1]) + interaction + self.bias


class HeterogeneousClassifier(nn.Module):
    def __init__(
        self, encoder_config, architecture="tabm", width=128, dropout=0.2,
        members=8, positive_rate=0.05, rank=32, experts=1, gating=True,
    ):
        super().__init__()
        self.encoder = FeatureEncoder(**encoder_config)
        dimension = self.encoder.output_dim
        if architecture == "tabm":
            from tabm import TabM

            self.backbone = TabM.make(
                n_num_features=dimension, d_out=1, k=members,
                n_blocks=2, d_block=width, dropout=dropout,
            )
            bias = self.backbone.output.bias
        elif architecture == "dcnv2":
            self.backbone = DCNv2Backbone(dimension, width, dropout, rank, experts)
            bias = self.backbone.output.bias
        elif architecture == "finalmlp":
            self.backbone = FinalMLPBackbone(dimension, width, dropout, gating=gating)
            bias = self.backbone.bias
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        with torch.no_grad():
            bias.fill_(float(np.log(positive_rate / (1 - positive_rate))))
        self.architecture = architecture

    def forward(self, numeric, categorical):
        output = self.backbone(self.encoder(numeric, categorical))
        return output.squeeze(-1) if self.architecture == "tabm" else output

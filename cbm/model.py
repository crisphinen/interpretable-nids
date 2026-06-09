# cbm model variants for iot nids.
# 1. mlpbaseline   - standard mlp, direct classification, no concepts
# 2. jointcbm      - encoder -> concept heads + classifier, jointly trained
# 3. sequentialcbm - concept predictors trained first, classifier on predicted concepts
# 4. hybridcbm     - jointcbm with additional skip connection from embedding to classifier

import torch
import torch.nn as nn
import torch.nn.functional as F


# shared encoder: 3-layer mlp, in_dim -> 256 -> 256 -> embed_dim
# with layernorm and relu activations.
class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# model 1: standard mlp with no concept bottleneck.
# encoder -> linear(embed_dim -> n_classes).
class MLPBaseline(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, embed_dim: int = 64):
        super().__init__()
        self.encoder = MLPEncoder(in_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor):
        emb = self.encoder(x)
        logits = self.classifier(emb)
        return logits, None  # (logits, concept_preds)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# model 2: end-to-end trained cbm.
# encoder -> concept_heads (sigmoid) -> classifier
# loss: ce(class) + lambda * bce(concepts)
class JointCBM(nn.Module):
    def __init__(self, in_dim: int, n_concepts: int, n_classes: int, embed_dim: int = 64):
        super().__init__()
        self.encoder = MLPEncoder(in_dim, embed_dim)
        self.concept_heads = nn.Linear(embed_dim, n_concepts)
        self.classifier = nn.Linear(n_concepts, n_classes)

    def forward(self, x: torch.Tensor):
        emb = self.encoder(x)
        concept_logits = self.concept_heads(emb)
        concept_preds = torch.sigmoid(concept_logits)
        logits = self.classifier(concept_preds)
        return logits, concept_preds

    # return concept vector (the bottleneck) for ood detection.
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        return torch.sigmoid(self.concept_heads(emb))

    # set concept at concept_idx to concept_value for all samples,
    # run classifier, return class logits.
    def intervene(self, x: torch.Tensor, concept_idx: int, concept_value: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        concept_preds = torch.sigmoid(self.concept_heads(emb))
        concept_preds = concept_preds.clone()
        concept_preds[:, concept_idx] = concept_value
        return self.classifier(concept_preds)


# model 3: two-stage training.
# stage 1: train encoder + concept_heads with bce
# stage 2: freeze encoder+concept_heads, train classifier on predicted concepts
# at inference: encoder -> concept_preds -> classifier.
class SequentialCBM(nn.Module):
    def __init__(self, in_dim: int, n_concepts: int, n_classes: int, embed_dim: int = 64):
        super().__init__()
        self.encoder = MLPEncoder(in_dim, embed_dim)
        self.concept_heads = nn.Linear(embed_dim, n_concepts)
        self.classifier = nn.Linear(n_concepts, n_classes)

    def forward(self, x: torch.Tensor):
        emb = self.encoder(x)
        concept_logits = self.concept_heads(emb)
        concept_preds = torch.sigmoid(concept_logits)
        logits = self.classifier(concept_preds)
        return logits, concept_preds

    # stage 1: forward pass for concept prediction only.
    def forward_concepts_only(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        return torch.sigmoid(self.concept_heads(emb))

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        return torch.sigmoid(self.concept_heads(emb))

    def intervene(self, x: torch.Tensor, concept_idx: int, concept_value: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        concept_preds = torch.sigmoid(self.concept_heads(emb))
        concept_preds = concept_preds.clone()
        concept_preds[:, concept_idx] = concept_value
        return self.classifier(concept_preds)

    # freeze encoder and concept heads for stage-2 classifier training.
    def freeze_concept_stage(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.concept_heads.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


# model 4: jointcbm with a skip connection.
# classifier receives [concept_preds || embedding] concatenated.
class HybridCBM(nn.Module):
    def __init__(self, in_dim: int, n_concepts: int, n_classes: int, embed_dim: int = 64):
        super().__init__()
        self.encoder = MLPEncoder(in_dim, embed_dim)
        self.concept_heads = nn.Linear(embed_dim, n_concepts)
        # classifier gets concept vector + raw embedding
        self.classifier = nn.Linear(n_concepts + embed_dim, n_classes)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor):
        emb = self.encoder(x)
        concept_logits = self.concept_heads(emb)
        concept_preds = torch.sigmoid(concept_logits)
        combined = torch.cat([concept_preds, emb], dim=-1)
        logits = self.classifier(combined)
        return logits, concept_preds

    # return concept vector for ood detection.
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        return torch.sigmoid(self.concept_heads(emb))

    def intervene(self, x: torch.Tensor, concept_idx: int, concept_value: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        concept_preds = torch.sigmoid(self.concept_heads(emb))
        concept_preds = concept_preds.clone()
        concept_preds[:, concept_idx] = concept_value
        combined = torch.cat([concept_preds, emb], dim=-1)
        return self.classifier(combined)

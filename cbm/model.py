import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MLPBaseline(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, embed_dim: int = 64):
        super().__init__()
        self.encoder = MLPEncoder(in_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor):
        emb = self.encoder(x)
        logits = self.classifier(emb)
        return logits, None

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class JointCBM(nn.Module):
    """encoder → concept_heads (sigmoid) → classifier; loss = CE + λ·BCE"""
    def __init__(self, in_dim: int, n_concepts: int, n_classes: int, embed_dim: int = 64):
        super().__init__()
        self.encoder = MLPEncoder(in_dim, embed_dim)
        self.concept_heads = nn.Linear(embed_dim, n_concepts)
        self.classifier = nn.Linear(n_concepts, n_classes)

    def forward(self, x: torch.Tensor):
        emb = self.encoder(x)
        concept_preds = torch.sigmoid(self.concept_heads(emb))
        logits = self.classifier(concept_preds)
        return logits, concept_preds

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        return torch.sigmoid(self.concept_heads(emb))

    def intervene(self, x: torch.Tensor, concept_idx: int, concept_value: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        concept_preds = torch.sigmoid(self.concept_heads(emb)).clone()
        concept_preds[:, concept_idx] = concept_value
        return self.classifier(concept_preds)


class SequentialCBM(nn.Module):
    def __init__(self, in_dim: int, n_concepts: int, n_classes: int, embed_dim: int = 64):
        super().__init__()
        self.encoder = MLPEncoder(in_dim, embed_dim)
        self.concept_heads = nn.Linear(embed_dim, n_concepts)
        self.classifier = nn.Linear(n_concepts, n_classes)

    def forward(self, x: torch.Tensor):
        emb = self.encoder(x)
        concept_preds = torch.sigmoid(self.concept_heads(emb))
        logits = self.classifier(concept_preds)
        return logits, concept_preds

    def forward_concepts_only(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        return torch.sigmoid(self.concept_heads(emb))

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        return torch.sigmoid(self.concept_heads(emb))

    def intervene(self, x: torch.Tensor, concept_idx: int, concept_value: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        concept_preds = torch.sigmoid(self.concept_heads(emb)).clone()
        concept_preds[:, concept_idx] = concept_value
        return self.classifier(concept_preds)

    def freeze_concept_stage(self):
        for p in list(self.encoder.parameters()) + list(self.concept_heads.parameters()):
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


class HybridCBM(nn.Module):
    """JointCBM with skip: classifier receives [concept_preds ‖ embedding]."""
    def __init__(self, in_dim: int, n_concepts: int, n_classes: int, embed_dim: int = 64):
        super().__init__()
        self.encoder = MLPEncoder(in_dim, embed_dim)
        self.concept_heads = nn.Linear(embed_dim, n_concepts)
        self.classifier = nn.Linear(n_concepts + embed_dim, n_classes)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor):
        emb = self.encoder(x)
        concept_preds = torch.sigmoid(self.concept_heads(emb))
        logits = self.classifier(torch.cat([concept_preds, emb], dim=-1))
        return logits, concept_preds

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        return torch.sigmoid(self.concept_heads(emb))

    def intervene(self, x: torch.Tensor, concept_idx: int, concept_value: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        concept_preds = torch.sigmoid(self.concept_heads(emb)).clone()
        concept_preds[:, concept_idx] = concept_value
        return self.classifier(torch.cat([concept_preds, emb], dim=-1))

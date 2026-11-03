from src.models.bilstm import BiLSTMClassifier
from src.models.transformer import TransformerClassifier


def build_model(name, input_dim, num_classes, hidden=128):
    if name == "bilstm":
        return BiLSTMClassifier(input_dim, num_classes, hidden=hidden)
    if name == "transformer":
        return TransformerClassifier(input_dim, num_classes, d_model=hidden)
    raise ValueError(f"unknown model: {name}")
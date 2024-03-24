from transformers import AutoModel, AutoTokenizer, AutoProcessor
import torch

# Global cache for models to avoid re-loading
_model_cache = {}


def init_model(model_name):
    """
    Initialize the SigLIP model and processor with caching.
    """
    if model_name not in _model_cache:
        model = AutoModel.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        _model_cache[model_name] = (model, processor)
    return _model_cache[model_name]


def get_model():
    """
    Wrapper function to get the SigLIP model and processor with caching.
    """
    model_name = "google/siglip-so400m-patch14-384"
    if model_name not in _model_cache:
        init_model(model_name)
    return _model_cache[model_name]


def embed_text(texts):
    """
    Embed text using the SigLIP model.
    """
    model, processor = get_model()
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state  # [Batch, Seq_len, Emb_dim]


def embed_image(images):
    """
    Embed images using the SigLIP model.
    """
    model, processor = get_model()
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state  # [Batch, Seq_len, Emb_dim]

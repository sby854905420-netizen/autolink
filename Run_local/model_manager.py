import threading
import os
from sentence_transformers import SentenceTransformer
import torch
from progress_logger import get_progress_logger

class ModelManager:    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.model = None
            self.device = None
            self.model_lock = threading.Lock()
            self.initialized = True
    
    def load_model(self, model_path: str = None, device: str = "cuda:0"):
        if model_path is None:
            model_path = os.environ.get("SENTENCE_TRANSFORMER_MODEL", "BAAI/bge-large-en-v1.5")
                    
        with self.model_lock:
            if self.model is None or self.device != device:
                logger = get_progress_logger(__name__)
                logger.info("Loading embedding model | model=%s device=%s", model_path, device)
                if device.startswith("cuda") and not torch.cuda.is_available():
                    device = "cpu"
                    logger.warning("CUDA not available, falling back to CPU")
                
                self.model = SentenceTransformer(model_path, device=device)
                self.device = device
                logger.info("Embedding model loaded | device=%s", device)
                
                if device.startswith("cuda"):
                    gpu_id = int(device.split(":")[-1])
                    memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                    logger.info(
                        "Embedding GPU memory | gpu=%s allocated=%.2fGB reserved=%.2fGB",
                        gpu_id,
                        memory_allocated,
                        memory_reserved,
                    )
    
    def get_model(self):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model
    
    def encode(self, text: str):
        with self.model_lock:
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load_model() first.")
            return self.model.encode(text, convert_to_numpy=True)
    
    def get_device(self):
        return self.device
    
    def get_memory_usage(self):
        if self.device and self.device.startswith("cuda"):
            gpu_id = int(self.device.split(":")[-1])
            memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
            return {
                "allocated": memory_allocated,
                "reserved": memory_reserved,
                "device": self.device
            }
        return None

model_manager = ModelManager()

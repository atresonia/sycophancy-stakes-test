import numpy as np
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_tag(text: str) -> np.ndarray:
    vector = embed_model.encode(text.replace("_", " ").lower(), normalize_embeddings=True)
    return np.asarray(vector, dtype=np.float32)


class GroupScores:
    """Groups strategy tags by similarity for scoring"""
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.grouping: dict[str, np.ndarray] = {}
    
    def route(self, tag: str) -> str:
        """Route a tag to a group based on similarity to group vectors"""
        vector = embed_tag(tag)
        best_label, best_sim = None, -1.0
        for label, gv in self.grouping.items():
            sim = float(np.dot(vector, gv) / (np.linalg.norm(vector) * np.linalg.norm(gv)))
            if sim > best_sim:
                best_label, best_sim = label, sim
        if best_sim >= self.threshold:
            return best_label # route to most similar group
        self.grouping[tag] = vector # new group
        return tag # new group
from __future__ import annotations

from .shared import *  # noqa: F401,F403

def make_class_weights(labels: np.ndarray, class_count: int = 3) -> torch.Tensor:
    counts = np.bincount(labels, minlength=class_count).astype(np.float32)
    weights = np.ones(class_count, dtype=np.float32)
    total = counts.sum()
    for cls in range(class_count):
        if counts[cls] > 0:
            weights[cls] = total / (float(class_count) * counts[cls])
    return torch.tensor(weights, dtype=torch.float32)

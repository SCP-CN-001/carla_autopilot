import numpy as np


def normalize_depth(data):
    """
    Computes the normalized depth from a CARLA depth map.
    """
    data = data.astype(np.float64)

    normalized = np.dot(data, [65536.0, 256.0, 1.0])
    normalized /= 256 * 256 * 256 - 1
    normalized = np.clip(normalized, a_min=0.0, a_max=0.05)
    normalized = normalized * 20.0  # Rescale map to lie in [0,1]

    return normalized

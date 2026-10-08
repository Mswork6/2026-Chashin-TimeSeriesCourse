import numpy as np

from modules.utils import *


def top_k_discords(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k discords based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k: number of discords

    Returns
    --------
    discords: top-k discords (indices, distances to its nearest neighbor and the nearest neighbors indices)
    """
 
    discords_idx = []
    discords_dist = []
    discords_nn_idx = []

    discords_idx = []
    discords_dist = []
    discords_nn_idx = []

    # Копируем, чтобы не портить исходный профиль
    mp = matrix_profile['mp'].copy()
    mpi = matrix_profile['mpi'].copy()
    excl_zone = matrix_profile['excl_zone']

    for _ in range(top_k):
        # Находим диссонанс с максимальным расстоянием до ближайшего соседа
        discord_idx = int(np.argmax(mp))
        discord_dist = float(mp[discord_idx])
        nn_idx = int(mpi[discord_idx])

        discords_idx.append(discord_idx)
        discords_dist.append(discord_dist)
        discords_nn_idx.append(nn_idx)

        # Затираем окрестность найденного диссонанса
        mp = apply_exclusion_zone(mp, discord_idx, excl_zone, -np.inf)

    return {
        'indices' : discords_idx,
        'distances' : discords_dist,
        'nn_indices' : discords_nn_idx
        }

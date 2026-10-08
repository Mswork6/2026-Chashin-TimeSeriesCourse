import numpy as np

from modules.utils import *


def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k motifs based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k : number of motifs

    Returns
    --------
    motifs: top-k motifs (left and right indices and distances)
    """

    motifs_idx = []
    motifs_dist = []

    # Копируем матричный профиль, чтобы не портить оригинал
    mp = matrix_profile['mp'].copy()
    mpi = matrix_profile['mpi'].copy()
    excl_zone = matrix_profile['excl_zone']

    for _ in range(top_k):
        # Находим индекс мотива с минимальным расстоянием
        motif_idx = np.argmin(mp)
        motif_dist = mp[motif_idx]
        
        # Находим его ближайшего соседа
        nn_idx = int(mpi[motif_idx])
        
        # Сохраняем результат
        # motifs_idx.append([int(motif_idx), nn_idx])
        motifs_idx.append(sorted([int(motif_idx), nn_idx]))
        motifs_dist.append(float(motif_dist))
        
        # Применяем зону исключения вокруг обоих найденных индексов,
        # чтобы не найти тот же самый мотив снова
        mp = apply_exclusion_zone(mp, motif_idx, excl_zone, np.inf)
        mp = apply_exclusion_zone(mp, nn_idx, excl_zone, np.inf)

    return {
        "indices" : motifs_idx,
        "distances" : motifs_dist
        }

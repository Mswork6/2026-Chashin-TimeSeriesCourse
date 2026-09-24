import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """

    # Вычисляем рызность и возводим в квадрат
    squared_diff = (ts1 - ts2) ** 2

    # Вычисляем сумму квадратов разностей и извлекаем корень
    ed_dist = np.sqrt(np.sum(squared_diff))

    return ed_dist


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    n = len(ts1)

    # Вычисляем средние значения временных рядов
    mean1 = np.mean(ts1)
    mean2 = np.mean(ts2)

    # Вычисляем стандартные отклонения
    std1 = np.std(ts1)
    std2 = np.std(ts2)

    # Вычисляем скалярное произведение рядов
    dot_product = np.dot(ts1, ts2)

    # Вычисляем нормализованное евклидово расстояние по формуле
    correlation = (dot_product - n * mean1 * mean2) / (n * std1 * std2)

    norm_ed_dist = np.sqrt(np.abs(2 * n * (1 - correlation)))

    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    n = len(ts1)
    m = len(ts2)

    # Переводим относительную ширину полосы Сако—Чиба
    # в количество элементов временного ряда
    window = int(r * max(n, m))

    window = max(window, abs(n - m))

    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):

        # Определяем границы полосы Сако—Чиба.
        # Вычисления выполняются только около главной диагонали матрицы
        j_start = max(1, i - window)
        j_end = min(m, i + window)

        for j in range(j_start, j_end + 1):
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2

            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1]
            )

    dtw_dist = dtw_matrix[n, m]

    return dtw_dist

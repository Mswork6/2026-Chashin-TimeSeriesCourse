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

    # Задаем матрицу, заполненную бесконечностями
    dtw_matrix = np.full((n + 1, m + 1), np.inf)

    # В начальной точке стоимость 0
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Считаем квадрат разности между элементами
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2

            # Берем минимум из трех возможных путей
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],  # сдвиг в первом ряду
                dtw_matrix[i, j - 1],  # сдвиг во втором ряду
                dtw_matrix[i - 1, j - 1]  # соответствие
            )

    # Возвращаем корень из значения в правом нижнем углу
    return dtw_matrix[n, m]

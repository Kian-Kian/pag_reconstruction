from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation
import heapq

from .data import component_boundary_mask


def balanced_boundary_loss(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, pos_weight: float
) -> torch.Tensor:
    if not torch.any(valid):
        return logits.sum() * 0.0
    weight = torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits[valid], target[valid], pos_weight=weight)


def binary_metrics(probability: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, float]:
    target = target.astype(bool)
    prediction = probability >= threshold
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    tn = int(np.count_nonzero(~prediction & ~target))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(len(target), 1)
    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def best_f1_threshold(probability: np.ndarray, target: np.ndarray) -> tuple[float, dict[str, float]]:
    candidates = np.linspace(0.05, 0.95, 91)
    scored = [(float(value), binary_metrics(probability, target, float(value)))
              for value in candidates]
    threshold, metrics = max(scored, key=lambda item: (item[1]["f1"], item[1]["recall"]))
    return threshold, metrics


def roc_auc(probability: np.ndarray, target: np.ndarray) -> float:
    target = target.astype(bool)
    positive = int(target.sum())
    negative = len(target) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(probability, kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    sorted_probability = probability[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_probability[end] == sorted_probability[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return float((ranks[target].sum() - positive * (positive + 1) / 2) / (positive * negative))


def average_precision(probability: np.ndarray, target: np.ndarray) -> float:
    target = target.astype(bool)
    positive = int(target.sum())
    if positive == 0:
        return float("nan")
    order = np.argsort(-probability, kind="mergesort")
    sorted_target = target[order]
    precision = np.cumsum(sorted_target) / np.arange(1, len(target) + 1)
    return float(precision[sorted_target].sum() / positive)


def gaec_reconstruct_components(
    edge_index: np.ndarray,
    boundary_probability: np.ndarray,
    node_area: np.ndarray,
    shared_boundary: np.ndarray,
    merge_same_probability: float = 0.4,
    max_region_fraction: float = 0.8,
) -> tuple[np.ndarray, dict[str, int]]:
    """Additive edge contraction without small-region cleanup.

    Positive interface costs favor contraction. When two components merge, all
    interface costs to a common neighbor are added, so later decisions use the
    complete component boundary rather than a single edge.
    """

    node_area = np.asarray(node_area, dtype=np.float64)
    probability = np.clip(np.asarray(boundary_probability, dtype=np.float64), 1e-5, 1 - 1e-5)
    same_probability = 1.0 - probability
    weight = np.maximum(np.asarray(shared_boundary, dtype=np.float64), 1.0)
    prior = float(np.clip(merge_same_probability, 1e-5, 1 - 1e-5))
    prior_logit = np.log(prior / (1.0 - prior))
    edge_cost = weight * (np.log(same_probability / probability) - prior_logit)

    node_count = len(node_area)
    parent = np.arange(node_count, dtype=np.int64)
    component_area = node_area.copy()
    adjacency: list[dict[int, float]] = [dict() for _ in range(node_count)]
    heap: list[tuple[float, int, int]] = []
    for edge_id in range(edge_index.shape[1]):
        first = int(edge_index[0, edge_id])
        second = int(edge_index[1, edge_id])
        cost = float(edge_cost[edge_id])
        adjacency[first][second] = adjacency[first].get(second, 0.0) + cost
        adjacency[second][first] = adjacency[second].get(first, 0.0) + cost
    for first in range(node_count):
        for second, cost in adjacency[first].items():
            if first < second:
                heapq.heappush(heap, (-cost, first, second))

    maximum_area = max_region_fraction * float(node_area.sum())
    gaec_merges = area_guard_rejections = 0
    while heap:
        negative_cost, first, second = heapq.heappop(heap)
        if parent[first] != first or parent[second] != second:
            continue
        current = adjacency[first].get(second)
        if current is None or not np.isclose(current, -negative_cost, rtol=1e-10, atol=1e-10):
            continue
        if current <= 0.0:
            break
        del adjacency[first][second]
        del adjacency[second][first]
        if component_area[first] + component_area[second] > maximum_area:
            area_guard_rejections += 1
            continue
        if len(adjacency[first]) < len(adjacency[second]):
            first, second = second, first
        parent[second] = first
        component_area[first] += component_area[second]
        second_neighbors = list(adjacency[second].items())
        for neighbor, second_cost in second_neighbors:
            if parent[neighbor] != neighbor or neighbor == first:
                continue
            first_cost = adjacency[first].get(neighbor, 0.0)
            adjacency[neighbor].pop(second, None)
            combined = first_cost + second_cost
            adjacency[first][neighbor] = combined
            adjacency[neighbor][first] = combined
            heapq.heappush(heap, (-combined, min(first, neighbor), max(first, neighbor)))
        adjacency[second].clear()
        gaec_merges += 1

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = int(parent[root])
        while parent[value] != value:
            next_value = int(parent[value])
            parent[value] = root
            value = next_value
        return root

    rescue_merges = 0
    roots = np.asarray([find(index) for index in range(node_count)])
    _, labels = np.unique(roots, return_inverse=True)
    report = {
        "gaec_merges": gaec_merges,
        "rescue_merges": rescue_merges,
        "area_guard_rejections": area_guard_rejections,
        "final_components": int(np.unique(labels).size),
    }
    return labels.astype(np.int32), report


def adjusted_rand_index(first: np.ndarray, second: np.ndarray) -> float:
    _, first = np.unique(first, return_inverse=True)
    _, second = np.unique(second, return_inverse=True)
    pairs, counts = np.unique(np.stack([first, second], axis=1), axis=0, return_counts=True)
    first_counts = np.bincount(first)
    second_counts = np.bincount(second)
    def choose2(values): return np.sum(values * (values - 1) / 2.0)
    contingency = choose2(counts)
    first_sum = choose2(first_counts)
    second_sum = choose2(second_counts)
    total = len(first)
    total_pairs = total * (total - 1) / 2.0
    expected = first_sum * second_sum / max(total_pairs, 1.0)
    maximum = 0.5 * (first_sum + second_sum)
    return float((contingency - expected) / max(maximum - expected, 1e-12))


def _entropy_metrics(first: np.ndarray, second: np.ndarray) -> tuple[float, float, float]:
    _, first_codes = np.unique(first, return_inverse=True)
    _, second_codes = np.unique(second, return_inverse=True)
    pairs, counts = np.unique(
        np.stack([first_codes, second_codes], axis=1), axis=0, return_counts=True
    )
    total = float(len(first_codes))
    first_counts = np.bincount(first_codes)
    second_counts = np.bincount(second_codes)
    p_first = first_counts / total
    p_second = second_counts / total
    joint = counts / total
    entropy_first = -float(np.sum(p_first * np.log(p_first)))
    entropy_second = -float(np.sum(p_second * np.log(p_second)))
    mutual = float(
        np.sum(
            joint
            * np.log(joint / (p_first[pairs[:, 0]] * p_second[pairs[:, 1]]))
        )
    )
    return entropy_first, entropy_second, mutual


def segmentation_metrics(
    predicted_node_labels: np.ndarray,
    alpha_grain_id: np.ndarray,
    true_parent_id: np.ndarray,
    gamma_grain_id: np.ndarray,
    tolerance: int = 2,
) -> dict[str, float]:
    clean_nodes = true_parent_id >= 0
    pixel_nodes = alpha_grain_id
    valid = (pixel_nodes >= 0) & clean_nodes[np.maximum(pixel_nodes, 0)]
    if not np.any(valid):
        raise ValueError("No pixels have high-confidence parent labels")
    predicted_pixels = predicted_node_labels[pixel_nodes[valid]]
    true_pixels = true_parent_id[pixel_nodes[valid]]
    ari = adjusted_rand_index(predicted_pixels, true_pixels)
    entropy_pred, entropy_true, mutual = _entropy_metrics(predicted_pixels, true_pixels)
    vi = entropy_pred + entropy_true - 2.0 * mutual

    predicted_map = np.full(alpha_grain_id.shape, -1, dtype=np.int32)
    alpha_valid = alpha_grain_id >= 0
    predicted_map[alpha_valid] = predicted_node_labels[alpha_grain_id[alpha_valid]]
    predicted_boundary = component_boundary_mask(predicted_map)
    true_boundary = component_boundary_mask(gamma_grain_id)
    dilated_true = binary_dilation(true_boundary, iterations=tolerance)
    dilated_predicted = binary_dilation(predicted_boundary, iterations=tolerance)
    precision = np.count_nonzero(predicted_boundary & dilated_true) / max(
        np.count_nonzero(predicted_boundary), 1
    )
    recall = np.count_nonzero(true_boundary & dilated_predicted) / max(
        np.count_nonzero(true_boundary), 1
    )
    boundary_f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    predicted_count = len(np.unique(predicted_pixels))
    true_count = len(np.unique(true_pixels))
    predicted_areas = np.bincount(predicted_pixels)
    true_areas = np.bincount(np.unique(true_pixels, return_inverse=True)[1])
    return {
        "pixel_ari": ari,
        "pixel_vi": float(vi),
        "boundary_tolerant_precision": float(precision),
        "boundary_tolerant_recall": float(recall),
        "boundary_tolerant_f1": float(boundary_f1),
        "predicted_grains": int(predicted_count),
        "true_grains": int(true_count),
        "grain_count_relative_error": (predicted_count - true_count) / max(true_count, 1),
        "largest_predicted_fraction": float(predicted_areas.max() / predicted_areas.sum()),
        "largest_true_fraction": float(true_areas.max() / true_areas.sum()),
    }

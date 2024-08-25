import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

from utils import load_json, save_json
from settings import QUERIES_RESULTS_FILE
from logger_config import logger


def combsum_fusion(ranked_lists):
    """
    Combines multiple ranked lists into a fused list using the CombSUM fusion method.

    Args:
        ranked_lists (list): A list of ranked lists, where each ranked list is a list of (doc_id, score) tuples.

    Returns:
        list: The fused list of (doc_id, score) tuples, sorted by score in descending order.
    """
    # Create a set of all unique document IDs
    all_doc_ids = set(
        doc_id for ranked_list in ranked_lists for _, doc_id in ranked_list)

    # Create a dictionary to map document IDs to their index in the numpy array
    doc_id_to_index = {doc_id: i for i, doc_id in enumerate(all_doc_ids)}

    # Initialize a numpy array to hold all scores
    scores_array = np.zeros((len(all_doc_ids), len(ranked_lists)))

    # Fill the scores array
    for list_idx, ranked_list in enumerate(ranked_lists):
        for score, doc_id in ranked_list:
            # reverse the score, because it's simillary score
            scores_array[doc_id_to_index[doc_id], list_idx] = 1/ score

    # Normalize scores using MinMaxScaler
    scaler = MinMaxScaler()
    normalized_scores = scaler.fit_transform(scores_array)

    # Sum up the normalized scores
    fused_scores = np.sum(normalized_scores, axis=1)

    # Create the fused list of (doc_id, score) tuples
    fused_list = [(doc_id, fused_scores[idx])
                  for doc_id, idx in doc_id_to_index.items()]

    # Sort the fused list by score in descending order
    fused_list.sort(key=lambda x: x[1], reverse=True)
    fused_list = [[score, doc_id] for doc_id, score in fused_list]
    return fused_list


def rrf_fusion(ranked_lists, k=60):
    """
    Combines multiple ranked lists into a fused list using the Reciprocal Rank Fusion (RRF) method.

    Args:
        ranked_lists (list): A list of ranked lists, where each ranked list is a list of (doc_id, score) tuples.
        k (int): The rank weight parameter (default is 60).

    Returns:
        list: The fused list of (doc_id, score) tuples, sorted by score in descending order.
    """
    fused_scores = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, (_, doc_id) in enumerate(ranked_list, start=1):
            fused_scores[doc_id] += 1 / (k + rank)

    # Sort the fused list by score in descending order
    fused_list = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    fused_list = [[score, doc_id] for doc_id, score in fused_list]
    return fused_list


def comb_results(queries_results, output_path, fusion_method):
    try:
        fusion_method = FUSION_TO_FUNC_MAPPER[fusion_method]
    except KeyError:
        logger.error("Invalid fusion method: %s, Please choose from the following methods: %s", fusion_method,
                     list(FUSION_TO_FUNC_MAPPER.keys()))
        return
    fusion_result = {}
    first_signal = list(queries_results.keys())[0]
    for query_id in queries_results[first_signal]:
        ranked_lists = [queries_results[signal][query_id]
                        for signal in queries_results]
        fusion_result[query_id] = fusion_method(ranked_lists)
    save_json(fusion_result, output_path / QUERIES_RESULTS_FILE)


FUSION_TO_FUNC_MAPPER = {
    "RRF": rrf_fusion,
    "CombSum": combsum_fusion,
}

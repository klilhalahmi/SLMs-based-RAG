from utils import save_json, load_json
from settings import QUERIES_RESULTS_FILE, METRICS_CSV_FILE, METRICS_AVERAGE_JSON_FILE
import pandas as pd


def calculate_ir_metrics(result_ids, relevant_ids):
    """
    Calculates information retrieval metrics based on the given result IDs and relevant IDs.

    Parameters:
    - result_ids (list): A list of result IDs.
    - relevant_ids (list): A list of relevant IDs.

    Returns:
    - dict: A dictionary containing the following metrics:
        - "ap" (float): Average Precision.
        - "p3" (float): Precision at 3.
        - "p5" (float): Precision at 5.
        - "recall" (float): Recall.
    """
    relevant_set = set(relevant_ids)
    num_relevant = len(relevant_set)

    if num_relevant == 0:
        return {"ap": 0.0, "p3": 0.0, "p5": 0.0, "recall": 0.0}

    hits = 0
    sum_precisions = 0.0
    p3, p5 = 0.0, 0.0

    for i, result in enumerate(result_ids, 1):
        if result in relevant_set:
            hits += 1
            precision = hits / i
            sum_precisions += precision

        if i == 3:
            p3 = hits / i
        elif i == 5:
            p5 = hits / i

    ap = sum_precisions / num_relevant
    recall = hits / num_relevant

    return {
        "ap": ap,
        "p3": p3,
        "p5": p5,
        "recall": recall
    }


def get_query_result(query, model_manager, model):
    """
    Retrieves the search results for a given query.

    Args:
        query (str): The query string.
        model_manager (ModelManager): The model manager object.
        model (Model): The model object.

    Returns:
        list: A list of search results.
    """
    query_embedding = model.encode(query)
    results = model_manager.search(query_embedding)
    return results


def get_queries_results(queries, model_manager, output_path, model):
    """
    Retrieves the results of multiple queries and saves them as a JSON file.

    Args:
        queries (dict): A dictionary containing query IDs as keys and queries as values.
        model_manager: The model manager object used to retrieve query results.
        output_path (str): The file path where the JSON file will be saved.
        model: The model object used to retrieve query results.

    Returns:
        None
    """
    result_dict = {}
    for query_id, query in queries.items():
        results = get_query_result(query, model_manager, model)
        result_dict[query_id] = results
    save_json(result_dict, output_path)


def calculate_metrics(output_path, qrels):
    """
    Calculate metrics for each model based on the given configurations, output path, and relevance judgments.

    Args:
        configs (object): The configurations object containing model names.
        output_path (str): The path to the output directory.
        qrels (dict): The relevance judgments for each query.

    Returns:
        None
    """
    results = load_json(output_path / QUERIES_RESULTS_FILE)
    metrics = {}
    for query_id, results in results.items():
        relevant_ids = list(qrels[query_id].keys())
        result_ids = [doc_id for _, doc_id in results]
        metrics[query_id] = calculate_ir_metrics(
            result_ids, relevant_ids)
    df = pd.DataFrame(metrics).T
    df.to_csv(output_path / METRICS_CSV_FILE)
    average_metrics = df.mean().to_dict()
    save_json(average_metrics, output_path / METRICS_AVERAGE_JSON_FILE)

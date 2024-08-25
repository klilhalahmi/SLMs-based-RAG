import pandas as pd
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from pathlib import Path

from fusion import comb_results
from eval import calculate_metrics
from index_manager import SingleIndexManager, index_corpus
from utils import save_json, load_yaml, load_json
from configs import Config
from settings import (CONFIG_FILE, INDEX_FOLDER, DATASET_FOLDER,
                      QUERIES_RESULTS_FILE, METRICS_AVERAGE_JSON_FILE, RESULTS_CSV_FILE)
from logger_config import logger


def get_model_basename(model_name):
    """
    Get the base name of a model from its full name.

    Args:
        model_name (str): The full name of the model.

    Returns:
        str: The base name of the model.
    """
    if "/" not in model_name:
        return model_name
    return model_name.split("/", 1)[1]


def download_dataset(dataset_name, out_dir):
    """
    Downloads a dataset from a given URL and unzips it to the specified output directory.

    Args:
        dataset_name (str): The name of the dataset to download.
        out_dir (str): The output directory where the dataset will be saved.

    Returns:
        data_path (str): The path to the downloaded and unzipped dataset.

    """
    datset_path = out_dir / DATASET_FOLDER
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        dataset_name)
    data_path = util.download_and_unzip(url, datset_path)
    return data_path


def get_dataset(dataset_name, out_dir):
    """
    Downloads the specified dataset and returns the corpus, queries, and qrels.

    Args:
        dataset_name (str): The name of the dataset to download.
        out_dir (str): The output directory where the dataset will be saved.

    Returns:
        tuple: A tuple containing the corpus, queries, and qrels.
    """
    data_path = download_dataset(dataset_name, out_dir)
    corpus, queries, qrels = GenericDataLoader(
        data_folder=data_path).load(split="test")
    return corpus, queries, qrels


def get_model_query_result_on_dataset(model, queries, model_manager, output_path):
    """
    Retrieves the query results for a given model on a dataset.

    Args:
        model (object): The model used for encoding queries.
        queries (dict): A dictionary of query IDs and their corresponding queries.
        model_manager (object): The manager object for the model.
        output_path (str): The path to save the query results.

    Returns:
        None
    """
    queries_result_path = output_path / QUERIES_RESULTS_FILE
    results = {}
    for query_id, query in queries.items():
        query_embedding = model.encode(query)
        results[query_id] = model_manager.search(query_embedding)
    save_json(results, queries_result_path)


def get_all_models_results_on_dataset(configs, output_path):
    """
    Retrieves the results of all SLMs models on a dataset.

    Args:
        configs (object): The configuration object.
        output_path (str): The path to save the results.

    Returns:
        None
    """
    results = defaultdict(dict)
    for model_name in configs.slm_models_names:
        model_folder = get_model_basename(model_name)
        model_output_folder = output_path / model_folder
        queries_result_path = model_output_folder / QUERIES_RESULTS_FILE
        results[model_folder] = load_json(queries_result_path)
    return results


def calculate_dataset_performance(dataset_name, configs):
    """
    Calculate the performance of a dataset on all models.

    Args:
        dataset_name (str): The name of the dataset.
        configs (object): The configuration object.

    Returns:
        None
    """
    dataset_output_folder = configs.output_folder / dataset_name
    dataset_output_folder.mkdir(parents=True, exist_ok=True)
    corpus, queries, qrels = get_dataset(
        dataset_name, dataset_output_folder)
    # # first get the results for all slm models
    for model_name in configs.slm_models_names:
        model_eval(configs, corpus, queries, model_name,
                   qrels, dataset_output_folder)
    # then get the results for the llm model
    model_eval(configs, corpus, queries, configs.llm_model_name,
               qrels, dataset_output_folder)
    # # get queries results per each SLM model
    results = get_all_models_results_on_dataset(configs, dataset_output_folder)
    # Create the fusion results
    for fusion_method in configs.fusion_methods:
        logger.info("Fusing the results using %s", fusion_method)
        fusion_output_folder = dataset_output_folder / fusion_method
        fusion_output_folder.mkdir(parents=True, exist_ok=True)
        comb_results(results, fusion_output_folder, fusion_method)
        calculate_metrics(fusion_output_folder, qrels)
    aggregate_results(configs, dataset_output_folder)


def get_model_avg_results(output_path, model_name):
    """
    Retrieve the average results of a model.

    Args:
        output_path (str): The path to the output directory.
        model_name (str): The name of the model.

    Returns:
        dict: The average results of the model.
    """
    model_folder = get_model_basename(model_name)
    model_output_folder = output_path / model_folder
    queries_result_path = model_output_folder / METRICS_AVERAGE_JSON_FILE
    return load_json(queries_result_path)


def aggregate_results(configs, output_path):
    """
    Aggregate the results of different models and fusion methods.

    Args:
        configs (object): The configurations object containing the model names and fusion methods.
        output_path (str): The path to save the aggregated results.

    Returns:
        None
    """
    all_models = configs.slm_models_names + \
        [configs.llm_model_name] + configs.fusion_methods
    models_name = [get_model_basename(model_name) for model_name in all_models]
    results = {model_name: get_model_avg_results(
        output_path, model_name) for model_name in models_name}
    results_df = pd.DataFrame.from_dict(results, orient="index").round(3)
    results_df.to_csv(output_path / RESULTS_CSV_FILE)


def model_eval(configs, corpus, queries, model_name, qrels, dataset_output_folder):
    """
    Evaluates a given model on a corpus and queries.

    Args:
        configs (object): Configuration object containing settings.
        corpus (list): List of sentences in the corpus.
        queries (list): List of query sentences.
        model_name (str): Name of the model to be evaluated.

    Returns:
        None
    """
    logger.info("Calculating results for %s", model_name)
    model_folder = get_model_basename(model_name)
    model_output_folder = dataset_output_folder / model_folder
    model_output_folder.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name)
    model_manager = SingleIndexManager(
        model.get_sentence_embedding_dimension())
    indexes_path = configs.output_folder / \
        INDEX_FOLDER if configs.save_index else None
    index_corpus(corpus, model, model_manager, indexes_path)
    get_model_query_result_on_dataset(
        model, queries, model_manager, model_output_folder)
    # calculate the metrics
    calculate_metrics(model_output_folder, qrels)


def main():
    logger.info("Initializing the experiment")
    configs_data = load_yaml(CONFIG_FILE)
    configs = Config(**configs_data)
    configs.output_folder = Path(configs.output_folder)
    for dataset_name in configs.datasets_names:
        logger.info("Running the experiment on %s", dataset_name)
        try:
            calculate_dataset_performance(dataset_name, configs)
            logger.info("Finished the experiment on %s", dataset_name)
        except Exception as e:
            logger.error("An error occurred while running the experiment on %s: %s", dataset_name, e)


if __name__ == "__main__":
    main()

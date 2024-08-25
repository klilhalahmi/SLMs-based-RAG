from dataclasses import dataclass, field


@dataclass(slots=True)
class Config:
    # Names of the SLMs we want to use
    slm_models_names: list
    # Names of the LLM we want to test them against
    llm_model_name: str
    # Name of the datasets we want to test them on
    datasets_names: list
    # Folder to all the datasets
    output_folder: str
    # Save the vector index of the dataset per model
    save_index: bool = False
    # List of methods to use for fusion, can be "RRF" or "CombSum" or both
    fusion_methods: list = field(default_factory=lambda: ["RRF", "CombSum"])

# text_gen_utils.py

from typing import List, Dict, Callable, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, GenerationConfig
import wandb
import torch

def gen_pipeline(model_name: str, text: str, generation_pipeline: Pipeline, **kwargs) -> Dict:
    """
    Generate text using a specific model pipeline and generation parameters.

    Args:
        model_name (str): Name of the model being used.
        text (str): The prompt text to generate from.
        generation_pipeline (Pipeline): Hugging Face pipeline object for text generation.
        **kwargs: Variable keyword arguments for generation parameters.

    Returns:
        dict: A dictionary containing the model name, prompt, generated text, and generation parameters.
    """
    generated_output = generation_pipeline(text, **kwargs)[0]['generated_text']
    return {
        "model_name": model_name,
        "prompt": text,
        "generated_text": generated_output,
        "gen_config": kwargs
    }

def run_single_param_experiment(model_name: str, pipeline: Pipeline, prompt: str, param_name: str, values: List, gen_function: Callable, results_dict: Dict, wandb_project: str, wandb_entity: str):
    """
    Run experiments for a single parameter across its range of values for a given prompt and model.
    Logs results to wandb, saves them in a provided dictionary, and prints the generated text.

    Args:
        model_name (str): Name of the model.
        pipeline (Pipeline): The generation pipeline.
        prompt (str): The text prompt for generation.
        param_name (str): Name of the parameter being tested.
        values (List): List of values for the parameter.
        gen_function (Callable): The generation function to use.
        results_dict (Dict): Dictionary to store the results.
        wandb_project (str): The wandb project name.
        wandb_entity (str): The wandb entity name.

    Returns:
        None
    """
    for value in values:
        run = wandb.init(project=wandb_project, entity=wandb_entity)
        text_table = wandb.Table(columns=["model_name", "prompt", "parameter", "parameter_value", "generated_text"])
        log_entry = gen_function(model_name, prompt, pipeline, **{param_name: value})
        text_table.add_data(model_name, prompt, param_name, value, log_entry['generated_text'])
        run.log({"generation": text_table})
        wandb.finish()

        # Save the results in the dictionary
        result_key = (model_name, prompt, param_name, value)
        results_dict[result_key] = log_entry

        # Print the generated text
        print(f"Model: {model_name}, Param: {param_name}, Value: {value}")
        print(f"Prompt: {prompt}")
        print(f"Generated: {log_entry['generated_text']}\n")

def run_experiments(models: Dict[str, Pipeline], dataset: List[Dict], gen_params: Dict[str, List], gen_function: Callable, prompt_field: str) -> Dict:
    """
    Run experiments across different models and generation parameters using prompts from a dataset.
    Logs results to wandb and also saves them in a dictionary.

    Args:
        models (Dict[str, Pipeline]): Dictionary of model names and their corresponding pipeline objects.
        dataset (List[Dict]): List containing the prompts.
        gen_params (Dict[str, List]): Dictionary of generation parameters and their corresponding values.
        gen_function (Callable): The function to use for generating text.
        prompt_field (str): The field in the dataset that contains the prompt text.

    Returns:
        Dict: Dictionary containing the results of all experiments.
    """
    results_dict = {}

    for model_name, pipeline in models.items():
        for data_entry in dataset:
            prompt = data_entry[prompt_field]
            for param, values in gen_params.items():
                run_single_param_experiment(model_name, pipeline, prompt, param, values, gen_function, results_dict, "your_wandb_project", "your_wandb_entity")
    return results_dict

def instantiate_huggingface_model(
    model_name: str,
    quantization_config: BitsAndBytesConfig = None,
    device_map: str = "auto",
    use_cache: bool = False,
    trust_remote_code: bool = False,
    pad_token: str = None,
    padding_side: str = "left"
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Instantiate a HuggingFace model with optional quantization using the BitsAndBytes library.

    Args:
        model_name (str): The name of the model to load from HuggingFace's model hub.
        quantization_config (BitsAndBytesConfig, optional): Configuration for model quantization.
          If None, defaults to a pre-defined quantization configuration for 4-bit quantization.
        device_map (str, optional): Device placement strategy for model layers ('auto' by default).
        use_cache (bool, optional): Whether to cache model outputs (False by default).
        trust_remote_code (bool, optional): Whether to trust remote code for custom layers (False by default).
        pad_token (str, optional): The pad token to be used by the tokenizer. If None, uses the EOS token.
        padding_side (str, optional): The side on which to pad the sequences ('left' by default).

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: The instantiated model and tokenizer.
    """
    # Default quantization configuration
    if quantization_config is None:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        use_cache=use_cache,
        trust_remote_code=trust_remote_code
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = pad_token if pad_token is not None else tokenizer.eos_token
    tokenizer.padding_side = padding_side

    return model, tokenizer

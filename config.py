from pathlib import Path

def get_config():
    
    """
    Returns the configuration settings for the training process.

    This function provides a dictionary of hyperparameters and settings used for training
    a machine learning model, including parameters for batch size, learning rate, sequence length,
    model dimensions, source and target languages, and file paths for saving models and tokenizers.

    Returns:
        dict: A dictionary containing the configuration settings.
    """
    
    return {
        "batch_size" : 16,
        "num_epochs" : 30,
        "lr" : 10**-4,
        "seq_len": 150,
        "d_model": 512,
        "lang_src" : "en",
        "lang_tgt" : "hi",
        "model_folder" : "weights",
        "model_basename" : "tmodel_",
        "preload" : None,
        "tokenizer_file" : "tokenizer_{0}.json",
        "experiment_name" : "runs/tmodel"
    }
    
def get_weights_file_path(config, epoch: str):
    
    """
    Constructs the file path for saving or loading model weights.

    Given the configuration settings and the epoch number, this function generates the appropriate
    file path for saving or loading model weights. The path is constructed based on the model folder,
    basename, and epoch provided.

    Args:
        config (dict): The configuration dictionary containing settings including the model folder and basename.
        epoch (str): The epoch number to include in the filename.

    Returns:
        str: The constructed file path for the model weights.
    """
    
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    
    return str(Path('.') / model_folder / model_filename)

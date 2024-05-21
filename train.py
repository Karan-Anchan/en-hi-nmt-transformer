import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

from datasets import load_dataset, concatenate_datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import BertPreTokenizer

def get_all_sentences(ds, lang):
    
    """
    Generator function that yields sentences in a specified language from the dataset.

    Args:
        ds (Dataset): The dataset containing multilingual translations.
        lang (str): The language code of the sentences to yield.

    Yields:
        str: Sentences in the specified language.
    """
    
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    
    """
    Retrieves an existing tokenizer or trains a new tokenizer on the dataset if it doesn't exist.

    Args:
        config (dict): Configuration dictionary containing the tokenizer file path and language codes.
        ds (Dataset): The dataset containing multilingual translations.
        lang (str): The language code for which to build the tokenizer.

    Returns:
        Tokenizer: The trained tokenizer for the specified language.
    """
    
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = BertPreTokenizer()
        trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], 
                             min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
    return tokenizer

def get_ds(config):
    
    """
    Loads the dataset, trains or loads tokenizers for the source and target languages, 
    and splits the dataset into training and validation sets.

    Args:
        config (dict): Configuration dictionary containing language codes and tokenizer file paths.

    Returns:
        tuple: A tuple containing the training and validation datasets, and the tokenizers for source and target languages.
    """
    
    ds_splitted = load_dataset('cfilt/iitb-english-hindi')
    ds_raw = concatenate_datasets([ds_splitted['train'], ds_splitted['test'], ds_splitted['validation']])
    
    # Build Tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    # Keep 90% for Training and 10% for Validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
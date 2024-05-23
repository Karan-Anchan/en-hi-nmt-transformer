import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_config, get_weights_file_path

from datasets import load_dataset, concatenate_datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import BertPreTokenizer

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import warnings

def get_all_sentences(ds, 
                      lang):
    
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

def get_or_build_tokenizer(config, 
                           ds, 
                           lang):
    
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
    
    train_ds = BilingualDataset(train_ds_raw, 
                                tokenizer_src, 
                                tokenizer_tgt, 
                                config['lang_src'], 
                                config['lang_tgt'],
                                config['seq_len'])
    
    val_ds = BilingualDataset(val_ds_raw, 
                                tokenizer_src, 
                                tokenizer_tgt, 
                                config['lang_src'], 
                                config['lang_tgt'],
                                config['seq_len'])
    
    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
        
        src_ids = tokenizer_src.encode(item['translation']).ids
        tgt_ids = tokenizer_tgt.encode(item['translation']).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
        
    print(f'Max Length Of Source Sentence: {max_len_src}')
    print(f'Max Length Of Target Sentence: {max_len_tgt}')
    
    train_dataloader = DataLoader(train_ds, 
                                  batch_size=config['batch_size'], 
                                  shuffle=True)
    
    val_dataloader = DataLoader(val_ds, 
                                batch_size=1, 
                                shuffle=True)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config,
              vocab_src_len,
              vocab_tgt_len):
    
    """
    Builds and returns a Transformer model based on the given configuration and vocabulary sizes.

    This function uses the provided configuration settings and vocabulary sizes for the source
    and target languages to build a Transformer model. The `build_transformer` function is called
    with the appropriate parameters to create and initialize the model.

    Args:
        config (dict): A dictionary containing the configuration settings including sequence length
                       and model dimensions.
        vocab_src_len (int): The size of the source vocabulary.
        vocab_tgt_len (int): The size of the target vocabulary.

    Returns:
        nn.Module: An instance of the Transformer model initialized with the given parameters.
    """
    
    model = build_transformer(vocab_src_len,
                              vocab_tgt_len, 
                              config['seq_len'],
                              config['seq_len'],
                              config['d_model'])
    
    return model

def train_model(config):
    
    """
    Trains a Transformer model based on the given configuration.

    This function sets up the training environment, including device configuration,
    data loading, model initialization, and training loop. It also supports resuming
    training from a checkpoint, logs training progress using TensorBoard, and saves
    model checkpoints after each epoch.

    Args:
        config (dict): A dictionary containing configuration settings such as
                       batch size, number of epochs, learning rate, sequence length,
                       model dimensions, language settings, and file paths.

    Returns:
        None
    """
    
    # Define The Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using Device {device}')
    
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # TensorBoard
    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading Model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing Epoch {epoch:02d}')
        for batch in batch_iterator:
            
            encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)
            
            # Run The Tensors Through The Transformer
            encoder_output = model.encode(encoder_input, 
                                          encoder_mask) # (B, seq_len, d_model)
            
            decoder_output = model.decode(encoder_output, 
                                          encoder_mask, 
                                          decoder_input, 
                                          decoder_mask) # (B, seq_len, d_model)
            
            proj_output = model.project(decoder_output) # (B, seq_len, tgt_vocab_size)
            
            label = batch['label'].to(device) # (B, seq_len)
            
            # (B, seq_len, tgt_vocab_size) --> (B * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss" : f"{loss.item(): 6.3f}"})
            
            # Log The Loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            
            # Backpropagate The Loss
            loss.backward()
            
            # Update The Weights
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1
            
        # Save The Model At The End Of Every Epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'global_step' : global_step,
        }, model_filename)
        
if __name__ == '__main__':
    # warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
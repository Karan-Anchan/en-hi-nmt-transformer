import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Any

class BilingualDataset(Dataset):
    
    """
    A PyTorch Dataset class for handling bilingual datasets.

    This dataset handles bilingual text pairs, applies tokenization,
    and generates inputs and masks for encoder-decoder models.

    Args:
        ds (Dataset): The dataset containing translation pairs.
        tokenizer_src (Tokenizer): The tokenizer for the source language.
        tokenizer_tgt (Tokenizer): The tokenizer for the target language.
        src_lang (str): The source language code.
        tgt_lang (str): The target language code.
        seq_len (int): The fixed sequence length for inputs and outputs.

    Attributes:
        ds (Dataset): The dataset containing translation pairs.
        tokenizer_src (Tokenizer): The tokenizer for the source language.
        tokenizer_tgt (Tokenizer): The tokenizer for the target language.
        src_lang (str): The source language code.
        tgt_lang (str): The target language code.
        sos_token (Tensor): The tensor representing the start-of-sequence token.
        pad_token (Tensor): The tensor representing the padding token.
        eos_token (Tensor): The tensor representing the end-of-sequence token.
    """
    
    def __init__(self, 
                 ds, 
                 tokenizer_src, 
                 tokenizer_tgt, 
                 src_lang, 
                 tgt_lang, 
                 seq_len) -> None:
        
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype = torch.int64)
        
    def __len__(self):
        
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        
        return len(self.ds)
    
    def __getitem__(self, index: Any) -> Any:
        
        """
        Retrieves a single data point from the dataset at the specified index.

        Args:
            index (int): The index of the data point to retrieve.

        Returns:
            dict: A dictionary containing encoder inputs, decoder inputs, masks, labels, and original texts.
        """
        
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
        
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence Is Too Long')
        
        # Add SOS and EOS as well as PAD tokens to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )
        
        # Add SOS and PAD to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )
        
        # Add EOS and PAD to the label
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        return {
            "encoder_input" : encoder_input, # (seq_len)
            "decoder_input" : decoder_input, # (seq_len)
            "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, 1, seq_len) & (1, seq_len, seq_len)
            "label" : label, # (seq_len)
            "src_text" : src_text,
            "tgt_text" : tgt_text
        }
        
def causal_mask(size):
    
    """
    Creates a causal mask for decoder inputs to prevent attention to future tokens.

    Args:
        size (int): The size of the mask (sequence length).

    Returns:
        Tensor: A causal mask tensor of shape (1, size, size).
    """
    
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
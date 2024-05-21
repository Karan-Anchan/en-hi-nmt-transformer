import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    
    """
    InputEmbeddings is a PyTorch module for creating input embeddings for a Transformer model.

    This module combines a standard embedding layer with a scaling factor to produce 
    embeddings of a specified dimension (`d_model`). The scaling factor is the square 
    root of `d_model`, which helps in stabilizing the gradients during training.

    Attributes:
        d_model (int): The dimension of the embeddings.
        vocab_size (int): The size of the vocabulary.
        embedding (nn.Embedding): The embedding layer which converts input tokens to 
                                  dense vectors of size `d_model`.

    Methods:
        forward(x):
            Applies the embedding layer to the input tensor `x` and scales the embeddings.

    Parameters:
        d_model (int): The dimension of the embeddings.
        vocab_size (int): The size of the vocabulary.
    """

    def __init__(self, 
                 d_model: int, 
                 vocab_size: int) -> None:
        
        """
        Initializes the InputEmbeddings module.

        Args:
            d_model (int): The dimension of the embeddings.
            vocab_size (int): The size of the vocabulary.
        """
        
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, 
                x):
        
        """
        Applies the embedding layer to the input tensor `x` and scales the embeddings.

        Args:
            x (Tensor): Input tensor of token indices with shape (batch_size, sequence_length).

        Returns:
            Tensor: Scaled embeddings with shape (batch_size, sequence_length, d_model).
        """
        
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    
    """
    PositionalEncoding is a PyTorch module that adds positional information to the input embeddings.

    This module creates a matrix of positional encodings which are added to the input embeddings 
    to provide the model with information about the position of each token in the sequence. 
    It also applies dropout to the resulting tensor.

    Attributes:
        d_model (int): The dimension of the embeddings.
        seq_len (int): The maximum length of the input sequences.
        dropout (nn.Dropout): Dropout layer for regularization.
        pe (Tensor): Positional encoding matrix of shape (1, seq_len, d_model).

    Methods:
        forward(x):
            Adds positional encodings to the input tensor `x` and applies dropout.

    Parameters:
        d_model (int): The dimension of the embeddings.
        seq_len (int): The maximum length of the input sequences.
        dropout (float): The dropout rate to apply to the output.
    """
    
    def __init__(self, 
                 d_model: int, 
                 seq_len: int, 
                 dropout: float) -> None:
        
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimension of the embeddings.
            seq_len (int): The maximum length of the input sequences.
            dropout (float): The dropout rate to apply to the output.
        """
        
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Matrix Of Shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Vector Of Shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Sine Function applied on Even Positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Cosine Function applied on Odd Positions
        pe[:, 1::2] = torch.cos(position * div_term)

        # Expand Dimension to include Batches (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        # Register the Tensor in Buffer to Save Along the Model
        self.register_buffer('pe', pe)

    def forward(self, 
                x):
        
        """
        Adds positional encodings to the input tensor `x` and applies dropout.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: The input tensor with added positional encodings and applied dropout,
                    of shape (batch_size, seq_len, d_model).
        """
        
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    
    """
    LayerNormalization is a PyTorch module that applies layer normalization to its inputs.

    This module normalizes the inputs across the features for each data point, ensuring that 
    the mean and variance of the inputs remain constant, which helps in stabilizing and speeding 
    up the training process. It also includes learnable scaling (`alpha`) and shifting (`bias`) 
    parameters to allow the model to undo normalization if necessary.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        alpha (nn.Parameter): Learnable scaling parameter, initialized to ones.
        bias (nn.Parameter): Learnable shifting parameter, initialized to zeros.

    Methods:
        forward(x):
            Applies layer normalization to the input tensor `x`.

    Parameters:
        eps (float): A small value added to the denominator for numerical stability. Default is 1e-6.
    """

    def __init__(self, 
                 eps: float = 10**-6) -> None:
        
        """
        Initializes the LayerNormalization module.

        Args:
            eps (float): A small value added to the denominator for numerical stability. Default is 1e-6.
        """
        
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self, 
                x):
        
        """
        Applies layer normalization to the input tensor `x`.

        Args:
            x (Tensor): Input tensor of shape (batch_size, ..., feature_size).

        Returns:
            Tensor: The normalized tensor with the same shape as `x`.
        """
        
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        
        return self.alpha * (x - mean)/ (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    
    """
    FeedForwardBlock is a PyTorch module that implements a feedforward neural network block 
    used in Transformer models.

    This module consists of two linear transformations with a ReLU activation in between, 
    followed by dropout for regularization. It is typically used after the self-attention 
    mechanism in each layer of the Transformer to further process the data.

    Attributes:
        linear_1 (nn.Linear): The first linear transformation layer (d_model to d_ff).
        dropout (nn.Dropout): Dropout layer for regularization.
        linear_2 (nn.Linear): The second linear transformation layer (d_ff to d_model).

    Methods:
        forward(x):
            Applies the feedforward block to the input tensor `x`.

    Parameters:
        d_model (int): The dimension of the input and output features.
        d_ff (int): The dimension of the inner layer.
        dropout (float): The dropout rate to apply after the first linear transformation.
    """

    def __init__(self, 
                 d_model: int, 
                 d_ff: int, 
                 dropout: float):
        
        """
        Initializes the FeedForwardBlock module.

        Args:
            d_model (int): The dimension of the input and output features.
            d_ff (int): The dimension of the inner layer.
            dropout (float): The dropout rate to apply after the first linear transformation.
        """
        
        super().__init__()

        # Weights are already true in the Linear Layer
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 + b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 + b2

    def forward(self, 
                x):
        
        """
        Applies the feedforward block to the input tensor `x`.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model) after applying 
                    two linear transformations with ReLU activation and dropout in between.
        """
        
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttention(nn.Module):
    
    """
    MultiHeadAttention is a PyTorch module that implements the multi-head attention mechanism 
    used in Transformer models.

    This module applies multiple attention heads to the input sequences, allowing the model to 
    focus on different parts of the input for each head. It consists of linear transformations 
    for queries, keys, and values, as well as a final linear transformation for the output.

    Attributes:
        d_model (int): The dimension of the input and output features.
        h (int): The number of attention heads.
        d_k (int): The dimension of the queries and keys for each head (d_model // h).
        w_q (nn.Linear): Linear layer for transforming inputs to queries.
        w_k (nn.Linear): Linear layer for transforming inputs to keys.
        w_v (nn.Linear): Linear layer for transforming inputs to values.
        w_o (nn.Linear): Linear layer for transforming concatenated outputs of all heads.
        dropout (nn.Dropout): Dropout layer for regularization.

    Methods:
        attention(query, key, value, mask, dropout):
            Computes scaled dot-product attention.

        forward(q, k, v, mask):
            Applies the multi-head attention mechanism to the input tensors.

    Parameters:
        d_model (int): The dimension of the input and output features.
        h (int): The number of attention heads.
        dropout (float): The dropout rate to apply to the attention weights.
    """

    def __init__(self, 
                 d_model: int, 
                 h: int, 
                 dropout: float) -> None:
        
        """
        Initializes the MultiHeadAttention module.

        Args:
            d_model (int): The dimension of the input and output features.
            h (int): The number of attention heads.
            dropout (float): The dropout rate to apply to the attention weights.
        """
        
        super().__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model not divisible by h"
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv
        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, 
                  key, 
                  value, 
                  mask, 
                  dropout: nn.Dropout):
        
        """
        Computes scaled dot-product attention.

        Args:
            query (Tensor): Query tensor of shape (batch_size, h, seq_len, d_k).
            key (Tensor): Key tensor of shape (batch_size, h, seq_len, d_k).
            value (Tensor): Value tensor of shape (batch_size, h, seq_len, d_k).
            mask (Tensor): Mask tensor to prevent attention to certain positions.
            dropout (nn.Dropout): Dropout layer to apply to the attention scores.

        Returns:
            Tensor: Output tensor after applying attention mechanism.
            Tensor: Attention scores.
        """
        
        d_k = query.shape[-1]

        # (Batch, h, seq_len, d_k) --> (Batch, h , seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1))/math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, h seq_len, seq_len)
        if dropout is not None:
            attention_scores =  dropout(attention_scores)

        return (attention_scores @ value), attention_scores


    def forward(self,
                q, 
                k, 
                v, 
                mask):
        
        """
        Applies the multi-head attention mechanism to the input tensors.

        Args:
            q (Tensor): Query tensor of shape (batch_size, seq_len, d_model).
            k (Tensor): Key tensor of shape (batch_size, seq_len, d_model).
            v (Tensor): Value tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor): Mask tensor to prevent attention to certain positions.

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    
    """
    ResidualConnection is a PyTorch module that implements a residual connection with 
    layer normalization and dropout.

    This module is used in Transformer models to add the input of a sublayer to its output, 
    which helps in training deep networks by mitigating the vanishing gradient problem. 
    It includes layer normalization before the sublayer and dropout after the sublayer.

    Attributes:
        dropout (nn.Dropout): Dropout layer for regularization.
        norm (LayerNormalization): Layer normalization applied before the sublayer.

    Methods:
        forward(x, sublayer):
            Applies the residual connection to the input tensor `x` and the output of the `sublayer`.

    Parameters:
        dropout (float): The dropout rate to apply after the sublayer.
    """

    def __init__(self, 
                 dropout: float) -> None:
        
        """
        Initializes the ResidualConnection module.

        Args:
            dropout (float): The dropout rate to apply after the sublayer.
        """
        
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, 
                x, 
                sublayer):
        
        """
        Applies the residual connection to the input tensor `x` and the output of the `sublayer`.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            sublayer (Callable): A sublayer function or module that takes `x` as input and returns a tensor of the same shape.

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model) after applying layer normalization, 
                    the sublayer, dropout, and adding the input `x`.
        """
        
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    
    """
    EncoderBlock is a PyTorch module that implements a single block of the encoder in a Transformer model.

    This module consists of a multi-head self-attention mechanism followed by a position-wise 
    feed-forward neural network, each with residual connections and layer normalization.

    Attributes:
        self_attention_block (MultiHeadAttention): Multi-head self-attention mechanism.
        feed_forward_block (FeedForwardBlock): Position-wise feed-forward neural network.
        residual_connections (nn.ModuleList): List of residual connections for the self-attention and feed-forward blocks.

    Methods:
        forward(x, src_mask):
            Applies the encoder block to the input tensor `x` with the provided source mask `src_mask`.

    Parameters:
        self_attention_block (MultiHeadAttention): Multi-head self-attention mechanism.
        feed_forward_block (FeedForwardBlock): Position-wise feed-forward neural network.
        dropout (float): The dropout rate to apply in the residual connections.
    """

    def __init__(self, 
                 self_attention_block: MultiHeadAttention, 
                 feed_forward_block: FeedForwardBlock, 
                 dropout: float) -> None:
        
        """
        Initializes the EncoderBlock module.

        Args:
            self_attention_block (MultiHeadAttention): Multi-head self-attention mechanism.
            feed_forward_block (FeedForwardBlock): Position-wise feed-forward neural network.
            dropout (float): The dropout rate to apply in the residual connections.
        """
        
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, 
                x, 
                src_mask):
        
        """
        Applies the encoder block to the input tensor `x` with the provided source mask `src_mask`.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            src_mask (Tensor): Source mask tensor to prevent attention to certain positions.

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model) after applying the self-attention 
                    and feed-forward blocks with residual connections.
        """
        
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        
        return x
    
class Encoder(nn.Module):
    
    """
    Encoder is a PyTorch module that implements the encoder component of a Transformer model.

    The Encoder consists of a stack of encoder blocks, each containing a multi-head self-attention mechanism 
    followed by a position-wise feed-forward neural network, with residual connections and layer normalization.

    Attributes:
        layers (nn.ModuleList): List of encoder blocks.
        norm (LayerNormalization): Layer normalization applied to the output of the final encoder block.

    Methods:
        forward(x, mask):
            Applies the stack of encoder blocks to the input tensor `x` with the provided mask `mask`.

    Parameters:
        layers (nn.ModuleList): List of encoder blocks.
    """

    def __init__(self, 
                 layers: nn.ModuleList) -> None:
        
        """
        Initializes the Encoder module.

        Args:
            layers (nn.ModuleList): List of encoder blocks.
        """
        
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, 
                x, 
                mask):
        
        """
        Applies the stack of encoder blocks to the input tensor `x` with the provided mask `mask`.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor): Mask tensor to prevent attention to certain positions.

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model) after applying the encoder blocks 
                    and layer normalization.
        """
        
        for layer in self.layers:
            x = layer(x, mask)
            
        return self.norm(x)
        
class DecoderBlock(nn.Module):
    
    """
    DecoderBlock is a PyTorch module that implements a single block of the decoder in a Transformer model.

    This module consists of a self-attention mechanism, a cross-attention mechanism that attends to 
    the encoder's output, and a position-wise feed-forward neural network, each with residual connections 
    and layer normalization.

    Attributes:
        self_attention_block (MultiHeadAttention): Multi-head self-attention mechanism.
        cross_attention_block (MultiHeadAttention): Multi-head cross-attention mechanism for attending to encoder outputs.
        feed_forward_block (FeedForwardBlock): Position-wise feed-forward neural network.
        residual_connections (nn.ModuleList): List of residual connections for the self-attention, cross-attention, 
                                              and feed-forward blocks.

    Methods:
        forward(x, encoder_output, src_mask, tgt_mask):
            Applies the decoder block to the input tensor `x` with the provided encoder output and masks.

    Parameters:
        self_attention_block (MultiHeadAttention): Multi-head self-attention mechanism.
        cross_attention_block (MultiHeadAttention): Multi-head cross-attention mechanism for attending to encoder outputs.
        feed_forward_block (FeedForwardBlock): Position-wise feed-forward neural network.
        dropout (float): The dropout rate to apply in the residual connections.
    """

    def __init__(self, 
                 self_attention_block: MultiHeadAttention, 
                 cross_attention_block: MultiHeadAttention, 
                 feed_forward_block: FeedForwardBlock, 
                 dropout: float) -> None:
        
        """
        Initializes the DecoderBlock module.

        Args:
            self_attention_block (MultiHeadAttention): Multi-head self-attention mechanism.
            cross_attention_block (MultiHeadAttention): Multi-head cross-attention mechanism for attending to encoder outputs.
            feed_forward_block (FeedForwardBlock): Position-wise feed-forward neural network.
            dropout (float): The dropout rate to apply in the residual connections.
        """
        
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        
    def forward(self, 
                x, 
                encoder_output, 
                src_mask, 
                tgt_mask):
        
        """
        Applies the decoder block to the input tensor `x` with the provided encoder output and masks.

        Args:
            x (Tensor): Input tensor of shape (batch_size, tgt_seq_len, d_model).
            encoder_output (Tensor): Output tensor from the encoder of shape (batch_size, src_seq_len, d_model).
            src_mask (Tensor): Source mask tensor to prevent attention to certain positions in the encoder output.
            tgt_mask (Tensor): Target mask tensor to prevent attention to certain positions in the target input.

        Returns:
            Tensor: Output tensor of shape (batch_size, tgt_seq_len, d_model) after applying the self-attention, 
                    cross-attention, and feed-forward blocks with residual connections.
        """
        
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        
        return x

class Decoder(nn.Module):
    
    """
    Decoder is a PyTorch module that implements the decoder component of a Transformer model.

    The Decoder consists of a stack of decoder blocks, each containing a self-attention mechanism, 
    a cross-attention mechanism that attends to the encoder's output, and a position-wise feed-forward 
    neural network, with residual connections and layer normalization.

    Attributes:
        layers (nn.ModuleList): List of decoder blocks.
        norm (LayerNormalization): Layer normalization applied to the output of the final decoder block.

    Methods:
        forward(x, encoder_output, src_mask, tgt_mask):
            Applies the stack of decoder blocks to the input tensor `x` with the provided encoder output and masks.

    Parameters:
        layers (nn.ModuleList): List of decoder blocks.
    """

    def __init__(self, 
                 layers: nn.ModuleList) -> None:
        
        """
        Initializes the Decoder module.

        Args:
            layers (nn.ModuleList): List of decoder blocks.
        """
        
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, 
                x, 
                encoder_output, 
                src_mask, 
                tgt_mask):
        
        """
        Applies the stack of decoder blocks to the input tensor `x` with the provided encoder output and masks.

        Args:
            x (Tensor): Input tensor of shape (batch_size, tgt_seq_len, d_model).
            encoder_output (Tensor): Output tensor from the encoder of shape (batch_size, src_seq_len, d_model).
            src_mask (Tensor): Source mask tensor to prevent attention to certain positions in the encoder output.
            tgt_mask (Tensor): Target mask tensor to prevent attention to certain positions in the target input.

        Returns:
            Tensor: Output tensor of shape (batch_size, tgt_seq_len, d_model) after applying the decoder blocks 
                    and layer normalization.
        """
        
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
            
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    
    """
    ProjectionLayer is a PyTorch module that implements the projection layer of a Transformer model.

    The ProjectionLayer linearly transforms the input tensor from the dimension of `d_model` to the 
    dimension of `vocab_size`, followed by a log softmax operation along the last dimension.

    Attributes:
        proj (nn.Linear): Linear transformation layer.

    Methods:
        forward(x):
            Applies the projection layer to the input tensor `x`.

    Parameters:
        d_model (int): Dimensionality of the input tensor.
        vocab_size (int): Size of the vocabulary, i.e., the number of output classes.
    """

    def __init__(self, 
                 d_model: int,
                 vocab_size: int) -> None:
        
        """
        Initializes the ProjectionLayer module.

        Args:
            d_model (int): Dimensionality of the input tensor.
            vocab_size (int): Size of the vocabulary, i.e., the number of output classes.
        """
        
        super().__init__()
        self.proj = nn.Linear(d_model, 
                              vocab_size)

    def forward(self, 
                x):
        
        """
        Applies the projection layer to the input tensor `x`.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, vocab_size) after applying the linear transformation 
                    followed by log softmax operation along the last dimension.
        """
        
        # (Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), 
                                 dim = -1)
    
class Transformer(nn.Module):
    
    """
    Transformer is a PyTorch module that implements the Transformer model.

    The Transformer model consists of an encoder, a decoder, input embeddings, positional encodings,
    and a projection layer. It is designed for sequence-to-sequence tasks such as machine translation.

    Attributes:
        encoder (Encoder): Encoder component of the Transformer.
        decoder (Decoder): Decoder component of the Transformer.
        src_embed (InputEmbeddings): Input embeddings for the source sequence.
        tgt_embed (InputEmbeddings): Input embeddings for the target sequence.
        src_pos (PositionalEncoding): Positional encodings for the source sequence.
        tgt_pos (PositionalEncoding): Positional encodings for the target sequence.
        projection_layer (ProjectionLayer): Projection layer to generate output probabilities.

    Methods:
        encode(src, src_mask):
            Encodes the source sequence using the encoder.
        decode(src_mask, encoder_output, tgt, tgt_mask):
            Decodes the target sequence using the decoder and encoder output.
        project(x):
            Projects the output tensor `x` to obtain the final output probabilities.

    Parameters:
        encoder (Encoder): Encoder component of the Transformer.
        decoder (Decoder): Decoder component of the Transformer.
        src_embed (InputEmbeddings): Input embeddings for the source sequence.
        tgt_embed (InputEmbeddings): Input embeddings for the target sequence.
        src_pos (PositionalEncoding): Positional encodings for the source sequence.
        tgt_pos (PositionalEncoding): Positional encodings for the target sequence.
        projection_layer (ProjectionLayer): Projection layer to generate output probabilities.
    """
    
    def __init__(self, 
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: InputEmbeddings,
                 tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer) -> None:
        
        """
        Initializes the Transformer module.

        Args:
            encoder (Encoder): Encoder component of the Transformer.
            decoder (Decoder): Decoder component of the Transformer.
            src_embed (InputEmbeddings): Input embeddings for the source sequence.
            tgt_embed (InputEmbeddings): Input embeddings for the target sequence.
            src_pos (PositionalEncoding): Positional encodings for the source sequence.
            tgt_pos (PositionalEncoding): Positional encodings for the target sequence.
            projection_layer (ProjectionLayer): Projection layer to generate output probabilities.
        """
        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
    def encode(self, 
               src, 
               src_mask):
        
        """
        Encodes the source sequence using the encoder.

        Args:
            src (Tensor): Source sequence tensor.
            src_mask (Tensor): Mask tensor for the source sequence.

        Returns:
            Tensor: Encoded representation of the source sequence.
        """
        
        src = self.src_embed(src)
        src = self.src_pos(src)
        
        return self.encoder(src, 
                            src_mask)
    
    def decode(self, 
               src_mask,
               encoder_output, 
               tgt, 
               tgt_mask):
        
        """
        Decodes the target sequence using the decoder and encoder output.

        Args:
            src_mask (Tensor): Mask tensor for the source sequence.
            encoder_output (Tensor): Output tensor from the encoder.
            tgt (Tensor): Target sequence tensor.
            tgt_mask (Tensor): Mask tensor for the target sequence.

        Returns:
            Tensor: Decoded representation of the target sequence.
        """
        
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        
        return self.decoder(tgt, 
                            encoder_output, 
                            src_mask, 
                            tgt_mask)
        
    def project(self, 
                x):
        
        """
        Projects the output tensor `x` to obtain the final output probabilities.

        Args:
            x (Tensor): Input tensor to be projected.

        Returns:
            Tensor: Output tensor after projection.
        """
        
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, 
                      tgt_vocab_size: int, 
                      src_seq_len: int, 
                      tgt_seq_len: int, 
                      d_model: int = 512, 
                      N: int = 6, 
                      h: int = 8, 
                      dropout: float = 0.1, 
                      d_ff: int = 2048) -> Transformer:
    
    """
    Builds and initializes a Transformer model for sequence-to-sequence tasks.

    The function constructs and initializes the components of a Transformer model 
    based on the provided parameters.

    Args:
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        src_seq_len (int): Length of the source sequence.
        tgt_seq_len (int): Length of the target sequence.
        d_model (int, optional): Dimensionality of the model. Defaults to 512.
        N (int, optional): Number of encoder and decoder blocks. Defaults to 6.
        h (int, optional): Number of attention heads. Defaults to 8.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        d_ff (int, optional): Dimensionality of the feed-forward network. Defaults to 2048.

    Returns:
        Transformer: Initialized Transformer model.
    """
    
    # Create The Embedding Layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    # Create The Positional Encoding Layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create The Encoder Blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        encoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, encoder_feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
        
    # Create The Decoder Blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
        
    # Create The Encoder And Decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    # Create The Projection Layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create The Transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize The Parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .transformer import BOS_IDX, EOS_IDX, PAD_IDX, UNK_IDX


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(torch.tensor(10000)) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        device = token_embeddings.device
        batch_size, n_tokens, _ = token_embeddings.size()
        position_ids = torch.arange(
            n_tokens, dtype=torch.long, device=device).repeat(batch_size, 1)
        # TODO Getting rid of the squeeze would require changing the pretrained weights, 
        # and thus break back-compat...
        position_embeddings = F.embedding(position_ids, weight=self.pos_embedding.squeeze(1))
        positioned_embeddings = token_embeddings + position_embeddings
        positioned_embeddings = self.dropout(positioned_embeddings)
        return positioned_embeddings


class JitTransformerG2P(nn.Module):
    """
    TorchScript compliant variant of the original model
    """

    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = torch.jit.trace(TokenEmbedding(src_vocab_size, emb_size), torch.randint(1, 400, size=(1, 5)))
        self.tgt_tok_emb = torch.jit.trace(TokenEmbedding(tgt_vocab_size, emb_size), torch.randint(1, 400, size=(1, 5)))
        self.positional_encoding = torch.jit.trace(PositionalEncoding(emb_size, dropout=dropout), torch.randn(1, 5, emb_size))

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        self.BOS_IDX = BOS_IDX
        self.EOS_IDX = EOS_IDX
        self.PAD_IDX = PAD_IDX
        self.UNK_IDX = UNK_IDX
    
    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_padding_mask: torch.Tensor,
        tgt_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)
    
    @torch.jit.export
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    @torch.jit.export
    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor)-> torch.Tensor:
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )
    
    @torch.jit.export
    def transcribe(self, grapheme_ids: torch.Tensor) -> torch.Tensor:
        device = grapheme_ids.device

        num_tokens = grapheme_ids.shape[1]
        batch_size = grapheme_ids.shape[0]

        mask = torch.zeros(
            size=(num_tokens, num_tokens),
            dtype=torch.bool, device=device
        )
        max_len = num_tokens + 5

        encoder_hidden_states = self.encode(grapheme_ids, mask)

        ys = grapheme_ids.new_ones(batch_size, 1).fill_(self.BOS_IDX)
        is_done = torch.zeros(size=(batch_size,), device=device)
        for _ in range(max_len - 1):
            tgt_mask = self.transformer.generate_square_subsequent_mask(ys.size(1), device=device).to(torch.bool)
            out = self.decode(ys, encoder_hidden_states, tgt_mask)

            probs = self.generator(out[:, -1, :])
            # (B,)
            next_words = probs.argmax(dim=-1)
            
            
            is_done[(next_words == self.EOS_IDX)] = 1
            next_words[is_done == 1] = self.EOS_IDX

            ys = torch.cat([ys, next_words.unsqueeze(1)], dim=1)

            if is_done.all():
                break
        return ys


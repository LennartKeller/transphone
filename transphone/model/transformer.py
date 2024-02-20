import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Transformer

from transphone.config import TransphoneConfig
from transphone.model.utils import pad_sos_eos

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 0, 1, 1


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros(
        (src_seq_len, src_seq_len), device=TransphoneConfig.device
    ).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embeddings: Tensor):
        device = token_embeddings.device
        batch_size, n_tokens, _ = token_embeddings.size()
        position_ids = torch.arange(n_tokens, dtype=torch.long, device=device).repeat(
            batch_size, 1
        )
        # TODO Getting rid of the squeeze would require changing the pretrained weights,
        # and thus break back-compat...
        position_embeddings = F.embedding(
            position_ids, weight=self.pos_embedding.squeeze(1)
        )
        positioned_embeddings = token_embeddings + position_embeddings
        positioned_embeddings = self.dropout(positioned_embeddings)
        return positioned_embeddings


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network
class TransformerG2P(nn.Module):
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
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
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

    def encode(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)),
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor,
        memory_key_padding_mask: Optional[Tensor] = None,
    ):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)),
            memory,
            tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Single training step
        Args:
            x (torch.Tensor): Input sequence
            y (torch.Tensor): Target sequence

        Returns:
            torch.Tensor: Cross-Entropy Loss
        """

        self.train()

        ys_in, ys_out = pad_sos_eos(y, 1, 1)

        # T,B
        # tgt_input = ys_in.transpose(1,0)
        # tgt_out = ys_out.transpose(1,0)

        # B, T
        tgt_input = ys_in
        tgt_out = ys_out

        src = x.transpose(1, 0)

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )

        logits = self.forward(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        return loss

    @torch.no_grad
    def inference(self, src):
        self.eval()
        device = next(self.parameters()).device

        src = src.view(1, -1)
        num_tokens = src.shape[1]
        src_mask = torch.zeros(
            size=(num_tokens, num_tokens), dtype=torch.bool, device=device
        )

        encoder_hidden_states = self.encode(src, src_mask)

        max_len = num_tokens + 5
        ys = torch.full(
            size=(1, 1), fill_value=BOS_IDX, device=device, dtype=torch.long
        )
        for _ in range(max_len - 1):
            tgt_mask = self.transformer.generate_square_subsequent_mask(
                ys.size(1), device=device
            ).bool()
            out = self.decode(ys, encoder_hidden_states, tgt_mask)
            probs = self.generator(out[:, -1, :])
            next_token_id = probs.argmax(dim=-1)

            ys = torch.cat([ys, next_token_id.reshape(1, 1)], dim=1)
            if next_token_id == EOS_IDX:
                break
        out = ys.squeeze(0)[1:]
        if out[-1] == 1:
            out = out[:-1]

        return out

    @torch.no_grad
    def inference_batch(self, src, src_mask=None, src_key_padding_mask=None):
        self.eval()
        device = next(self.parameters()).device

        batch_size, num_tokens = src.size()
        if src_mask is None:
            src_mask = torch.zeros(
                size=(num_tokens, num_tokens), dtype=torch.bool, device=device
            )

        max_len = num_tokens + 5
        encoder_hidden_states = self.encode(src, src_mask, src_key_padding_mask)

        ys = src.new_ones(batch_size, 1).fill_(BOS_IDX)
        is_done = torch.zeros(size=(batch_size,), device=device)
        for _ in range(max_len - 1):
            tgt_mask = self.transformer.generate_square_subsequent_mask(
                ys.size(1), device=device
            ).bool()
            out = self.decode(
                ys,
                encoder_hidden_states,
                tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )

            probs = self.generator(out[:, -1, :])
            # (B,)
            next_words = probs.argmax(dim=-1)

            is_done[(next_words == EOS_IDX)] = 1
            next_words[is_done == 1] = EOS_IDX

            ys = torch.cat([ys, next_words.unsqueeze(1)], dim=1)

            if is_done.all():
                break
        return ys

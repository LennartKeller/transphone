import math

import torch
import torch.nn as nn
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
        n_tokens = token_embeddings.size(1)
        positioned_embeddings = token_embeddings + self.pos_embedding[
            :n_tokens, :
        ].view_as(token_embeddings)
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

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
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

    def inference(self, x):
        self.eval()
        device = next(self.parameters()).device

        src = x.view(1, -1)
        print("src", src.size())
        num_tokens = src.shape[1]
        print("n_tok", num_tokens)
        src_mask = torch.zeros(
            size=(num_tokens, num_tokens), dtype=torch.bool, device=device
        )

        encoder_hidden_states = self.encode(src, src_mask)
        print("ehs", encoder_hidden_states.size())

        max_len = num_tokens + 5
        ys = torch.ones(1, 1, device=device, dtype=torch.long)
        for _ in range(max_len - 1):
            print("ys", ys.size())
            tgt_mask = generate_square_subsequent_mask(ys.size(1), device=device).bool()
            out = self.decode(ys, encoder_hidden_states, tgt_mask)
            print("out", out.size())
            prob = self.generator(out[:, -1, :])
            print("p", prob.size())
            next_token_id = prob.argmax(dim=-1)

            ys = torch.cat([ys, next_token_id.reshape(1, 1)], dim=1)
            print(ys.size())
            print("ysl", ys)
            print(tgt_mask)
            if next_token_id == EOS_IDX:
                break
        print(ys.size())
        print(ys)
        out = ys.squeeze(0).tolist()[1:]
        print(out)
        if out[-1] == 1:
            out = out[:-1]

        return out

    def inference_batch(self, x):
        self.eval()

        # (T,B)
        # src = x.transpose(0, 1)

        # (B, t)
        src = x

        # num_tokens = src.shape[0]
        # batch_size = src.shape[1]

        num_tokens = src.shape[1]
        batch_size = src.shape[0]

        src_mask = (
            (torch.zeros(num_tokens, num_tokens))
            .type(torch.bool)
            .to(TransphoneConfig.device)
        )
        max_len = num_tokens + 5

        memory = self.encode(src, src_mask)
        ys = x.new_ones(batch_size, 1).fill_(BOS_IDX).type(torch.long)

        is_done = [False] * batch_size

        for i in range(max_len - 1):
            memory = memory.to(TransphoneConfig.device)
            tgt_mask = (
                generate_square_subsequent_mask(ys.size(1)).type(torch.bool)
            ).to(TransphoneConfig.device)
            out = self.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)

            prob = self.generator(out[:, -1])

            _, next_words = torch.max(prob, dim=-1)

            for j in range(batch_size):
                if next_words[j].item() == EOS_IDX:
                    is_done[j] = True

                if is_done[j]:
                    next_words[j] = EOS_IDX

            ys = torch.cat([ys, next_words.unsqueeze(0)], dim=1)

            if all(is_done):
                break

        outs = []
        for y in ys.transpose(0, 1).tolist():
            outs.append([i for i in y if i >= 2])

        return outs

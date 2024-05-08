import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

class SelfAttention(nn.Cell):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        self.values = nn.Dense(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Dense(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Dense(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Dense(heads * self.head_dim, embed_size)

    def construct(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = ops.einsum("nqhd,nkhd->nhqk", queries, keys)
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)

        if mask is not None:
            energy = ops.masked_fill(energy, mask == 0, float("-1e20"))  # mask = 0 / 1

        attention = ops.softmax(energy / (self.embed_size ** (1/2)), axis=3)

        out = ops.einsum("nhql,nlhd->nqhd", attention, values).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # after einsum (N, query_len, heads, head_dim) then flatten last 2 dimensions

        out = self.fc_out(out)
        return out

class EncoderBlock(nn.Cell):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(EncoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Dense(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Dense(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def construct(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    
class MaskEncoderBlock(nn.Cell):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(EncoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Dense(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Dense(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)
    
    def construct(self, value, key, query, mask, trg_mask):
        attention = self.attention(value, key, query, mask * trg_mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Cell):
    def __init__(
            self,
            src_ais_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.trace_embedding = nn.Embedding(src_ais_size, embed_size)
        self.positional_embedding = nn.Embedding(max_length, embed_size)

        self.layer = nn.CellList(
            [
                EncoderBlock(
                    embed_size, heads, dropout, forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def construct(self, x, mask):
        N, seq_length = x.shape
        position = ops.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(position))
        
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

class MaskEncoder(nn.Cell):
    def __init__(
            self,
            src_ais_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
    ):
        super(MaskEncoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.trace_embedding = nn.Embedding(src_ais_size, embed_size)
        self.positional_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.CellList(
            [
                MaskEncoderBlock(
                    embed_size, heads, dropout, forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def construct(self, x, src_mask, trg_mask):
        N, seq_length = x.shape
        position = ops.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(position))
        
        for layer in self.layers:
            out = layer(out, out, out, src_mask, trg_mask)

        return out


class DecoderBlock(nn.Cell):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(DecoderBlock).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = EncoderBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def construct(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Cell):
    def __init__(
            self,
            trg_ais_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_ais_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.CellList(
            [
                DecoderBlock(
                    embed_size, 
                    heads, 
                    forward_expansion, 
                    dropout
                )
            for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Dense(embed_size, trg_ais_size)
        self.dropout = nn.Dropout(dropout)

    def construct(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        position = ops.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(position)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out


class Transformer(nn.Cell):
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            embed_size,
            num_layers,
            forward_expansion,
            heads,
            dropout,
            device,
            max_length
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = ops.tril(ops.ones(trg_len, trg_len)).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)
    
    def construct(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


class MaskEncoderTransformer(nn.Cell):
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            embed_size,
            num_layers,
            forward_expansion,
            heads,
            dropout,
            device,
            max_length
    ):
        super(MaskEncoderTransformer, self).__init__()

        self.encoder = MaskEncoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = ops.tril(ops.ones(trg_len, trg_len)).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)
    
    def construct(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask, trg_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
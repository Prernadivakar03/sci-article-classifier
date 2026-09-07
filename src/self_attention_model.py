"""
self_attention_model.py
-------------------------
Core model of this mini project:

    "Scientific Article Classification using Self-Attention Networks
     with Long-Context Representation Learning"

Architecture (Transformer-encoder style classifier):
    Input (padded token ids, length = MAX_LEN, e.g. 200)
        -> Token Embedding + Positional Embedding
        -> [Transformer Block] x N          (Multi-Head Self-Attention
                                              + Feed-Forward + residual +
                                              layer norm)  <-- this is what
                                              gives the model its
                                              LONG-CONTEXT representation:
                                              every token can directly
                                              attend to every other token
                                              in the abstract, regardless
                                              of distance.
        -> Global Average Pooling (sequence -> fixed vector)
        -> Dropout
        -> Dense (ReLU)
        -> Dropout
        -> Dense (Softmax over categories)

Requires: tensorflow >= 2.10  (pip install tensorflow)
"""

import tensorflow as tf
from tensorflow.keras import layers


class TokenAndPositionEmbedding(layers.Layer):
    """Combines a learned token embedding with a learned positional
    embedding so the (permutation-invariant) self-attention layers below
    know the order of tokens in the abstract."""

    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        length = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        positions = self.pos_emb(positions)
        tokens = self.token_emb(x)
        return tokens + positions

    def get_config(self):
        config = super().get_config()
        config.update({"maxlen": self.maxlen, "vocab_size": self.vocab_size,
                        "embed_dim": self.embed_dim})
        return config


class TransformerBlock(layers.Layer):
    """A single Transformer encoder block: Multi-Head Self-Attention
    followed by a position-wise Feed-Forward network, each wrapped with a
    residual connection and Layer Normalization (Vaswani et al., 2017)."""

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        # Self-attention: every token attends to every other token in the
        # sequence -> captures LONG-RANGE / long-context dependencies that
        # RNN/CNN based models struggle with over long scientific abstracts.
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads,
                        "ff_dim": self.ff_dim, "rate": self.rate})
        return config


def build_self_attention_classifier(
    vocab_size=10000,
    max_len=200,
    embed_dim=64,
    num_heads=4,
    ff_dim=128,
    num_transformer_blocks=2,
    num_classes=5,
    embedding_matrix=None,
):
    """Builds and compiles the Self-Attention (Transformer-encoder) classifier."""
    inputs = layers.Input(shape=(max_len,))
    embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
    x = embedding_layer(inputs)

    if embedding_matrix is not None:
        # Optionally initialise token embedding weights with pretrained
        # GloVe vectors (still fine-tuned during training).
        embedding_layer.token_emb.build((None,))
        embedding_layer.token_emb.set_weights([embedding_matrix])

    for _ in range(num_transformer_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)

    x = layers.GlobalAveragePooling1D()(x)     # pool the long-context representation
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="self_attention_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    m = build_self_attention_classifier(num_classes=5)
    m.summary()

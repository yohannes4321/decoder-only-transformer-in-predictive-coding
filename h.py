


def token_positional_embedding(x, vocab_size, embed_dim, block_size, dropout_rate=0.0, *, rngs):
    """
    x: [batch_size, seq_len] token indices
    returns: [batch_size, seq_len, embed_dim]
    """

    # Create modules once
    wte = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)
    wpe = nnx.Embed(num_embeddings=block_size, features=embed_dim, rngs=rngs)
    dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    batch_size, seq_len = x.shape

    tok_emb = wte(x)  # (B, T, C)

    pos = jnp.arange(seq_len)[None, :]  # (1, T)
    pos_emb = wpe(pos)                  # (1, T, C)

    return dropout(tok_emb + pos_emb), (wte, wpe, dropout)


batch_size = 4
seq_len = 10
vocab_size = 10000
embed_dim = 512
block_size = 128

tokens = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)

embedded, modules = token_positional_embedding(tokens, vocab_size, embed_dim, block_size, rngs=rngs)

print(embedded.shape)  # (4, 10, 512)
   def __init__(self, name,  block_size,batch_size, vocab_size, embed_dim,shape, eta=0., weight_init=None, bias_init=None,
                 w_bound=1., is_nonnegative=False, prior=("constant", 0.), w_decay=0., sign_value=1.,
                 optim_type="sgd", pre_wght=1., post_wght=1., p_conn=1.,
                 resist_scale=1., **kwargs):
    
@staticmethod
    def token_positional_embedding(x, vocab_size, embed_dim, block_size, dropout_rate=0.0, ):
        """
        x: [batch_size, seq_len] token indices
        returns: [batch_size, seq_len, embed_dim]
        """
        key = jax.random.PRNGKey(0)
        rngs = nnx.Rngs(key)
        # Create modules once
        wte = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)
        wpe = nnx.Embed(num_embeddings=block_size, features=embed_dim, rngs=rngs)
        dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

        batch_size, seq_len = x.shape

        tok_emb = wte(x)  # (B, T, C)

        pos = jnp.arange(seq_len)[None, :]  # (1, T)
        pos_emb = wpe(pos)   
        drive = dropout(tok_emb + pos_emb)               # (1, T, C)
        drive_2d = drive.reshape(batch_size * seq_len, embed_dim)
        return drive_2d
@transition(output_compartments=["outputs", "weights"])
    @staticmethod
    def advance_state(dt, Rscale, w_bounds, w_decay, inputs, weights,
                       pre, post, eta):
        outputs = HebbianSynapse.token_positional_embedding(inputs, vocab_size, embed_dim, block_size, rngs)
        ########################################################################
        ## Run one step of 2-factor Hebbian adaptation online
        dW = jnp.matmul(pre.T, post)
        #db = jnp.sum(_post, axis=0, keepdims=True)
        ## reformulated bounding flag to be linear algebraic
        flag = (w_bounds > 0.) * 1.
        dW = (dW * (w_bounds - jnp.abs(weights))) * flag + (dW) * (1. - flag)
        ## add small amount of synaptic decay
        weights = weights + (dW - weights * w_decay) * eta
        weights = jnp.clip(weights, 0., w_bounds)
        ########################################################################
        return outputs, weights
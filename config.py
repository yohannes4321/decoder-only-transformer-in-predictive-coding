class Config:
    vocab_size = 50257
    n_embed = 32
    block_size = 8
    num_heads = 2
    num_layers = 2
    dropout = 0.1
    batch_size = 8
    T = 3
    eta = 0.001
    exp_dir = 'exp'
    model_name = 'transformer'

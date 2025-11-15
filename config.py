class Config:
    vocab_size = 50257
    
    num_heads = 8
    num_layers = 2
    dropout = 0.1
    batch_size = 32
    block_size = 128
    n_embed = 512
    T = 3
    eta = 0.001
    exp_dir = 'exp'
    model_name = 'transformer'

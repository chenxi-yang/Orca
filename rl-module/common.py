
def create_one_dim_tr_model(
    cfg,
    s_dim,
    a_dim,
    model_dir,
    model_normalization,
):
    """
    Creates a 1-D transition reward model from a given configuration.
    """
    # TODO: use fixed configs for now
    in_size = s_dim + act_dim
    out_size = s_dim + 1 # state + reward

    model = # TODO
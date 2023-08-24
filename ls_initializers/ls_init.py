from ls_initializers.lowranksparse import LowRankSparseInitializer


def get_ls_init(model, args):
    initializer = LowRankSparseInitializer(
        model,
        sparse_matrix=args.sparse_matrix,
        threshold=args.threshold,
        sparsity=args.sparsity,
        degree=args.degree,
        activation=args.activation,
        rank=args.rank,
    )
    if args.model == "mlp" or args.model == "residualmlp":
        initialized_model = initializer.initialize_low_rank_mlp().to("cuda")
    elif args.model == "mixer":
        initialized_model = initializer.initialize_low_rank_mixer().to("cuda")
    else:
        raise NotImplementedError
    return initialized_model

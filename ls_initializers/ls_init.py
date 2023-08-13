from ls_initializers.lowranksparse import LowRankSparseInitializer

def get_ls_init(model, args):
    initializer = LowRankSparseInitializer(model,
                                            sparse_matrix=args.sparse_matrix,
                                            sparsity=args.sparsity,
                                            degree=args.degree)
    initialized_model = initializer.initialize_low_rank().to('cuda')
    return initialized_model
        
        
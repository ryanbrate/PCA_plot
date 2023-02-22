# PCA_plot

Plot PCA-dim reduced vectors onto a 2d plan.

Note 1: config["vector_labels"] can be a list of [label, weight] lists or otherwise, a path to a json containing such a list.

Note 2: makes no assumption about the source of vectors, rather relies on corresponding user-defined loading functions to provide an access interface wrt., the (obscured) underlying source format (e.g., gensim embeddings).

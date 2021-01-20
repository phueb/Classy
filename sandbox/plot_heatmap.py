"""
Research questions:
1. How well do nouns cluster together in hierarchical cluster tree generated from corpus statistics?
"""

from sklearn.decomposition import PCA

from preppy import PartitionedPrep
from preppy.docs import load_docs

from represented.figs import plot_heatmap
from represented.utils import to_corr_mat, cluster, load_pos_words
from represented import configs


CORPUS_NAME = 'childes-20191206'
N_COMPONENTS = 32  # 512
PART_IDS = [0, 1]  # this is useful because clustering of second corr_mat is based on dg0 and dg1 of first


corpus_path = config.Dirs.corpora / f'{CORPUS_NAME}.txt'
train_docs, test_docs = load_docs(corpus_path)
prep = PartitionedPrep(train_docs,
                       reverse=False,
                       num_types=None,
                       num_parts=1,
                       num_iterations=(1, 1),
                       batch_size=1,
                       context_size=7,
                       )


pos_words = load_pos_words(f'{CORPUS_NAME}-nouns')


dg0, dg1 = None, None
for part_id in PART_IDS:

    # TODO get representations (with word-order)

    print('shape of reps={}'.format(token_reps.shape))
    assert len(token_reps) == prep.store.num_types, (len(token_reps), prep.store.num_types)

    # pca
    pca = PCA(n_components=N_COMPONENTS)
    token_reps = pca.fit_transform(token_reps)
    print('shape after PCA={}'.format(token_reps.shape))

    # plot
    corr_mat = to_corr_mat(token_reps)
    print('shape of corr_mat={}'.format(corr_mat.shape))
    clustered_corr_mat, rls, cls, dg0, dg1 = cluster(corr_mat, dg0, dg1, prep.store.types, prep.store.types)
    rls = [rl if rl in pos_words else '' for rl in rls]
    plot_heatmap(clustered_corr_mat, rls, cls, label_interval=1)


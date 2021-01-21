"""
Research questions:
1. How well can categories be distinguished in partition 1 vs. partition 2?
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from categoryeval.probestore import ProbeStore

from classy.io import load_tokens
from classy.memory import set_memory_limit
from classy.representation import make_probe_reps_median_split
from classy.contexts import get_probe_contexts

CORPUS_NAME = 'childes-20201026'
PROBES_NAME = 'sem-all'
PRESERVE_WORD_ORDER = True
CONTEXT_SIZE = 3

tokens = load_tokens(CORPUS_NAME)
w2id = {w: i for i, w in enumerate(set(tokens))}
probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, w2id)


probe2contexts, context_types = get_probe_contexts(probe_store.types,
                                                   tokens,
                                                   CONTEXT_SIZE,
                                                   PRESERVE_WORD_ORDER)

x1 = make_probe_reps_median_split(probe2contexts, context_types, split_id=0)
x2 = make_probe_reps_median_split(probe2contexts, context_types, split_id=1)

# note: LDA classifier appears to use l2 normalization internally
# because results are same if normalization is performed externally


set_memory_limit(prop=1.0)

# prepare y
y1 = [probe_store.cat2id[probe_store.probe2cat[p]] for p in probe_store.types]
y2 = [probe_store.cat2id[probe_store.probe2cat[p]] for p in probe_store.types]

for x, y in zip([x1, x2],
                [y1, y2]):

    print(f'Shape of input to classifier={x.shape}')
    clf = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                                     solver='svd', store_covariance=False)
    try:
        clf.fit(x, y)

    except MemoryError as e:
        raise SystemExit('Reached memory limit')

    # mean-accuracy
    score = clf.score(x, y)
    print(f'accuracy={score:.3f}')

print('CONTEXT_SIZE', CONTEXT_SIZE)
print('ORDERED', PRESERVE_WORD_ORDER)
"""
Research question:
1. Are semantic categories in part 1 of AO-CHILDES more easily clustered according to the gold structure?

There are two answers, depending on whether
* probe representations take into consideration word-order
* probe representations do not take into consideration word-order


this script does not directly compare performance between two partitions, as this might be unfair
 due to differential number of probes occurring in each.
instead, it ensures that for each probe, the same number of contexts occur in each "split"
(the word "partitions" doesn't really apply here, as the corpus isn't split, but contexts are split)

"""
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate


from categoryeval.probestore import ProbeStore
from categoryeval.ba import calc_score

from classy.representation import make_probe_reps_median_split
from classy.contexts import get_probe_contexts
from classy.io import load_tokens
from classy.memory import set_memory_limit

CORPUS_NAME = 'childes-20201026'
PROBES_NAME = 'sem-all'  # careful: some probe reps might be zero vectors if they do not occur in part
CONTEXT_SIZES = [1, 2, 3, 4]
METRIC = 'ck'
PRESERVE_WORD_ORDER = False

tokens = load_tokens(CORPUS_NAME)
w2id = {w: i for i, w in enumerate(set(tokens))}
probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, w2id)


if METRIC == 'ba':
    y_lims = [0.5, 1.0]
elif METRIC == 'ck':
    y_lims = [0.0, 0.2]
elif METRIC == 'f1':
    y_lims = [0.0, 0.2]
else:
    raise AttributeError('Invalid arg to "METRIC".')


set_memory_limit(prop=0.9)

split_ids = range(2)
split_id2scores = {split_id: [] for split_id in split_ids}

for context_size in CONTEXT_SIZES:

    probe2contexts, context_types = get_probe_contexts(probe_store.types,
                                                       tokens,
                                                       context_size,
                                                       PRESERVE_WORD_ORDER)

    for split_id in split_ids:

        probe_reps = make_probe_reps_median_split(probe2contexts, context_types, split_id)

        try:
            score = calc_score(cosine_similarity(probe_reps), probe_store.gold_sims, metric=METRIC)
        except MemoryError:
            print(RuntimeWarning('Warning: Memory Error'))
            break  # print table before exiting

        # collect
        split_id2scores[split_id].append(score)
        print('split_id={} score={:.3f}'.format(split_id, score))

# table
print(f'{PROBES_NAME} category information in {CORPUS_NAME}\ncaptured by CT representations')
headers = ['Split'] + CONTEXT_SIZES
rows = [['1'] + split_id2scores[0],
        ['2'] + split_id2scores[1]]
print(tabulate(rows,
               headers=headers,
               floatfmt=".4f"))
from typing import List, Tuple, Dict

from sortedcontainers import SortedSet, SortedDict

from represented.utils import get_sliding_windows


def get_probe_contexts(probes: SortedSet,
                       tokens: List[str],
                       context_size: int,
                       preserve_order: bool,
                       ) -> Tuple[Dict[str, Tuple[str]], SortedSet]:
    # get all probe contexts
    probe2contexts = SortedDict({p: [] for p in probes})
    contexts_in_order = get_sliding_windows(context_size, tokens)
    context_types = SortedSet()
    for n, context in enumerate(contexts_in_order[:-context_size]):
        next_context = contexts_in_order[n + 1]
        target = next_context[-1]
        if target in probes:

            if preserve_order:
                probe2contexts[target].append(context)
                context_types.add(context)
            else:
                single_word_contexts = [(w,) for w in context]
                probe2contexts[target].extend(single_word_contexts)
                context_types.update(single_word_contexts)

    return probe2contexts, context_types
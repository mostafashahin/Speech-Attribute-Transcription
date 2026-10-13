"""Per-attribute-group loss weighting for the SCTC-SB trainer.

Added for the class-weighted-loss experiment (Thesis C). Python 3.6 compatible.

The SCTC trainer computes one binary CTC loss per attribute group and sums
them. Rare attributes (positive in <2 % of training tokens) contribute almost
nothing to that sum, because most utterances in a batch carry no positive for
them and their CTC loss is near zero. This module re-allocates the summed loss
across groups by how rare each group's positive token is, WITHOUT changing the
overall scale of the loss:

    f_g   = (# p_<g> tokens) / (# tokens of group g)   over the TRAIN split
    raw_g = (f_med / f_g) ** alpha        f_med = median of f_g over groups with f_g > 0
    raw_g = clip(raw_g, 1, max_weight)    frequent groups are never down-weighted
                                          before normalisation; no group gets more
                                          than max_weight
    raw_g = 1                             if f_g == 0 (no positives to learn from)
    w_g   = raw_g / mean(raw)             so mean(w) == 1 and the summed loss keeps
                                          its scale (gradient clipping and Adam
                                          see comparable magnitudes)

mode 'none' returns None, and the trainer then runs the original code path
unchanged, so existing configs reproduce bit-for-bit.
"""

import json
import logging

logger = logging.getLogger(__name__)

SUPPORTED_MODES = ('none', 'group_inv_freq')

DEFAULTS = {
    'mode': 'none',
    'alpha': 0.5,
    'max_weight': 10.0,
}


def read_config(config):
    """Return a validated loss_weighting dict from the parsed yaml (top-level key
    `loss_weighting`). Missing block == mode none."""
    block = dict(DEFAULTS)
    block.update(config.get('loss_weighting', {}) or {})
    mode = str(block['mode'])
    if mode not in SUPPORTED_MODES:
        raise ValueError('loss_weighting.mode must be one of %s, got %r'
                         % (', '.join(SUPPORTED_MODES), mode))
    block['mode'] = mode
    block['alpha'] = float(block['alpha'])
    block['max_weight'] = float(block['max_weight'])
    if block['alpha'] < 0:
        raise ValueError('loss_weighting.alpha must be >= 0')
    if block['max_weight'] < 1:
        raise ValueError('loss_weighting.max_weight must be >= 1')
    return block


def count_positive_fractions(label_rows, p_token_ids):
    """label_rows: iterable over utterances; each is a list (one per group) of
    token-id lists, as produced by TrainSAModel._prepare_dataset ('labels').
    p_token_ids: list, the vocab id of p_<att> for each group, in group order.
    Returns (pos_counts, tok_counts) lists, one entry per group."""
    ngroups = len(p_token_ids)
    pos = [0] * ngroups
    tot = [0] * ngroups
    for row in label_rows:
        if len(row) != ngroups:
            raise ValueError('label row has %d groups, expected %d' % (len(row), ngroups))
        for g in range(ngroups):
            seq = row[g]
            tot[g] += len(seq)
            pid = p_token_ids[g]
            pos[g] += sum(1 for t in seq if t == pid)
    return pos, tot


def compute_group_weights(pos_counts, tok_counts, alpha, max_weight):
    """Pure function, see module docstring. Returns list of floats, mean 1."""
    n = len(pos_counts)
    if n == 0:
        return []
    freqs = []
    for p, t in zip(pos_counts, tok_counts):
        freqs.append(float(p) / t if t > 0 else 0.0)
    nonzero = sorted(f for f in freqs if f > 0)
    if not nonzero:
        return [1.0] * n
    m = len(nonzero)
    f_med = nonzero[m // 2] if m % 2 == 1 else 0.5 * (nonzero[m // 2 - 1] + nonzero[m // 2])
    raw = []
    for f in freqs:
        if f <= 0:
            raw.append(1.0)
        else:
            r = (f_med / f) ** alpha
            r = max(1.0, min(max_weight, r))
            raw.append(r)
    mean = sum(raw) / n
    return [r / mean for r in raw]


def build_weights(trainer_obj, cfg):
    """Called from TrainSAModel after load_data() and create_processor().
    Returns None for mode 'none', else a list of per-group weights.
    Also writes <working_dir>/loss_weights.json for the experiment record."""
    if cfg['mode'] == 'none':
        return None
    groups = trainer_obj.groups
    tok = trainer_obj.processor.tokenizer
    p_ids = [tok.convert_tokens_to_ids(g[0]) for g in groups]
    rows = trainer_obj.data_train['labels']
    pos, tot = count_positive_fractions(rows, p_ids)
    weights = compute_group_weights(pos, tot, cfg['alpha'], cfg['max_weight'])
    record = {
        'mode': cfg['mode'],
        'alpha': cfg['alpha'],
        'max_weight': cfg['max_weight'],
        'n_train_utterances': len(rows),
        'groups': [],
    }
    for g, w, p, t in zip(groups, weights, pos, tot):
        record['groups'].append({
            'attribute': g[0][2:],
            'pos_tokens': p,
            'tokens': t,
            'pos_fraction': (float(p) / t) if t else 0.0,
            'weight': w,
        })
    import os
    out = os.path.join(trainer_obj.working_dir, 'loss_weights.json')
    with open(out, 'w') as f:
        json.dump(record, f, indent=2)
    logger.info('loss_weighting %s: weights written to %s (min %.3f, max %.3f)'
                % (cfg['mode'], out, min(weights), max(weights)))
    return weights

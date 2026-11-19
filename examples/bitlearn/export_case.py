"""Export a bitlearn 0.2.0 model, the inputs of one training epoch and bitlearn's own
results, so the CUDA port (bitlearn_tm.cu) can be checked bit for bit.

    python export_case.py <out_dir>

Data: scikit-learn's bundled 8x8 digits (no download). Files are raw little-endian arrays
plus meta.txt; the CUDA program byte-swaps on big-endian machines.
"""
import os
import sys

import numpy as np
from sklearn.datasets import load_digits

import bitlearn
from bitlearn import TsetlinClassifier, pack_literals
from bitlearn import _core
from bitlearn.classifier import _random_pool, _word_mask


def save(out, name, arr, dtype):
    a = np.ascontiguousarray(arr, dtype=dtype)
    a.astype(a.dtype.newbyteorder('<'), copy=False).tofile(os.path.join(out, name + '.bin'))


def main(out):
    os.makedirs(out, exist_ok=True)
    digits = load_digits()
    X = (digits.data > 4).astype(np.uint8)          # 64 binary features
    y = digits.target.astype(np.int64)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    tr, te = idx[:1400], idx[1400:]

    clf = TsetlinClassifier(n_clauses=200, T=20, s=4.0, random_state=7)
    clf.fit(X[tr], y[tr], epochs=3)
    acc = clf.score(X[te], y[te])
    sums = clf.decision_function(X[te])               # bitlearn's inference result

    Xp = pack_literals(X[tr])
    Xp_test = pack_literals(X[te])
    mask = _word_mask(X.shape[1])
    k, c, b, w = clf.ta_.shape
    meta = {'n_classes': k, 'n_clauses': c, 'n_bits': b, 'n_words': w, 'n_features': X.shape[1],
            'n_train': len(tr), 'n_test': len(te), 'bitlearn_version': bitlearn.__version__,
            'bitlearn_accuracy': f'{acc:.6f}', 'python': sys.version.split()[0]}

    save(out, 'ta0', clf.ta_, np.uint64)
    save(out, 'w0', clf.weights_, np.int32)
    save(out, 'xp_train', Xp, np.uint64)
    save(out, 'y_train', y[tr], np.int64)
    save(out, 'xp_test', Xp_test, np.uint64)
    save(out, 'y_test', y[te], np.int64)
    save(out, 'sums', sums, np.int32)
    save(out, 'mask', mask, np.uint64)

    # one more epoch, run by bitlearn's own kernel, in two feedback modes
    cases = [('a', dict(T=20, boost=True, weighted=False, max_weight=255)),
             ('b', dict(T=25, boost=False, weighted=True, max_weight=7))]
    for tag, opt in cases:
        r = np.random.default_rng(100 + ord(tag))
        order = r.permutation(len(tr)).astype(np.int64)
        pool = _random_pool(r, 1.0 / 4.0)
        seeds = r.integers(1, 2**63, size=k, dtype=np.uint64)
        ta = clf.ta_.copy()
        wts = clf.weights_.copy()
        _core.fit_epoch(ta, wts, Xp, y[tr], order, mask, pool, opt['T'], opt['boost'], opt['weighted'],
                        opt['max_weight'], seeds)
        changed = int(np.count_nonzero(ta != clf.ta_))
        save(out, f'order_{tag}', order, np.int64)
        save(out, f'pool_{tag}', pool, np.uint64)
        save(out, f'seeds_{tag}', seeds, np.uint64)
        save(out, f'ta1_{tag}', ta, np.uint64)
        save(out, f'w1_{tag}', wts, np.int32)
        meta.update({f'T_{tag}': opt['T'], f'boost_{tag}': int(opt['boost']),
                     f'weighted_{tag}': int(opt['weighted']), f'max_weight_{tag}': opt['max_weight'],
                     f'pool_len_{tag}': len(pool), f'changed_words_{tag}': changed})
    with open(os.path.join(out, 'meta.txt'), 'w', encoding='utf-8') as f:
        for key, val in meta.items():
            f.write(f'{key}={val}\n')
    print(f'bitlearn {bitlearn.__version__} on Python {meta["python"]}: test accuracy {acc:.4f}; '
          f'case files written to {out}')
    print({kk: vv for kk, vv in meta.items() if kk.startswith('changed')})


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'tm_case')

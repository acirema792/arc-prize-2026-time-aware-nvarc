"""Program-search solver for ARC-AGI tasks.

Each generator looks at the training pairs and proposes candidate programs
(grid -> grid functions). A program is kept only if it reproduces every
training output exactly; the first two distinct test predictions become
attempt_1 and attempt_2. Generators are ordered from simplest to most
flexible, so simpler explanations win ties.
"""
import time
from collections import Counter

import numpy as np

# ---------------------------------------------------------------- utilities

DIHEDRAL = [
    ("id", lambda g: g),
    ("rot90", lambda g: np.rot90(g, 1)),
    ("rot180", lambda g: np.rot90(g, 2)),
    ("rot270", lambda g: np.rot90(g, 3)),
    ("flipud", lambda g: g[::-1]),
    ("fliplr", lambda g: g[:, ::-1]),
    ("transpose", lambda g: g.T),
    ("antitranspose", lambda g: np.rot90(g, 2).T),
]


def background(g):
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(counts)])


def components(g, bg, diag=True, multicolor=False):
    """Connected components of non-background cells: list of (mask, color_set)."""
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    out = []
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diag:
        nbrs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for r in range(h):
        for c in range(w):
            if seen[r, c] or g[r, c] == bg:
                continue
            col = g[r, c]
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and g[ny, nx] != bg \
                            and (multicolor or g[ny, nx] == col):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            m = np.zeros_like(g, dtype=bool)
            ys, xs = zip(*cells)
            m[list(ys), list(xs)] = True
            out.append(m)
    return out


def bbox(mask):
    ys, xs = np.nonzero(mask)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def learn_colormap(pairs):
    """Consistent per-pixel color map a->b over same-shape pairs, or None."""
    m = {}
    for a, b in pairs:
        if a.shape != b.shape:
            return None
        for x, y in zip(a.ravel().tolist(), b.ravel().tolist()):
            if m.setdefault(x, y) != y:
                return None
    return m


def apply_colormap(g, m):
    out = g.copy()
    for a, b in m.items():
        out[g == a] = b
    return out


# --------------------------------------------------------------- generators
# Each generator: gen(train) -> iterable of (name, fn). train is a list of
# (input ndarray, output ndarray). Programs are verified by the caller.

def gen_dihedral(train):
    for name, f in DIHEDRAL:
        yield name, f


def gen_dihedral_colormap(train):
    for name, f in DIHEDRAL:
        try:
            m = learn_colormap([(f(a), b) for a, b in train])
        except Exception:
            m = None
        if m:
            yield f"{name}+cmap", (lambda g, f=f, m=m: apply_colormap(f(g), m))


def gen_constant(train):
    outs = [b for _, b in train]
    if all(o.shape == outs[0].shape and (o == outs[0]).all() for o in outs):
        yield "constant", (lambda g, o=outs[0]: o.copy())


def gen_upscale(train):
    a, b = train[0]
    if b.shape[0] % a.shape[0] or b.shape[1] % a.shape[1]:
        return
    ky, kx = b.shape[0] // a.shape[0], b.shape[1] // a.shape[1]
    for name, f in DIHEDRAL:
        yield f"upscale{ky}x{kx}+{name}", (lambda g, f=f: np.kron(f(g), np.ones((ky, kx), dtype=g.dtype)))
    # fractal: each input cell becomes a copy of the input (or blank)
    if (ky, kx) == a.shape:
        bg = background(a)
        yield "fractal", (lambda g: np.kron(g != bg, np.ones_like(g)) * np.tile(g, g.shape) + bg * (1 - np.kron(g != bg, np.ones_like(g))))
        yield "fractal_inv", (lambda g: np.kron(g == bg, np.ones_like(g)) * np.tile(g, g.shape) + bg * (1 - np.kron(g == bg, np.ones_like(g))))


def gen_tile(train):
    """Output is a ky x kx tiling of transformed copies of the input."""
    a, b = train[0]
    if b.shape[0] % a.shape[0] or b.shape[1] % a.shape[1]:
        return
    ky, kx = b.shape[0] // a.shape[0], b.shape[1] // a.shape[1]
    if ky * kx == 1:
        return
    # choose, per tile slot, the dihedral transform consistent with all pairs
    choice = {}
    for ty in range(ky):
        for tx in range(kx):
            for name, f in DIHEDRAL:
                ok = True
                for x, y in train:
                    h, w = x.shape
                    if y.shape != (h * ky, w * kx):
                        return
                    fx = f(x)
                    if fx.shape != (h, w) or not (y[ty * h:(ty + 1) * h, tx * w:(tx + 1) * w] == fx).all():
                        ok = False
                        break
                if ok:
                    choice[(ty, tx)] = f
                    break
            else:
                return

    def run(g):
        h, w = g.shape
        out = np.zeros((h * ky, w * kx), dtype=g.dtype)
        for (ty, tx), f in choice.items():
            out[ty * h:(ty + 1) * h, tx * w:(tx + 1) * w] = f(g)
        return out
    yield f"tile{ky}x{kx}", run


def gen_downscale(train):
    a, b = train[0]
    if a.shape[0] % b.shape[0] or a.shape[1] % b.shape[1]:
        return
    ky, kx = a.shape[0] // b.shape[0], a.shape[1] // b.shape[1]
    if ky * kx == 1:
        return

    def blocks(g):
        h, w = g.shape[0] // ky, g.shape[1] // kx
        return g[:h * ky, :w * kx].reshape(h, ky, w, kx).transpose(0, 2, 1, 3).reshape(h, w, ky * kx)

    yield "subsample", (lambda g: g[::ky, ::kx])

    def majority(g):
        bl = blocks(g)
        return np.apply_along_axis(lambda v: np.bincount(v, minlength=10).argmax(), 2, bl)

    def anycolor(g):
        bg = background(g)
        bl = blocks(g)
        return np.apply_along_axis(lambda v: next((c for c in v if c != bg), bg), 2, bl)
    yield "block_majority", majority
    yield "block_any", anycolor


def _object_selectors():
    """Rules that pick one object from a list of (mask, grid) by a property."""
    def size(m, g): return m.sum()
    def area(m, g):
        y0, y1, x0, x1 = bbox(m)
        return (y1 - y0) * (x1 - x0)
    def ncolors(m, g): return len(np.unique(g[m]))
    # position-based (top/left) and density selectors fit training pairs by
    # coincidence too often: 5 of the 6 wrong programs on the training set
    props = [("size", size), ("area", area), ("ncolors", ncolors)]
    sels = []
    for pname, p in props:
        sels.append((f"max_{pname}", p, max))
        sels.append((f"min_{pname}", p, min))
    return sels


def _color_rarity_select(objs, g, rarest):
    cols = [Counter(g[m].tolist()).most_common(1)[0][0] for m in objs]
    cnt = Counter(cols)
    target = [c for c in cnt if cnt[c] == (min if rarest else max)(cnt.values())]
    cand = [m for m, c in zip(objs, cols) if c in target]
    return cand[0] if len(cand) == 1 else None


def _shape_key(m, g):
    y0, y1, x0, x1 = bbox(m)
    return (m[y0:y1, x0:x1].tobytes(), m[y0:y1, x0:x1].shape)


def gen_crop(train):
    """Output is the bounding-box crop of one selected object (or of all content)."""
    def crop_all(g):
        bg = background(g)
        y0, y1, x0, x1 = bbox(g != bg)
        return g[y0:y1, x0:x1]
    yield "crop_content", crop_all

    for diag in (True, False):
        for multi in (True, False):
            for sname, p, agg in _object_selectors():
                def run(g, diag=diag, multi=multi, p=p, agg=agg, masked=False):
                    bg = background(g)
                    objs = components(g, bg, diag, multi)
                    if not objs:
                        return None
                    vals = [p(m, g) for m in objs]
                    best = agg(vals)
                    pick = [m for m, v in zip(objs, vals) if v == best]
                    if len(pick) != 1:
                        return None
                    y0, y1, x0, x1 = bbox(pick[0])
                    return g[y0:y1, x0:x1]
                yield f"crop_{sname}_d{int(diag)}m{int(multi)}", run
            for rarest in (True, False):
                def run2(g, diag=diag, multi=multi, rarest=rarest):
                    bg = background(g)
                    objs = components(g, bg, diag, multi)
                    m = _color_rarity_select(objs, g, rarest) if objs else None
                    if m is None:
                        return None
                    y0, y1, x0, x1 = bbox(m)
                    return g[y0:y1, x0:x1]
                yield f"crop_{'rare' if rarest else 'common'}color_d{int(diag)}m{int(multi)}", run2

            def run3(g, diag=diag, multi=multi):
                # the object whose shape is unique among the objects
                bg = background(g)
                objs = components(g, bg, diag, multi)
                keys = [_shape_key(m, g) for m in objs]
                cnt = Counter(keys)
                pick = [m for m, k in zip(objs, keys) if cnt[k] == 1]
                if len(pick) != 1:
                    return None
                y0, y1, x0, x1 = bbox(pick[0])
                return g[y0:y1, x0:x1]
            yield f"crop_uniqueshape_d{int(diag)}m{int(multi)}", run3


def _split_parts(g):
    """Split a grid into cells separated by full-length single-color lines."""
    h, w = g.shape
    rows = [r for r in range(h) if len(set(g[r].tolist())) == 1]
    cols = [c for c in range(w) if len(set(g[:, c].tolist())) == 1]
    sep_colors = set(g[r, 0] for r in rows) | set(g[0, c] for c in cols)
    # keep only separators of a color that never appears as a 'content' line everywhere
    parts = []
    if rows or cols:
        for sc in sep_colors:
            rr = [-1] + [r for r in rows if g[r, 0] == sc] + [h]
            cc = [-1] + [c for c in cols if g[0, c] == sc] + [w]
            ps = []
            for i in range(len(rr) - 1):
                for j in range(len(cc) - 1):
                    if rr[i + 1] - rr[i] > 1 and cc[j + 1] - cc[j] > 1:
                        ps.append(g[rr[i] + 1:rr[i + 1], cc[j] + 1:cc[j + 1]])
            if len(ps) >= 2 and len(set(p.shape for p in ps)) == 1:
                parts.append(ps)
    return parts


def _halves(g, n, axis):
    if g.shape[axis] % n:
        return None
    return np.split(g, n, axis=axis)


def gen_split_logic(train):
    """Output combines equally sized parts of the input with a boolean op."""
    out_colors = set()
    for _, b in train:
        out_colors |= set(np.unique(b).tolist())

    def part_lists(g):
        res = list(_split_parts(g))
        for n in (2, 3):
            for ax in (0, 1):
                p = _halves(g, n, ax)
                if p:
                    res.append(p)
        return res

    ops = {
        "and": lambda m: np.logical_and.reduce(m),
        "or": lambda m: np.logical_or.reduce(m),
        "xor": lambda m: np.logical_xor.reduce(m),
        "nor": lambda m: ~np.logical_or.reduce(m),
        "nand": lambda m: ~np.logical_and.reduce(m),
        "a_not_b": lambda m: m[0] & ~np.logical_or.reduce(m[1:]),
        "b_not_a": lambda m: m[-1] & ~np.logical_or.reduce(m[:-1]),
    }
    n_ways = len(part_lists(train[0][0]))
    for way in range(n_ways):
        for opname, op in ops.items():
            for fg in out_colors:
                for bgc in out_colors:
                    if fg == bgc:
                        continue

                    def run(g, way=way, op=op, fg=fg, bgc=bgc):
                        pl = part_lists(g)
                        if way >= len(pl):
                            return None
                        parts = pl[way]
                        bg = Counter(np.concatenate([p.ravel() for p in parts]).tolist()).most_common(1)[0][0]
                        m = op([p != bg for p in parts])
                        return np.where(m, fg, bgc)
                    yield f"split{way}_{opname}_{fg}_{bgc}", run

        # overlay parts in priority order (later parts drawn on top of earlier ones)
        for order in ("fwd", "rev"):
            def run_ov(g, way=way, order=order):
                pl = part_lists(g)
                if way >= len(pl):
                    return None
                parts = pl[way] if order == "fwd" else pl[way][::-1]
                bg = Counter(np.concatenate([p.ravel() for p in parts]).tolist()).most_common(1)[0][0]
                out = parts[0].copy()
                for p in parts[1:]:
                    out[p != bg] = p[p != bg]
                return out
            yield f"split{way}_overlay_{order}", run_ov

        def run_odd(g, way=way):
            # the one part that differs from the others
            pl = part_lists(g)
            if way >= len(pl):
                return None
            keys = [p.tobytes() for p in pl[way]]
            cnt = Counter(keys)
            pick = [p for p, k in zip(pl[way], keys) if cnt[k] == 1]
            return pick[0] if len(pick) == 1 else None
        yield f"split{way}_oddone", run_odd


def gen_gravity(train):
    def make(direction):
        def run(g):
            bg = background(g)
            k = {"down": 0, "left": 1, "up": 2, "right": 3}[direction]
            r = np.rot90(g, k).copy()
            for c in range(r.shape[1]):
                col = r[:, c]
                vals = col[col != bg]
                r[:, c] = np.concatenate([np.full(len(col) - len(vals), bg), vals])
            return np.rot90(r, -k)
        return run
    for d in ("down", "up", "left", "right"):
        yield f"gravity_{d}", make(d)


def _enclosed(g, bg):
    """Background cells not 4-connected to the border."""
    h, w = g.shape
    reach = np.zeros_like(g, dtype=bool)
    stack = [(r, c) for r in range(h) for c in (0, w - 1)] + [(r, c) for c in range(w) for r in (0, h - 1)]
    while stack:
        y, x = stack.pop()
        if 0 <= y < h and 0 <= x < w and not reach[y, x] and g[y, x] == bg:
            reach[y, x] = True
            stack += [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]
    return (g == bg) & ~reach


def gen_fill_enclosed(train):
    cols = set()
    for a, b in train:
        if a.shape != b.shape:
            return
        cols |= set(b[a != b].tolist())
    for c in cols:
        def run(g, c=c):
            out = g.copy()
            out[_enclosed(g, background(g))] = c
            return out
        yield f"fill_enclosed_{c}", run


# local cellular-automaton style rules: out[y,x] = table[features(y,x)]

def _neigh_features(kind):
    def feat(g, bg):
        h, w = g.shape
        p = np.pad(g, 2, constant_values=-1)
        F = []
        for y in range(h):
            for x in range(w):
                Y, X = y + 2, x + 2
                if kind == "c":
                    f = (p[Y, X],)
                elif kind == "c4":
                    f = (p[Y, X], p[Y - 1, X], p[Y + 1, X], p[Y, X - 1], p[Y, X + 1])
                elif kind == "c8":
                    f = tuple(p[Y - 1:Y + 2, X - 1:X + 2].ravel())
                elif kind == "c24":
                    f = tuple(p[Y - 2:Y + 3, X - 2:X + 3].ravel())
                elif kind == "rowcol":
                    # center + set of colors in row and column
                    f = (p[Y, X], frozenset(g[y].tolist()) - {bg}, frozenset(g[:, x].tolist()) - {bg})
                elif kind == "rays":
                    # center + first non-bg color seen in each of the 4 directions
                    def ray(dy, dx):
                        yy, xx = y + dy, x + dx
                        while 0 <= yy < h and 0 <= xx < w:
                            if g[yy, xx] != bg:
                                return g[yy, xx]
                            yy += dy
                            xx += dx
                        return -1
                    f = (p[Y, X], ray(-1, 0), ray(1, 0), ray(0, -1), ray(0, 1))
                F.append(f)
        return F
    return feat


def gen_local_rules(train):
    if any(a.shape != b.shape for a, b in train):
        return
    for kind in ("c", "c4", "c8", "rays", "rowcol", "c24"):
        feat = _neigh_features(kind)
        table = {}
        ok = True
        for a, b in train:
            bg = background(a)
            for f, v in zip(feat(a, bg), b.ravel().tolist()):
                if table.setdefault(f, v) != v:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        def run(g, feat=feat, table=table):
            bg = background(g)
            vals = []
            for f in feat(g, bg):
                if f not in table:
                    return None
                vals.append(table[f])
            return np.array(vals).reshape(g.shape)
        yield f"local_{kind}", run


def gen_object_recolor(train):
    """Each object keeps its shape but is recolored by a property (size, shape, border contact, ...)."""
    if any(a.shape != b.shape for a, b in train):
        return

    def touches(m):
        return bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())

    def rank(objs, f):
        vals = sorted(set(f(m) for m in objs))
        return lambda m: vals.index(f(m))

    props = {
        "size": lambda objs: (lambda m: int(m.sum())),
        "size_rank": lambda objs: rank(objs, lambda m: int(m.sum())),
        "size_rank_desc": lambda objs: (lambda r: (lambda m: -r(m)))(rank(objs, lambda m: -int(m.sum()))),
        "shape": lambda objs: (lambda m: _shape_key(m, None)),
        "border": lambda objs: touches,
        "height": lambda objs: (lambda m: bbox(m)[1] - bbox(m)[0]),
        "width": lambda objs: (lambda m: bbox(m)[3] - bbox(m)[2]),
        "holes": lambda objs: (lambda m: int(_enclosed(np.where(m, 1, 0), 0).sum() > 0)),
    }
    for diag in (False, True):
        for pname, pf in props.items():
            table = {}
            ok = True
            for a, b in train:
                bg = background(a)
                if not ((a == b) | (a != bg)).all():   # only object cells may change
                    ok = False
                    break
                objs = components(a, bg, diag, False)
                f = pf(objs)
                for m in objs:
                    cols = np.unique(b[m])
                    if len(cols) != 1 or table.setdefault(f(m), int(cols[0])) != int(cols[0]):
                        ok = False
                        break
                if not ok:
                    break
            if not ok or not table:
                continue

            def run(g, diag=diag, pf=pf, table=table):
                bg = background(g)
                objs = components(g, bg, diag, False)
                f = pf(objs)
                out = g.copy()
                for m in objs:
                    if f(m) not in table:
                        return None
                    out[m] = table[f(m)]
                return out
            yield f"recolor_{pname}_d{int(diag)}", run


def gen_compose(train):
    """A shape-changing base program followed by a dihedral transform and/or a color map."""
    for base_gen in (gen_crop, gen_split_logic, gen_downscale, gen_upscale):
        for bname, bfn in base_gen(train):
            mids = []
            for a, _ in train:
                m = _safe(bfn, a)
                if m is None:
                    break
                mids.append(m)
            if len(mids) != len(train):
                continue
            for pname, pf in DIHEDRAL[1:]:
                if all(pf(m).shape == b.shape for m, (_, b) in zip(mids, train)):
                    cm = learn_colormap([(pf(m), b) for m, (_, b) in zip(mids, train)])
                    if cm is not None:
                        yield f"{bname}>{pname}", (lambda g, bfn=bfn, pf=pf: pf(bfn(g)))
                        if any(k != v for k, v in cm.items()):
                            yield f"{bname}>{pname}+cmap", (lambda g, bfn=bfn, pf=pf, cm=cm: apply_colormap(pf(bfn(g)), cm))
            if all(m.shape == b.shape for m, (_, b) in zip(mids, train)):
                cm = learn_colormap([(m, b) for m, (_, b) in zip(mids, train)])
                if cm and any(k != v for k, v in cm.items()):
                    yield f"{bname}+cmap", (lambda g, bfn=bfn, cm=cm: apply_colormap(bfn(g), cm))


GENERATORS = [
    gen_constant,
    gen_dihedral,
    gen_dihedral_colormap,
    gen_upscale,
    gen_tile,
    gen_downscale,
    gen_crop,
    gen_split_logic,
    gen_gravity,
    gen_fill_enclosed,
    gen_object_recolor,
    gen_local_rules,
    gen_compose,
]


# ------------------------------------------------------------------- driver

def _safe(fn, g):
    try:
        r = fn(g)
    except Exception:
        return None
    if r is None:
        return None
    r = np.asarray(r)
    if r.ndim != 2 or r.size == 0 or r.shape[0] > 30 or r.shape[1] > 30:
        return None
    return r.astype(int)


def _plausible(pred, test_in, train):
    """Reject degenerate test predictions that no training output resembles."""
    if pred.shape == test_in.shape and (pred == test_in).all() and \
            not any(a.shape == b.shape and (a == b).all() for a, b in train):
        return False
    if len(np.unique(pred)) == 1 and all(len(np.unique(b)) > 1 for _, b in train):
        return False
    return True


def verified_candidates(task, time_budget=20.0):
    """Programs that reproduce every training pair: (names, per-test list of predictions)."""
    t0 = time.time()
    train = [(np.array(p["input"]), np.array(p["output"])) for p in task["train"]]
    tests = [np.array(t["input"]) for t in task["test"]]
    found = []  # (name, [prediction per test])
    for gen in GENERATORS:
        if time.time() - t0 > time_budget or len(found) >= 2:
            break
        try:
            programs = list(gen(train))
        except Exception:
            continue
        for name, fn in programs:
            if time.time() - t0 > time_budget:
                break
            if all((r := _safe(fn, a)) is not None and r.shape == b.shape and (r == b).all() for a, b in train):
                preds = [_safe(fn, t) for t in tests]
                if all(p is not None and _plausible(p, t, train) for p, t in zip(preds, tests)):
                    key = tuple(p.tobytes() + bytes(p.shape) for p in preds)
                    if all(key != k for _, _, k in found):
                        found.append((name, preds, key))
                        if len(found) >= 2:
                            break
    names = [f[0] for f in found]
    return names, [[f[1][i].tolist() for f in found] for i in range(len(tests))]


def solve_task(task, time_budget=20.0, return_names=False):
    names, per_test = verified_candidates(task, time_budget)
    attempts = []
    for cands, t in zip(per_test, task["test"]):
        fallback = t["input"]
        a1 = cands[0] if cands else fallback
        a2 = cands[1] if len(cands) > 1 else (fallback if a1 != fallback else [[0]])
        attempts.append([a1, a2])
    return (attempts, names) if return_names else attempts

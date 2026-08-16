#!/usr/bin/env python3
"""
Step 2 -- Search for a single shared focal length f per view, holding
everything else fixed, instead of jointly fitting the full intrinsics +
distortion + pose. Every fit already in this repo tries to solve too many
parameters (fx, fy, cx, cy, k1, k2, p1, p2, plus 6 pose DOF) from too few
correspondences (2-10 tags per view) and the result shows: the
recovered-correspondence fit in data/calibration/<stem>_refined_intrinsics.json
has 20px RMS baked into its own fit, and the line-curvature distortion
search noted its own cost surface was "nearly flat". Underconstrained
optimization doesn't fail loudly, it just converges on a confident-looking
wrong answer.

So: start from the simplest model that could possibly be right --

  fx = fy = f          (square pixels, one unknown instead of two)
  cx = W/2, cy = H/2    (principal point at the crop centre, fixed)
  dist = 0              (no distortion, fixed)

-- leaving f as the ONLY free parameter, and search it with a real 1-D
objective: for each candidate f, build K and hand it to
pipeline/03_solve_pose.py's actual solve_view() (same RANSAC/SQPNP pose
solve and same centers/corner-fallback logic pipeline/03 uses for real,
not a simplified stand-in), and use its reported inlier RMS reprojection
error as the cost. Coarse grid first (the cost surface isn't guaranteed
unimodal -- a wrong-by-2x focal length can sometimes fake a locally low
residual via a compensating pose), then a golden-section refine around
the best coarse point.

This is deliberately the FIRST rung of a ladder, not the whole ladder.
Once a view has enough correspondence points (>=
MIN_POINTS_FOR_PRINCIPAL_POINT), the second rung -- cx, cy -- gets added
automatically: a view that's a Y-only crop of a larger sensor readout
(this project's "main", see pipeline/00_split_views.py) has no reason to
have its true principal point sit at the crop's own geometric centre, and
empirically it doesn't (search on this project's own footage moved
"main"'s cy by +200px, a real, physically-motivated correction, not
overfitting -- cx barely moved, consistent with that view being uncropped
in X). Coordinate-descent alternates refining f and refining (cx, cy)
until it converges.

Third rung, k1 (>= MIN_POINTS_FOR_DISTORTION points): added only after
leave-one-out cross-validation justified it, not just eyeballing a lower
in-sample residual (which improves with ANY extra parameter, trivially,
telling you nothing about whether it's real). The tell that k1 specifically
was missing: a well-constrained fit (a few px on the tags it was fit to)
that still visibly diverges on the field boundary -- points nowhere near
any tag, or near the image edges where radial distortion is strongest. A
camera model that's actually correct is a real geometric law, not a local
curve fit -- it must extrapolate correctly to ANY point in the rigid
scene, not just the ones it was fit against. Divergence specifically on
the unfit/far points is itself the diagnostic. Confirmed on this project's
own 2026 Worlds footage: dropping each tag out of "main"'s fit in turn and
re-solving without it, mean reprojection error on the HELD-OUT tag fell
from 5.2px (k1=0) to 2.8px (k1 free) -- real signal, not overfitting, on
8 of 9 tags.

Outputs
-------
  data/calibration/<stem>_intrinsics.json
    merged into any existing file -- only the searched views' entries are
    replaced, matching the exact schema pipeline/03_solve_pose.py and
    viz/03_overlay.py already read:
      { "<view>": { "K": [[f,0,cx],[0,f,cy],[0,0,1]], "dist": [k1,0,0,0,0],
                    "image_size": [w,h], "focal_px": f, "cx": ..., "cy": ...,
                    "k1": ..., "rms_reproj_all_px": ...,
                    "fit": "focal_search_1d" | "focal_and_pp_search"
                           | "focal_pp_k1_search" }, ... }

Usage
-----
  python pipeline/02_search_focal.py --video match.mp4
  python pipeline/02_search_focal.py --video match.mp4 --view main --f-min 400 --f-max 2500

Install: pip install opencv-python numpy
"""

import argparse, importlib.util, io, json, math, pathlib, sys
from contextlib import redirect_stderr
import numpy as np

DATA_DIR       = pathlib.Path(__file__).parent.parent / "data"
FIELD_DIR      = DATA_DIR / "field"
DETECTIONS_DIR = DATA_DIR / "detections"
CALIB_DIR      = DATA_DIR / "calibration"

DEFAULT_F_MIN = 250.0
DEFAULT_F_MAX = 3000.0
COARSE_STEP   = 25.0

# pipeline/03_solve_pose.py's filename starts with a digit -- can't `import`
# it as a module name, so load it by path instead. This reuses its actual
# solve_view() (RANSAC + SQPNP fallback + the centers/corner-fallback logic)
# rather than reimplementing a simplified stand-in that could silently
# diverge from what pipeline/03 really does.
_spec = importlib.util.spec_from_file_location(
    "solve_pose", pathlib.Path(__file__).parent / "03_solve_pose.py")
solve_pose = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(solve_pose)


# ---------------------------------------------------------------------------
# 1-D search
# ---------------------------------------------------------------------------

def _cost(f: float, view_name: str, view_data: dict, field_tags: dict,
         cx: float, cy: float, k1: float = 0.0) -> float:
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    with redirect_stderr(io.StringIO()):   # solve_view is chatty; silence during search
        result = solve_pose.solve_view(view_name, view_data, field_tags, K, dist,
                                       do_per_frame=False)
    if result is None:
        return math.inf
    # Deliberately rms_reproj_all_px (every correspondence), NOT the
    # RANSAC inlier-only rms_reproj_px -- confirmed on this project's own
    # "main" view that the inlier-only number can be gamed: at f=375px
    # RANSAC quietly drops 5/10 tags as "outliers" and reports a great
    # rms=2.8 on the remaining 5, while those same 5 dropped tags land
    # off by hundreds of pixels (all-points rms=1570) -- a wrong-by-2.5x
    # focal length faking a low residual via a smaller inlier set. The
    # real, broad, physically-plausible basin (f=850-1000, matching this
    # project's earlier hand-fit and the CV605-BK spec range) only shows
    # up once every point has to agree, not just whichever subset RANSAC
    # kept.
    return result["rms_reproj_all_px"]


def _golden_section_min(fn, lo: float, hi: float, tol: float = 0.5, max_iter: int = 60):
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = fn(d)
    xm = (a + b) / 2
    return xm, fn(xm)


def search_focal(view_name: str, view_data: dict, field_tags: dict,
                 cx: float, cy: float, f_min: float, f_max: float, coarse_step: float,
                 k1: float = 0.0):
    curve = []
    f = f_min
    while f <= f_max:
        curve.append((f, _cost(f, view_name, view_data, field_tags, cx, cy, k1)))
        f += coarse_step
    finite = [(f, c) for f, c in curve if math.isfinite(c)]
    if not finite:
        return None

    best_f, best_cost = min(finite, key=lambda fc: fc[1])
    lo = max(f_min, best_f - 2 * coarse_step)
    hi = min(f_max, best_f + 2 * coarse_step)
    refined_f, refined_cost = _golden_section_min(
        lambda x: _cost(x, view_name, view_data, field_tags, cx, cy, k1), lo, hi)
    if refined_cost > best_cost:   # golden-section assumes unimodality; fall back if it drifted
        refined_f, refined_cost = best_f, best_cost

    # Flatness check: how much does cost change +/-100px from the minimum?
    # A near-flat neighborhood means f isn't actually well-constrained by
    # this view's correspondences, however low the minimum looks -- see
    # module docstring on why that's exactly the failure mode to distrust.
    def cost_at_offset(off):
        target = refined_f + off
        if target < f_min or target > f_max:
            return None
        return _cost(target, view_name, view_data, field_tags, cx, cy, k1)

    c_minus = cost_at_offset(-100.0)
    c_plus  = cost_at_offset(100.0)
    flatness = None
    if c_minus is not None and c_plus is not None and refined_cost > 1e-6:
        flatness = (min(c_minus, c_plus) - refined_cost) / refined_cost

    return {
        "f": refined_f, "rms": refined_cost, "curve": curve,
        "flatness_ratio": flatness,   # low (<~0.2) => poorly constrained
    }


# Below this many correspondence points, freeing cx/cy on top of f is more
# parameters than the data can support (see module docstring's whole
# argument for starting minimal) -- stay at the crop-center assumption.
MIN_POINTS_FOR_PRINCIPAL_POINT = 20


def search_principal_point(view_name: str, view_data: dict, field_tags: dict,
                           f: float, cx0: float, cy0: float, w: int, h: int,
                           k1: float = 0.0, grid_n: int = 13):
    """
    Coarse (cx, cy) grid search with f held fixed, +/- 40% of the crop
    half-width/height around the crop-centre seed -- generous enough to
    find a genuinely off-centre principal point (expected for a view
    that's a crop of a larger sensor readout, see module docstring) without
    wandering into implausible territory.
    """
    best = None
    for cx in np.linspace(cx0 - 0.4 * w, cx0 + 0.4 * w, grid_n):
        for cy in np.linspace(cy0 - 0.4 * h, cy0 + 0.4 * h, grid_n):
            c = _cost(f, view_name, view_data, field_tags, cx, cy, k1)
            if best is None or c < best[0]:
                best = (c, cx, cy)
    return best  # (cost, cx, cy)


# Below this many correspondence points, adding k1 on top of f/cx/cy is one
# parameter too many to trust (see module docstring's ladder argument) --
# leave-one-out cross-validation on this project's own "main" views (2026 +
# 2025 Worlds footage, both well past this threshold) is what justified
# adding k1 to the ladder at all: mean HELD-OUT tag reprojection error
# (not in-sample, which improves with any extra parameter trivially) fell
# from 5.2px to 2.8px, improving 8/9 tags -- real signal, not overfitting.
MIN_POINTS_FOR_DISTORTION = 30


def search_k1(view_name: str, view_data: dict, field_tags: dict,
             f: float, cx: float, cy: float, k1_range: float = 0.5):
    """1-D golden-section search for k1, f/cx/cy held fixed."""
    return _golden_section_min(
        lambda k1: _cost(f, view_name, view_data, field_tags, cx, cy, k1),
        -k1_range, k1_range, tol=0.002)


def search_intrinsics(view_name: str, view_data: dict, field_tags: dict, w: int, h: int,
                      f_min: float, f_max: float, coarse_step: float,
                      fit_principal_point: bool, fit_distortion: bool):
    cx, cy = w / 2.0, h / 2.0
    result = search_focal(view_name, view_data, field_tags, cx, cy, f_min, f_max, coarse_step)
    if result is None or not fit_principal_point:
        if result is not None:
            result["cx"], result["cy"], result["k1"] = cx, cy, 0.0
        return result

    # Coordinate descent: alternate refining (cx, cy) with f fixed and
    # refining f with (cx, cy) fixed. Converges fast in practice (this
    # project's own "main" view settles within 2 rounds) since the two
    # are only loosely coupled for a roughly-frontal view.
    f = result["f"]
    for _ in range(3):
        cost_before = _cost(f, view_name, view_data, field_tags, cx, cy)
        _, cx_new, cy_new = search_principal_point(view_name, view_data, field_tags,
                                                    f, cx, cy, w, h)
        refit = search_focal(view_name, view_data, field_tags, cx_new, cy_new,
                             f_min, f_max, coarse_step)
        if refit is None:
            break
        f, cost_after = refit["f"], refit["rms"]
        cx, cy = cx_new, cy_new
        result = refit
        if cost_before - cost_after < 0.05:   # converged
            break

    k1 = 0.0
    if fit_distortion:
        # Same coordinate-descent pattern, one rung further: alternate
        # refining k1 (f/cx/cy fixed) and refitting f+cx+cy (k1 fixed).
        # See MIN_POINTS_FOR_DISTORTION for why this rung is trusted at all.
        for _ in range(3):
            cost_before = _cost(f, view_name, view_data, field_tags, cx, cy, k1)
            k1_new, _ = search_k1(view_name, view_data, field_tags, f, cx, cy)
            _, cx_new, cy_new = search_principal_point(view_name, view_data, field_tags,
                                                        f, cx, cy, w, h, k1=k1_new)
            refit = search_focal(view_name, view_data, field_tags, cx_new, cy_new,
                                 f_min, f_max, coarse_step, k1=k1_new)
            if refit is None:
                break
            f, cost_after = refit["f"], refit["rms"]
            cx, cy, k1 = cx_new, cy_new, k1_new
            result = refit
            if cost_before - cost_after < 0.05:
                break

    result["cx"], result["cy"], result["k1"] = cx, cy, k1
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, metavar="PATH")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--view", metavar="NAME",
                    help="search only this view (default: all views with usable tags)")
    ap.add_argument("--tags", metavar="PATH")
    ap.add_argument("--f-min", type=float, default=DEFAULT_F_MIN)
    ap.add_argument("--f-max", type=float, default=DEFAULT_F_MAX)
    ap.add_argument("--coarse-step", type=float, default=COARSE_STEP)
    ap.add_argument("--out", metavar="PATH",
                    help="output path (default: data/calibration/<stem>_intrinsics.json, merged)")
    ap.add_argument("--no-principal-point", action="store_true",
                    help="keep cx,cy fixed at the crop centre even on views with enough "
                         f"points (>= {MIN_POINTS_FOR_PRINCIPAL_POINT}) to fit them")
    ap.add_argument("--no-distortion", action="store_true",
                    help="keep k1 fixed at 0 even on views with enough points "
                         f"(>= {MIN_POINTS_FOR_DISTORTION}) to fit it")
    args = ap.parse_args()

    field_path = FIELD_DIR / f"{args.year}_tags.json"
    if not field_path.exists():
        sys.exit(f"[error] field layout not found: {field_path}")
    field_tags = json.loads(field_path.read_text())["tags"]

    stem = pathlib.Path(args.video).stem
    tags_path = pathlib.Path(args.tags) if args.tags else DETECTIONS_DIR / f"{stem}_tags.json"
    if not tags_path.exists():
        sys.exit(f"[error] tags not found: {tags_path}\n"
                 f"        run pipeline/01_detect_tags.py --video first")
    tags_data = json.loads(tags_path.read_text())

    views = tags_data.get("views", {})
    view_names = [args.view] if args.view else sorted(views)
    missing = [v for v in view_names if v not in views]
    if missing:
        sys.exit(f"[error] view(s) {missing} not in {tags_path}")

    out_path = pathlib.Path(args.out) if args.out else CALIB_DIR / f"{stem}_intrinsics.json"
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}

    for vname in view_names:
        vdata = views[vname]
        x0, y0, x1, y1 = vdata["box"]
        w, h = x1 - x0, y1 - y0
        n_tags = len(vdata.get("decoded_tags", {}))

        # Probe point count at a placeholder f to decide whether this view
        # has enough correspondences to also fit cx,cy (see
        # MIN_POINTS_FOR_PRINCIPAL_POINT) -- independent of what f turns
        # out to be, since point count only depends on how many tags clear
        # HEAD_ON_SQUARENESS, not on the intrinsics being tested.
        with redirect_stderr(io.StringIO()):
            probe = solve_pose.solve_view(vname, vdata, field_tags,
                                          np.array([[1000.0, 0, w/2], [0, 1000.0, h/2], [0, 0, 1]]),
                                          np.zeros(5), False)
        n_points = probe["n_points_used"] if probe else 0
        fit_pp = n_points >= MIN_POINTS_FOR_PRINCIPAL_POINT and not args.no_principal_point
        fit_k1 = n_points >= MIN_POINTS_FOR_DISTORTION and not args.no_distortion

        print(f"[{vname}] {n_tags} decoded tag(s), {n_points} correspondence point(s), "
              f"searching f in [{args.f_min:.0f}, {args.f_max:.0f}]px"
              f"{' + principal point' if fit_pp else ''}"
              f"{' + k1' if fit_k1 else ''} ...", file=sys.stderr)

        result = search_intrinsics(vname, vdata, field_tags, w, h,
                                   args.f_min, args.f_max, args.coarse_step, fit_pp, fit_k1)
        if result is None:
            print(f"  [{vname}] no f in range produced a valid pose solve -- skipping",
                  file=sys.stderr)
            continue

        flat = result["flatness_ratio"]
        flat_note = ("well-constrained" if flat is not None and flat > 0.2
                    else "POORLY CONSTRAINED -- cost barely changes near this minimum, "
                          "don't trust this f" if flat is not None else "unknown")
        cx, cy, k1 = result["cx"], result["cy"], result["k1"]
        pp_note = (f"  cx={cx:.1f} (Δ{cx - w/2:+.1f})  cy={cy:.1f} (Δ{cy - h/2:+.1f})"
                  if fit_pp else "")
        k1_note = f"  k1={k1:+.4f}" if fit_k1 else ""
        print(f"  [{vname}] f={result['f']:.1f}px  rms_all_pts={result['rms']:.2f}px  "
              f"({flat_note}{f', ratio={flat:.2f}' if flat is not None else ''}){pp_note}{k1_note}",
              file=sys.stderr)

        existing[vname] = {
            "K": [[result["f"], 0.0, cx], [0.0, result["f"], cy], [0.0, 0.0, 1.0]],
            "dist": [k1, 0.0, 0.0, 0.0, 0.0],
            "image_size": [w, h],
            "focal_px": round(result["f"], 1),
            "cx": round(cx, 1), "cy": round(cy, 1),
            "k1": round(k1, 5),
            # all-correspondences RMS, deliberately NOT RANSAC inlier-only --
            # see _cost()'s docstring comment for why that distinction matters.
            "rms_reproj_all_px": round(result["rms"], 3),
            "flatness_ratio": round(flat, 3) if flat is not None else None,
            "fit": ("focal_pp_k1_search" if fit_k1 else
                    "focal_and_pp_search" if fit_pp else "focal_search_1d"),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"[out] {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

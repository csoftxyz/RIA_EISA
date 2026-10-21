#!/usr/bin/env python3
"""
nufit_parser.py -- Parser and interpolators for the official NuFIT 5.2
chi-squared release tables.

The official table file is
    https://www.nu-fit.org/sites/default/files/v52.release-SKyes-NO.txt.xz
(normal ordering, with Super-K atmospheric data).  Decompress with
`xz -dk v52.release-SKyes-NO.txt.xz` and point NUFiT_TABLE at the
resulting file (or set the environment variable).

Contents of the table (23 sections):
  * 1  three-dimensional projection  (sin2 t23, dm31/1e-3, delta)
  * 15 two-dimensional projections
  * 6  one-dimensional projections

All Delta-chi^2 values are with respect to the global best fit
(Delta chi^2 = 0 at the best-fit point).

Author: Yuxuan Zhang  (paper code, v1.0)
"""
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator

DEFAULT_TABLE = "/root/.openclaw/workspace/nufit_official/v52-NO"


def _load_sections(path):
    sections = []
    cur = None
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                if cur is not None and cur['rows']:
                    sections.append(cur)
                cur = {'name': line.strip('#').strip(), 'rows': []}
            else:
                parts = line.split()
                if (cur is not None and parts
                        and parts[0].replace('.', '').replace('-', '')
                                 .replace('+', '').isdigit()):
                    cur['rows'].append([float(x) for x in parts])
        if cur is not None and cur['rows']:
            sections.append(cur)
    return sections


class NuFIT52:
    """Parsed official NuFIT 5.2 tables (NO, with SK)."""

    def __init__(self, path=DEFAULT_TABLE):
        self.path = path
        self.sections = _load_sections(path)
        self._cache = {}

    # -- helpers ------------------------------------------------------
    def _find(self, name_prefix):
        target = name_prefix + ':'
        for s in self.sections:
            if s['name'].startswith(target):
                return np.array(s['rows'])
        raise KeyError(f"no section starting with {target!r}")

    # -- 1D projections -----------------------------------------------
    def proj1d(self, name_prefix):
        """Return (x, Delta_chi2) for a 1D projection, e.g.
        "T13 projection", "DCP projection", "DMA projection"."""
        key = ('1d', name_prefix)
        if key not in self._cache:
            r = self._find(name_prefix)
            assert r.shape[1] == 2
            self._cache[key] = (r[:, 0], r[:, 1])
        return self._cache[key]

    def chi2_1d(self, name_prefix, x):
        """Interpolate Delta chi2 at x in a 1D projection.
        Returns 1e6 outside the grid."""
        arr, c = self.proj1d(name_prefix)
        if np.isnan(x) or x < arr.min() or x > arr.max():
            return 1e6
        return float(np.interp(x, arr, c))

    # -- 2D projections -----------------------------------------------
    def proj2d(self, name_prefix):
        """Return (x, y, Z) on a regular grid for a 2D projection,
        e.g. "T13/T12 projection", "T13/DMS projection"."""
        key = ('2d', name_prefix)
        if key not in self._cache:
            r = self._find(name_prefix)
            x = np.unique(r[:, 0])
            y = np.unique(r[:, 1])
            Z = np.full((len(x), len(y)), np.nan)
            for row in r:
                xi = np.searchsorted(x, row[0])
                yi = np.searchsorted(y, row[1])
                Z[xi, yi] = row[2]
            assert not np.isnan(Z).any(), f"NaN in {name_prefix}"
            self._cache[key] = (x, y, Z)
        return self._cache[key]

    # -- 3D projection ------------------------------------------------
    def proj3d(self, name_prefix="T23/DMA/DCP projection"):
        """Return (x, y, z, Z) for the 3D projection."""
        key = ('3d', name_prefix)
        if key not in self._cache:
            r = self._find(name_prefix)
            x = np.unique(r[:, 0])
            y = np.unique(r[:, 1])
            z = np.unique(r[:, 2])
            Z = np.full((len(x), len(y), len(z)), np.nan)
            for row in r:
                xi = np.searchsorted(x, row[0])
                yi = np.searchsorted(y, row[1])
                zi = np.searchsorted(z, row[2])
                Z[xi, yi, zi] = row[3]
            assert not np.isnan(Z).any()
            self._cache[key] = (x, y, z, Z)
        return self._cache[key]

    # -- composed likelihoods -----------------------------------------
    def chi2_1d_sum(self, y):
        """Sum of the six 1D projections at the observable vector
        y = (s12, s13, s23, delta_deg, dm21, dm31).
        This is the standard model-evaluation procedure recommended
        for the release tables and is used as the fit objective."""
        s12, s13, s23, dc, dm21, dm31 = y
        return (self.chi2_1d("T12 projection", s12)
                + self.chi2_1d("T13 projection", s13)
                + self.chi2_1d("T23 projection", s23)
                + self.chi2_1d("DCP projection", dc)
                + self.chi2_1d("DMS projection", np.log10(dm21))
                + self.chi2_1d("DMA projection", dm31 / 1e-3))

    def chi2_block(self, y):
        """3D + 2D combination at y (cross-check, marginal lower
        bound): (s23,dm31,delta) + (s13,s12) + (s13,dms) + (s12,dms)."""
        s12, s13, s23, dc, dm21, dm31 = y
        x23, xdma, xdcp, Z3 = self.proj3d()
        interp3 = RegularGridInterpolator(
            (x23, xdma, xdcp), Z3, bounds_error=False, fill_value=None)
        c3 = float(interp3([[s23, dm31 / 1e-3, dc]]))
        x13a, x12a, Z1312 = self.proj2d("T13/T12 projection")
        x13b, xdmsb, Z13dms = self.proj2d("T13/DMS projection")
        x12c, xdmsc, Z12dms = self.proj2d("T12/DMS projection")
        i1312 = RegularGridInterpolator(
            (x13a, x12a), Z1312, bounds_error=False, fill_value=None)
        i13dms = RegularGridInterpolator(
            (x13b, xdmsb), Z13dms, bounds_error=False, fill_value=None)
        i12dms = RegularGridInterpolator(
            (x12c, xdmsc), Z12dms, bounds_error=False, fill_value=None)
        return (c3 + float(i1312([[s13, s12]]))
                + float(i13dms([[s13, np.log10(dm21)]]))
                + float(i12dms([[s12, np.log10(dm21)]])))

    # -- best-fit values from 1D projections --------------------------
    def best_fit_1d(self):
        """Central values and 1-sigma widths from the official 1D
        projections (used for the pull table)."""
        out = {}
        for name, key in [("T13 projection", "s13"),
                          ("T12 projection", "s12"),
                          ("T23 projection", "s23"),
                          ("DCP projection", "delta"),
                          ("DMS projection", "dms"),
                          ("DMA projection", "dma")]:
            x, c = self.proj1d(name)
            imin = np.argmin(c)
            mu = x[imin]
            inside = x[c <= 1.0]
            sig = (inside.max() - inside.min()) / 2.0
            out[key] = (mu, sig)
        return out


def table_available(path=DEFAULT_TABLE):
    return os.path.exists(path)


if __name__ == "__main__":
    if not table_available():
        print("NuFIT table not found:", DEFAULT_TABLE)
        raise SystemExit(1)
    nf = NuFIT52()
    print("=== Official NuFIT 5.2 best-fit values (1D projections) ===")
    bf = nf.best_fit_1d()
    for k, (mu, sg) in bf.items():
        print(f"  {k:6s}: mu={mu:.5f}  sigma={sg:.5f}")
    print("\n=== texture best-fit point (paper Sec. 4) ===")
    from z3lm import M6, obs_vec, BEST_FIT
    y = obs_vec(M6(BEST_FIT))
    print("  chi2_1d_sum =", nf.chi2_1d_sum(y))
    print("  chi2_block  =", nf.chi2_block(y))

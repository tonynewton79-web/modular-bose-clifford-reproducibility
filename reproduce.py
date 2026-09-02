"""Clean-room numerical reproducer for the manuscript
'Modular Bose--Clifford Fibre Matchings'.

This module implements only formulas and finite enumeration procedures stated
in the manuscript. It contains no research-search, theorem-discovery, prompt,
or private solver machinery.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import math
from typing import Iterable, Iterator, Sequence
from pathlib import Path

import numpy as np
from numba import njit
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

PI = math.pi
A_CELL = math.sqrt(math.pi)


@dataclass(frozen=True)
class Parameters:
    r: float
    omega0: float
    nu_a: float
    nu_b: float
    kappa: float
    D: float
    delta: float
    gamma: float

    @property
    def Q(self) -> np.ndarray:
        return 0.5 * np.array([[self.nu_a, self.kappa],
                               [self.kappa, self.nu_b]], dtype=float)

    @property
    def P(self) -> np.ndarray:
        return 0.5 * np.array([[self.nu_a, -self.kappa],
                               [-self.kappa, self.nu_b]], dtype=float)


@dataclass(frozen=True)
class Block:
    """One cell block in midpoint/displacement variables.

    kind='T': c is integer midpoint index, l is positive odd translation length.
    kind='R': c is half-integer reflection midpoint h, l is even displacement label e.
    """
    kind: str
    c: float
    l: int


def modular_parameters(r: float = 0.1, omega0: float = 0.01) -> Parameters:
    lam2 = math.exp(-2.0 * PI * omega0)
    d = r * omega0
    if d == 0.0:
        s_d = 1.0
    else:
        s_d = math.sinh(2.0 * PI * d) / (2.0 * PI * d)
    mu_plus = lam2 * s_d
    mu_minus = s_d / lam2
    nu_a = 2.0 / (1.0 - mu_plus) - 1.0
    nu_b = 2.0 / (mu_minus - 1.0) + 1.0
    kappa = 2.0 / math.sqrt((1.0 - mu_plus) * (mu_minus - 1.0))
    D = nu_a * nu_b - kappa * kappa
    delta = nu_a + nu_b - 2.0 * kappa
    gamma = D / delta
    return Parameters(r, omega0, nu_a, nu_b, kappa, D, delta, gamma)


def cosine_correlation(par: Parameters) -> float:
    return 0.5 * (
        math.exp(-PI * par.delta / 4.0)
        + math.exp(-PI * (par.nu_a + par.nu_b + 2.0 * par.kappa) / 4.0)
    )


def global_reflection_correlation(par: Parameters, k: int = 0) -> float:
    return (1.0 / par.D) * math.exp(
        -PI * par.delta * (2 * k + 1) ** 2 / (4.0 * par.D)
    )


def global_reflection_bell(par: Parameters, k: int = 0) -> float:
    return math.sqrt(2.0) * (
        cosine_correlation(par) + global_reflection_correlation(par, k)
    )


def reflection_threshold(r: float = 0.1) -> float:
    f = lambda w: global_reflection_bell(modular_parameters(r, w)) - 2.0
    return brentq(f, 1e-5, 0.08, xtol=2e-14, rtol=2e-14)


def j_interval(c: float) -> tuple[float, float]:
    return (c - 0.5) * A_CELL, (c + 0.5) * A_CELL


def rectangle_probability(cov: np.ndarray,
                          xlo: float, xhi: float,
                          ylo: float, yhi: float,
                          eps: float = 2e-12) -> float:
    """Deterministic 1D conditional-normal quadrature of a 2D Gaussian rectangle."""
    sx = math.sqrt(float(cov[0, 0]))
    cond_var = float(cov[1, 1] - cov[0, 1] ** 2 / cov[0, 0])
    sy = math.sqrt(max(cond_var, 0.0))
    beta = float(cov[1, 0] / cov[0, 0])

    def integrand(x: float) -> float:
        pdf = math.exp(-0.5 * (x / sx) ** 2) / (math.sqrt(2.0 * PI) * sx)
        mu = beta * x
        return pdf * (
            norm.cdf((yhi - mu) / sy) - norm.cdf((ylo - mu) / sy)
        )

    val, _ = quad(integrand, xlo, xhi, epsabs=eps, epsrel=eps, limit=120)
    return float(val)


def W_lm(par: Parameters, ell: int, m: int) -> float:
    minus = par.nu_a * ell * ell + par.nu_b * m * m - 2.0 * par.kappa * ell * m
    plus = par.nu_a * ell * ell + par.nu_b * m * m + 2.0 * par.kappa * ell * m
    return 2.0 * math.exp(-PI * minus / 4.0) + 2.0 * math.exp(-PI * plus / 4.0)


def block_from_cells(i: int, j: int) -> Block:
    if i > j:
        i, j = j, i
    if i == j or ((j - i) % 2 == 0):
        return Block("R", (i + j + 1) / 2.0, abs(j - i))
    return Block("T", (i + j + 1) / 2.0, abs(j - i))


def translation_blocks_from_matching(pairs: Sequence[tuple[int, int]]) -> list[Block]:
    out = []
    for i, j in pairs:
        b = block_from_cells(i, j)
        if b.kind != "T":
            raise ValueError("translation matching contains same-parity/reflection pair")
        out.append(b)
    return out


def exterior_translation_blocks(N: int, max_midpoint: int = 31) -> list[Block]:
    start = N + 1
    out: list[Block] = []
    for c in range(start, max_midpoint + 1, 2):
        out.extend([Block("T", -c, 1), Block("T", c, 1)])
    return out


class KernelEvaluator:
    """Exact finite-band block kernel with deterministic quadrature caches."""
    def __init__(self, par: Parameters, eps: float = 2e-12):
        self.par = par
        self.Q = par.Q
        self.P = par.P
        self.Qinv = np.linalg.inv(self.Q)
        self.Pinv = np.linalg.inv(self.P)
        self.Qdet = float(np.linalg.det(self.Q))
        self.Pdet = float(np.linalg.det(self.P))
        self.eps = eps
        self._mu_cache: dict[tuple, float] = {}
        self._eta_cache: dict[tuple, float] = {}
        self._rect_q: dict[tuple[float, float], float] = {}
        self._rect_pinv: dict[tuple[int, int, int, int], float] = {}

    def _q_rect(self, c: float, d: float) -> float:
        key = (c, d)
        if key not in self._rect_q:
            self._rect_q[key] = rectangle_probability(
                self.Q, *j_interval(c), *j_interval(d), eps=self.eps
            )
        return self._rect_q[key]

    @staticmethod
    def _de_intervals(e: int) -> tuple[tuple[float, float], ...]:
        if e == 0:
            return ((-A_CELL, A_CELL),)
        return (
            ((e - 1) * A_CELL, (e + 1) * A_CELL),
            (-(e + 1) * A_CELL, -(e - 1) * A_CELL),
        )

    def mu(self, alpha: Block, beta: Block) -> float:
        key = (alpha.kind, alpha.c, alpha.l, beta.kind, beta.c, beta.l)
        if key in self._mu_cache:
            return self._mu_cache[key]

        if alpha.kind == "T" and beta.kind == "T":
            value = self._q_rect(alpha.c, beta.c)
        elif alpha.kind == "T" and beta.kind == "R":
            xlo, xhi = j_interval(alpha.c)
            y = beta.c * A_CELL
            sy = math.sqrt(float(self.Q[1, 1]))
            dens_y = math.exp(-0.5 * (y / sy) ** 2) / (math.sqrt(2.0 * PI) * sy)
            mux = float(self.Q[0, 1] / self.Q[1, 1]) * y
            sx = math.sqrt(float(self.Q[0, 0] - self.Q[0, 1] ** 2 / self.Q[1, 1]))
            value = dens_y * (norm.cdf((xhi - mux) / sx) - norm.cdf((xlo - mux) / sx))
        elif alpha.kind == "R" and beta.kind == "T":
            x = alpha.c * A_CELL
            ylo, yhi = j_interval(beta.c)
            sx = math.sqrt(float(self.Q[0, 0]))
            dens_x = math.exp(-0.5 * (x / sx) ** 2) / (math.sqrt(2.0 * PI) * sx)
            muy = float(self.Q[1, 0] / self.Q[0, 0]) * x
            sy = math.sqrt(float(self.Q[1, 1] - self.Q[1, 0] ** 2 / self.Q[0, 0]))
            value = dens_x * (norm.cdf((yhi - muy) / sy) - norm.cdf((ylo - muy) / sy))
        else:
            x = alpha.c * A_CELL
            y = beta.c * A_CELL
            v = np.array([x, y], dtype=float)
            value = math.exp(-0.5 * float(v @ self.Qinv @ v)) / (2.0 * PI * math.sqrt(self.Qdet))

        self._mu_cache[key] = float(value)
        return float(value)

    def _int_exp_y_given_x(self, x: float, ylo: float, yhi: float) -> float:
        A, B, C = float(self.P[0, 0]), float(self.P[1, 1]), float(self.P[0, 1])
        pref = math.exp(-0.5 * (A - C * C / B) * x * x)
        shift = C * x / B
        return pref * math.sqrt(2.0 * PI / B) * (
            norm.cdf(math.sqrt(B) * (yhi + shift))
            - norm.cdf(math.sqrt(B) * (ylo + shift))
        )

    def _int_exp_x_given_y(self, y: float, xlo: float, xhi: float) -> float:
        A, B, C = float(self.P[0, 0]), float(self.P[1, 1]), float(self.P[0, 1])
        pref = math.exp(-0.5 * (B - C * C / A) * y * y)
        shift = C * y / A
        return pref * math.sqrt(2.0 * PI / A) * (
            norm.cdf(math.sqrt(A) * (xhi + shift))
            - norm.cdf(math.sqrt(A) * (xlo + shift))
        )

    def eta(self, alpha: Block, beta: Block) -> float:
        # eta depends on translation/reflection displacement labels, not midpoint locations.
        key = (alpha.kind, alpha.l, beta.kind, beta.l)
        if key in self._eta_cache:
            return self._eta_cache[key]

        if alpha.kind == "T" and beta.kind == "T":
            total = 0.0
            for x in (alpha.l * A_CELL, -alpha.l * A_CELL):
                for y in (beta.l * A_CELL, -beta.l * A_CELL):
                    total += math.exp(-0.5 * (
                        self.P[0, 0] * x * x + 2.0 * self.P[0, 1] * x * y + self.P[1, 1] * y * y
                    ))
            value = total
        elif alpha.kind == "T" and beta.kind == "R":
            total = 0.0
            for x in (alpha.l * A_CELL, -alpha.l * A_CELL):
                for ylo, yhi in self._de_intervals(beta.l):
                    total += 0.5 * self._int_exp_y_given_x(x, ylo, yhi)
            value = total
        elif alpha.kind == "R" and beta.kind == "T":
            total = 0.0
            for y in (beta.l * A_CELL, -beta.l * A_CELL):
                for xlo, xhi in self._de_intervals(alpha.l):
                    total += 0.5 * self._int_exp_x_given_y(y, xlo, xhi)
            value = total
        else:
            total_prob = 0.0
            for xlo, xhi in self._de_intervals(alpha.l):
                for ylo, yhi in self._de_intervals(beta.l):
                    total_prob += rectangle_probability(
                        self.Pinv, xlo, xhi, ylo, yhi, eps=self.eps
                    )
            value = 0.25 * (2.0 * PI / math.sqrt(self.Pdet)) * total_prob

        self._eta_cache[key] = float(value)
        return float(value)

    def K(self, alpha: Block, beta: Block) -> float:
        return self.mu(alpha, beta) * self.eta(alpha, beta)

    def translation_K(self, alpha: Block, beta: Block) -> float:
        if alpha.kind != "T" or beta.kind != "T":
            raise ValueError("translation_K requires translation blocks")
        return self._q_rect(alpha.c, beta.c) * W_lm(self.par, alpha.l, beta.l)


def score_blocks(blocks: Sequence[Block], kernel: KernelEvaluator) -> float:
    return sum(kernel.K(a, b) for a in blocks for b in blocks)


def score_translation_blocks(blocks: Sequence[Block], kernel: KernelEvaluator) -> float:
    return sum(kernel.translation_K(a, b) for a in blocks for b in blocks)


@njit(cache=True)
def _best_permutation_score(core_k: np.ndarray, ext_lin: np.ndarray, ext_const: float, n: int):
    perm = np.arange(n, dtype=np.int64)
    best = -1.0
    bestp = perm.copy()
    count = 0
    ids = np.empty(n, dtype=np.int64)
    while True:
        s = ext_const
        for a in range(n):
            ids[a] = a * n + perm[a]
            s += ext_lin[ids[a]]
        for a in range(n):
            ia = ids[a]
            for b in range(n):
                s += core_k[ia, ids[b]]
        if s > best:
            best = s
            bestp = perm.copy()
        count += 1

        i = n - 2
        while i >= 0 and perm[i] >= perm[i + 1]:
            i -= 1
        if i < 0:
            break
        j = n - 1
        while perm[j] <= perm[i]:
            j -= 1
        tmp = perm[i]
        perm[i] = perm[j]
        perm[j] = tmp
        lo, hi = i + 1, n - 1
        while lo < hi:
            tmp = perm[lo]
            perm[lo] = perm[hi]
            perm[hi] = tmp
            lo += 1
            hi -= 1
    return best, bestp, count


def exhaustive_translation_window(N: int, r: float, omega0: float,
                                  max_midpoint: int = 31):
    """Exhaust every opposite-parity perfect matching in W_N={-N,...,N-1}."""
    par = modular_parameters(r, omega0)
    ke = KernelEvaluator(par)
    cells = list(range(-N, N))
    evens = [i for i in cells if i % 2 == 0]
    odds = [i for i in cells if i % 2 != 0]
    n = len(evens)
    edge_blocks = [block_from_cells(e, o) for e in evens for o in odds]
    ext = exterior_translation_blocks(N, max_midpoint=max_midpoint)

    core_k = np.array([[ke.translation_K(a, b) for b in edge_blocks] for a in edge_blocks], dtype=float)
    ext_const = float(sum(ke.translation_K(a, b) for a in ext for b in ext))
    ext_lin = np.array([
        sum(ke.translation_K(a, b) for b in ext) + sum(ke.translation_K(b, a) for b in ext)
        for a in edge_blocks
    ], dtype=float)

    best, perm, count = _best_permutation_score(core_k, ext_lin, ext_const, n)
    pairs = [(evens[a], odds[int(perm[a])]) for a in range(n)]
    blocks = [block_from_cells(i, j) for i, j in pairs]
    bell = math.sqrt(2.0) * (cosine_correlation(par) + best)
    return {
        "C_gamma": float(best),
        "Bell": float(bell),
        "count": int(count),
        "pairs": pairs,
        "features": [(b.c, b.l) for b in blocks],
    }


def _involutions(items: tuple[int, ...]) -> Iterator[list[tuple[int, int]]]:
    if not items:
        yield []
        return
    i = items[0]
    rest = items[1:]
    for inv in _involutions(rest):
        yield [(i, i)] + inv
    for pos, j in enumerate(rest):
        rem = rest[:pos] + rest[pos + 1:]
        for inv in _involutions(rem):
            yield [(i, j)] + inv


def mixed_12cell_exhaustive(r: float = 0.1, omega0: float = 0.01,
                            max_midpoint: int = 31):
    """Exhaust all I_12=140,152 cellwise involutions with adjacent exterior."""
    par = modular_parameters(r, omega0)
    ke = KernelEvaluator(par)
    cells = list(range(-6, 6))
    all_pairs = [(i, j) for pos, i in enumerate(cells) for j in cells[pos:]]
    central_blocks = [block_from_cells(i, j) for i, j in all_pairs]
    pair_to_idx = {p: idx for idx, p in enumerate(all_pairs)}
    ext = exterior_translation_blocks(6, max_midpoint=max_midpoint)
    blocks = central_blocks + ext

    Kmat = np.array([[ke.K(a, b) for b in blocks] for a in blocks], dtype=float)
    ncentral = len(central_blocks)
    ext_idx = np.arange(ncentral, len(blocks), dtype=int)
    ext_const = float(Kmat[np.ix_(ext_idx, ext_idx)].sum())
    ext_lin = Kmat[:ncentral, :][:, ext_idx].sum(axis=1) + Kmat[ext_idx, :ncentral].sum(axis=0)

    best = -1.0
    best_inv: list[tuple[int, int]] | None = None
    count = 0
    for inv in _involutions(tuple(cells)):
        idx = np.array([pair_to_idx[(i, j)] for i, j in inv], dtype=int)
        s = ext_const + float(ext_lin[idx].sum()) + float(Kmat[np.ix_(idx, idx)].sum())
        count += 1
        if s > best:
            best = s
            best_inv = inv

    bell = math.sqrt(2.0) * (cosine_correlation(par) + best)
    return {
        "C_gamma": float(best),
        "Bell": float(bell),
        "count": int(count),
        "involution": best_inv,
        "features": [block_from_cells(i, j) for i, j in (best_inv or [])],
    }


def global_reflection_block_sum(r: float = 0.1, omega0: float = 0.01,
                                nmax: int = 20) -> float:
    par = modular_parameters(r, omega0)
    ke = KernelEvaluator(par)
    blocks = [Block("R", 0.5, 0)] + [Block("R", 0.5, 2 * i) for i in range(1, nmax + 1)]
    return score_blocks(blocks, ke)


def isolated_shell_gain(r: float, omega0: float, lengths: Sequence[int]) -> float:
    par = modular_parameters(r, omega0)
    ke = KernelEvaluator(par)
    shell_midpoints = (-1, 0, 1)
    L = lengths[len(lengths) // 2]
    p_shell = sum(ke._q_rect(c, d) for c in shell_midpoints for d in shell_midpoints)
    p00 = ke._q_rect(0, 0)
    return W_lm(par, L, L) * p_shell - p00 * sum(W_lm(par, l, m) for l in lengths for m in lengths)


def shell_transition_root(lengths: Sequence[int], r: float = 0.1) -> float:
    if tuple(lengths) == (1, 3, 5):
        bracket = (0.01, 0.02)
    elif tuple(lengths) == (7, 9, 11):
        bracket = (0.001, 0.005)
    else:
        raise ValueError("Supply a bracket for non-reference shell cohorts by calling brentq on isolated_shell_gain.")
    return brentq(lambda w: isolated_shell_gain(r, w, lengths), *bracket, xtol=2e-14, rtol=2e-14)


def adjacent_infinite_blocks(max_midpoint: int = 31) -> list[Block]:
    return [Block("T", c, 1) for c in range(-max_midpoint, max_midpoint + 1, 2)]


def first_shell_phase_blocks(max_midpoint: int = 31) -> list[Block]:
    blocks = [Block("T", -1, 3), Block("T", 0, 3), Block("T", 1, 3)]
    blocks.extend(Block("T", 0, l) for l in (7, 9, 11, 13, 15, 17, 19))
    blocks.extend(exterior_translation_blocks(10, max_midpoint=max_midpoint))
    return blocks


def bell_for_translation_blocks(r: float, omega0: float, blocks: Sequence[Block]) -> float:
    par = modular_parameters(r, omega0)
    ke = KernelEvaluator(par)
    return math.sqrt(2.0) * (cosine_correlation(par) + score_translation_blocks(blocks, ke))


def adjacent_bell_threshold(r: float = 0.1) -> float:
    blocks = adjacent_infinite_blocks(31)
    return brentq(lambda w: bell_for_translation_blocks(r, w, blocks) - 2.0,
                  0.005, 0.012, xtol=2e-14, rtol=2e-14)


def shell_bell_threshold(r: float = 0.1) -> float:
    blocks = first_shell_phase_blocks(31)
    return brentq(lambda w: bell_for_translation_blocks(r, w, blocks) - 2.0,
                  0.01, 0.014, xtol=2e-14, rtol=2e-14)


def tail_certificate_double(r: float = 0.1, omega0: float = 0.01):
    """Double-precision replay of the manuscript's positive-tail certificate."""
    full = exhaustive_translation_window(10, r, omega0, max_midpoint=31)
    central = [Block("T", c, l) for c, l in full["features"]]
    retained_ext = [Block("T", c, 1) for c in (-15, -13, -11, 11, 13, 15)]
    retained = central + retained_ext
    par = modular_parameters(r, omega0)
    ke = KernelEvaluator(par)
    c_gamma_15 = score_translation_blocks(retained, ke)
    cx = cosine_correlation(par)
    b15 = math.sqrt(2.0) * (cx + c_gamma_15)

    z_a = 16.5 * A_CELL / math.sqrt(par.Q[0, 0])
    z_b = 16.5 * A_CELL / math.sqrt(par.Q[1, 1])
    mills = lambda z: math.sqrt(2.0 / PI) * math.exp(-z * z / 2.0) / z
    m_a, m_b = mills(z_a), mills(z_b)
    epsilon = 4.0 * math.exp(-PI * par.gamma)
    w11 = W_lm(par, 1, 1)
    bell_budget = math.sqrt(2.0) * (w11 + 10.0 * epsilon) * (m_a + m_b)
    return {
        "C_X": cx,
        "C_gamma_15": c_gamma_15,
        "B_15": b15,
        "z_A": z_a,
        "z_B": z_b,
        "m_A_upper": m_a,
        "m_B_upper": m_b,
        "Bell_tail_upper": bell_budget,
    }


# -----------------------------------------------------------------------------
# Command-line reproduction interface
# -----------------------------------------------------------------------------

def _quick_results():
    p = modular_parameters(0.1, 0.01)
    return {
        "parameters": {
            "nu_A": p.nu_a,
            "nu_B": p.nu_b,
            "kappa": p.kappa,
            "D": p.D,
            "delta": p.delta,
            "gamma": p.gamma,
        },
        "C_X": cosine_correlation(p),
        "C_R": global_reflection_correlation(p),
        "B_R": global_reflection_bell(p),
        "reflection_block_sum_n20": global_reflection_block_sum(0.1, 0.01, 20),
    }


def _root_results():
    out = _quick_results()
    out.update({
        "reflection_threshold": reflection_threshold(0.1),
        "shell_root_1_3_5": shell_transition_root((1, 3, 5), 0.1),
        "shell_root_7_9_11": shell_transition_root((7, 9, 11), 0.1),
        "adjacent_threshold": adjacent_bell_threshold(0.1),
        "shell_threshold": shell_bell_threshold(0.1),
    })
    out["bandwidth_extension_percent"] = 100.0 * (
        out["shell_threshold"] / out["adjacent_threshold"] - 1.0
    )
    return out


def _full_results():
    out = _root_results()
    t20 = exhaustive_translation_window(10, 0.1, 0.01, 31)
    mix = mixed_12cell_exhaustive(0.1, 0.01, 31)
    tail = tail_certificate_double(0.1, 0.01)
    out["translation_20cell"] = t20
    out["mixed_12cell"] = {
        "C_gamma": mix["C_gamma"],
        "Bell": mix["Bell"],
        "count": mix["count"],
        "involution": mix["involution"],
        "features": [(b.kind, b.c, b.l) for b in mix["features"]],
    }
    out["tail_certificate"] = tail
    return out


def _main():
    import argparse
    import json
    parser = argparse.ArgumentParser(
        description="Reproduce the numerical values in Modular Bose--Clifford Fibre Matchings."
    )
    parser.add_argument("--roots", action="store_true",
                        help="also reproduce shell and Bell-threshold roots")
    parser.add_argument("--full", action="store_true",
                        help="also run both exhaustive finite-window searches")
    parser.add_argument("--json", default=None,
                        help="optional path for JSON output")
    args = parser.parse_args()
    results = _full_results() if args.full else (_root_results() if args.roots else _quick_results())
    text = json.dumps(results, indent=2, default=str)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    _main()

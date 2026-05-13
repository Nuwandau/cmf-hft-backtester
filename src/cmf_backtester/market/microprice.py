from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve

from cmf_backtester.data.loaders import MarketDataArrays
from cmf_backtester.market.features import build_state_id, imbalance_bucket, spread_state


@dataclass(frozen=True)
class MicropriceDiagnostics:
    n_states: int
    n_observations: int
    n_filtered_transitions: int
    min_state_count: int
    median_state_count: float
    max_state_count: int
    iterations: int
    converged: bool
    max_abs_adjustment_ticks: float
    max_abs_last_term_ticks: float


class MicropriceEstimator:
    """Finite-state Stoikov 2018 microprice estimator.

    States are `(spread_state, imbalance_bucket)`. The estimator works in tick units.
    """

    def __init__(
        self,
        n_imbalance_buckets: int = 10,
        max_spread_state_ticks: int = 10,
        max_mid_move_ticks: float = 1.0,
        min_state_count: int = 50,
        max_iterations: int = 1000,
        tolerance: float = 1e-10,
    ) -> None:
        self.n_imbalance_buckets = int(n_imbalance_buckets)
        self.max_spread_state_ticks = int(max_spread_state_ticks)
        self.n_spread_states = self.max_spread_state_ticks + 1
        self.n_states = self.n_imbalance_buckets * self.n_spread_states
        self.max_mid_move_ticks = float(max_mid_move_ticks)
        self.min_state_count = int(min_state_count)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.adjustment_ticks = np.zeros(self.n_states, dtype=np.float64)
        self.g1_ticks = np.zeros(self.n_states, dtype=np.float64)
        self.state_counts = np.zeros(self.n_states, dtype=np.int64)
        self.diagnostics: MicropriceDiagnostics | None = None

    def state_id(self, spread_ticks: int, imbalance: float) -> int:
        return build_state_id(
            spread_ticks,
            imbalance,
            self.n_imbalance_buckets,
            self.max_spread_state_ticks,
        )

    def fit(self, data: MarketDataArrays) -> "MicropriceEstimator":
        if len(data) < 2:
            raise ValueError("Need at least two market snapshots to fit microprice")

        q_counts = np.zeros((self.n_states, self.n_states), dtype=np.float64)
        t_counts = np.zeros((self.n_states, self.n_states), dtype=np.float64)
        r_jump_sum = np.zeros(self.n_states, dtype=np.float64)
        total_counts = np.zeros(self.n_states, dtype=np.float64)
        filtered_transitions = 0

        for i in range(len(data) - 1):
            x = self.state_id(int(data.spread_ticks[i]), float(data.imbalance[i]))
            y = self.state_id(int(data.spread_ticks[i + 1]), float(data.imbalance[i + 1]))
            d_half_ticks = int(data.mid_half_ticks[i + 1] - data.mid_half_ticks[i])
            if abs(d_half_ticks) / 2.0 > self.max_mid_move_ticks:
                filtered_transitions += 1
                continue
            self._add_observation(q_counts, t_counts, r_jump_sum, total_counts, x, y, d_half_ticks)

            # Symmetrization from Stoikov 2018: mirror imbalance and price move.
            x_sym = self._mirror_state(x)
            y_sym = self._mirror_state(y)
            self._add_observation(
                q_counts, t_counts, r_jump_sum, total_counts, x_sym, y_sym, -d_half_ticks
            )

        self.state_counts = total_counts.astype(np.int64)
        q = np.zeros_like(q_counts)
        t = np.zeros_like(t_counts)
        r_vec = np.zeros_like(r_jump_sum)
        nonzero = total_counts > 0
        q[nonzero] = q_counts[nonzero] / total_counts[nonzero, None]
        t[nonzero] = t_counts[nonzero] / total_counts[nonzero, None]
        r_vec[nonzero] = r_jump_sum[nonzero] / total_counts[nonzero]

        eye = np.eye(self.n_states)
        matrix = eye - q
        self.g1_ticks = solve(matrix, r_vec, assume_a="gen")
        b = solve(matrix, t, assume_a="gen")

        adjustment = self.g1_ticks.copy()
        term = self.g1_ticks.copy()
        converged = False
        iterations = 0
        for iterations in range(1, self.max_iterations + 1):
            term = b @ term
            adjustment += term
            if float(np.max(np.abs(term))) <= self.tolerance:
                converged = True
                break

        low_confidence = self.state_counts < self.min_state_count
        adjustment[low_confidence] = 0.0
        self.adjustment_ticks = adjustment
        populated_counts = self.state_counts[self.state_counts > 0]
        self.diagnostics = MicropriceDiagnostics(
            n_states=self.n_states,
            n_observations=int(np.sum(total_counts)),
            n_filtered_transitions=int(filtered_transitions),
            min_state_count=int(np.min(populated_counts)) if populated_counts.size else 0,
            median_state_count=float(np.median(populated_counts)) if populated_counts.size else 0.0,
            max_state_count=int(np.max(populated_counts)) if populated_counts.size else 0,
            iterations=int(iterations),
            converged=converged,
            max_abs_adjustment_ticks=float(np.max(np.abs(self.adjustment_ticks))),
            max_abs_last_term_ticks=float(np.max(np.abs(term))) if term.size else 0.0,
        )
        return self

    @staticmethod
    def _add_observation(
        q_counts: np.ndarray,
        t_counts: np.ndarray,
        r_jump_sum: np.ndarray,
        total_counts: np.ndarray,
        x: int,
        y: int,
        d_half_ticks: int,
    ) -> None:
        total_counts[x] += 1.0
        if d_half_ticks == 0:
            q_counts[x, y] += 1.0
        else:
            t_counts[x, y] += 1.0
            r_jump_sum[x] += d_half_ticks / 2.0

    def _mirror_state(self, state_id: int) -> int:
        spread_idx = state_id // self.n_imbalance_buckets
        imb_idx = state_id % self.n_imbalance_buckets
        mirror_imb_idx = self.n_imbalance_buckets - 1 - imb_idx
        return spread_idx * self.n_imbalance_buckets + mirror_imb_idx

    def predict_adjustment(self, spread_ticks: int, imbalance: float) -> float:
        state = self.state_id(spread_ticks, imbalance)
        return float(self.adjustment_ticks[state])

    def predict_adjustments(self, spread_ticks: np.ndarray, imbalance: np.ndarray) -> np.ndarray:
        out = np.zeros(len(spread_ticks), dtype=np.float64)
        for i in range(len(spread_ticks)):
            out[i] = self.predict_adjustment(int(spread_ticks[i]), float(imbalance[i]))
        return out

    def predict_microprice_ticks(
        self,
        mid_ticks: np.ndarray,
        spread_ticks: np.ndarray,
        imbalance: np.ndarray,
    ) -> np.ndarray:
        return mid_ticks.astype(np.float64) + self.predict_adjustments(spread_ticks, imbalance)

    def adjustment_table(self) -> list[dict[str, float | int]]:
        rows: list[dict[str, float | int]] = []
        for state in range(self.n_states):
            spread_idx = state // self.n_imbalance_buckets
            imb_idx = state % self.n_imbalance_buckets
            spread_label = (
                spread_idx + 1
                if spread_idx < self.max_spread_state_ticks
                else self.max_spread_state_ticks + 1
            )
            rows.append(
                {
                    "state": state,
                    "spread_state": spread_label,
                    "imbalance_bucket": imb_idx + 1,
                    "state_count": int(self.state_counts[state]),
                    "g1_ticks": float(self.g1_ticks[state]),
                    "adjustment_ticks": float(self.adjustment_ticks[state]),
                }
            )
        return rows

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        diag = self.diagnostics
        np.savez_compressed(
            path,
            n_imbalance_buckets=self.n_imbalance_buckets,
            max_spread_state_ticks=self.max_spread_state_ticks,
            max_mid_move_ticks=self.max_mid_move_ticks,
            min_state_count=self.min_state_count,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            adjustment_ticks=self.adjustment_ticks,
            g1_ticks=self.g1_ticks,
            state_counts=self.state_counts,
            diagnostics=np.asarray(
                [
                    diag.n_states if diag else self.n_states,
                    diag.n_observations if diag else 0,
                    diag.n_filtered_transitions if diag else 0,
                    diag.iterations if diag else 0,
                    int(diag.converged) if diag else 0,
                    diag.max_abs_adjustment_ticks if diag else 0.0,
                    diag.max_abs_last_term_ticks if diag else 0.0,
                ],
                dtype=np.float64,
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MicropriceEstimator":
        with np.load(path, allow_pickle=False) as data:
            estimator = cls(
                n_imbalance_buckets=int(data["n_imbalance_buckets"]),
                max_spread_state_ticks=int(data["max_spread_state_ticks"]),
                max_mid_move_ticks=float(data["max_mid_move_ticks"])
                if "max_mid_move_ticks" in data
                else 1.0,
                min_state_count=int(data["min_state_count"]) if "min_state_count" in data else 50,
                max_iterations=int(data["max_iterations"]),
                tolerance=float(data["tolerance"]),
            )
            estimator.adjustment_ticks = data["adjustment_ticks"].astype(np.float64)
            estimator.g1_ticks = data["g1_ticks"].astype(np.float64)
            estimator.state_counts = data["state_counts"].astype(np.int64)
        return estimator


def simple_microprice_adjustment_ticks(
    spread_ticks: np.ndarray,
    imbalance: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    return float(alpha) * (imbalance.astype(np.float64) - 0.5) * spread_ticks.astype(np.float64)

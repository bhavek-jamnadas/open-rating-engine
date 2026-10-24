"""Multi-period **individual** price optimizer — lifetime-multiplier (Idea A).

Built per ``docs/design/multi_period_individual_optimizer.md``. The decision is
**one constant multiplier per policy**, charged at every renewal, chosen to
maximise the NPV of each customer's lifetime expected margin subject to a
portfolio conversion constraint.

The problem looks intractable (one decision variable per policy, objective only
evaluable by forward simulation) but **decomposes** (design §6): in the
deterministic expected-value model policies do not interact (separability S1), so
the horizon objective is a plain sum of per-policy NPVs and the only coupling is
the portfolio conversion constraint — through a single scalar Lagrange multiplier
λ. So the solve is ``N`` independent per-policy argmaxes + a 1-D bisection on λ,
i.e. exactly :class:`PriceOptimizerIndividual`, fed **lifetime** ``(N, K)`` arrays
instead of single-period ones.

This class therefore:

- inherits the shared :class:`_MultiPeriodHorizonEngine` for aging / survivorship
  / ``previous_gwp`` feedback / NPV — *identical* to the rating-factor multi-period
  optimizer (design G3), no second copy to drift;
- builds the lifetime arrays by ``K`` full-book horizon sims, holding the
  per-policy overlay constant at each grid multiplier ``k`` (the multi-period
  analogue of :func:`simulate_adjustment_grid`, design §7.1);
- **delegates** the Lagrangian λ-bisection to :class:`PriceOptimizerIndividual`
  (unchanged); it does **not** call ``scipy.minimize`` (that is the rating-factor
  optimizer's path, intractable at individual scale — design §4.2);
- deploys the per-policy optimum as one static rating table (design §3.3).

The per-period *path* (dynamic-programming) variant — a different multiplier each
year — is designed separately and **not** implemented here (see
``docs/design/multi_period_individual_optimizer_path_dp.md``).
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from ._decision_space import _encode_keys
from ._horizon_engine import _COHORT_COL, _MultiPeriodHorizonEngine, _PERIOD_COL, _POLICY_ID_COL
from .individual_price_optimizer import PriceOptimizerIndividual

__all__ = ["PriceOptimizerIndividualMultiPeriod", "make_log_multiplier_penalty"]


# Horizon conversion summaries a policy's lifetime constraint can target (design
# §9.1). All are per-unit-of-birth-exposure rates so they average cleanly with the
# birth weights and reduce to the single-period conversion at ``n_periods == 1``.
_CONSTRAINT_METRICS = ("lifetime_average", "year1", "total_discounted")

# Default grid resolution when ``optimize`` is given ``bounds`` but no explicit grid.
_DEFAULT_GRID_SIZE = 21


def _per_exposure(total, birth_weights):
    """A per-policy total expressed per unit of birth exposure (``total / w_{i,0}``),
    guarding the rare zero-exposure policy (mapped to 0)."""
    safe = np.where(birth_weights > 0, birth_weights, 1.0)
    return np.where(birth_weights > 0, total / safe, 0.0)


def make_log_multiplier_penalty(l1_penalty=0.0, l2_penalty=0.0):
    """Build a per-multiplier ``regularizer`` for
    :meth:`PriceOptimizerIndividualMultiPeriod.optimize` (design §9.3).

    Penalizes the log of each grid multiplier — ``l1·|log k| + l2·(log k)²`` —
    vectorised over the grid. The optimizer subtracts it from every policy's
    per-exposure lifetime NPV before the argmax, nudging the overlay toward 1.0
    (no change): L1 encourages exact-1.0 sparsity, L2 shrinks large moves. The
    penalty is symmetric in ``log k`` so a markup and the reciprocal discount are
    penalized equally. The per-policy analogue of
    :func:`~open_rating_engine.optimization.multi_period_rating_factor_price_optimizer.make_log_multiplier_regularizer`
    (which takes a per-category dict; this takes the multiplier grid).
    """
    if l1_penalty < 0 or l2_penalty < 0:
        raise ValueError("l1_penalty and l2_penalty must be non-negative.")

    def _penalty(grid):
        log_k = np.log(np.asarray(grid, dtype=float))
        return l1_penalty * np.abs(log_k) + l2_penalty * log_k * log_k

    return _penalty


class PriceOptimizerIndividualMultiPeriod(_MultiPeriodHorizonEngine):
    """Per-policy multi-period optimizer (constant lifetime multiplier).

    Construct via :meth:`from_pipeline` from a per-policy demand-overlay
    ``RatingTablesLayer`` (one rating table keyed on the policy id, e.g.
    ``risk.demand_multiplier_individual`` keyed on ``risk_number``). Then
    :meth:`optimize` builds the lifetime ``(N, K)`` arrays and runs
    :class:`PriceOptimizerIndividual`'s λ-bisection over them; :meth:`to_per_policy_table`
    returns the deployable per-policy rating table.

    The optimization objective is the **per-exposure** lifetime NPV
    (``NPV_i / w_{i,0}``): the birth weight factors out of the Lagrangian per-row
    argmax (design §6), and this makes the optimizer reduce *exactly* to
    :class:`PriceOptimizerIndividual` when ``n_periods == 1`` and ``discount_rate == 0``
    (the keystone reduction, design §15.1).
    """

    # ------------------------------------------------------------------ #
    # Construction from a pipeline
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pipeline(
        cls,
        pipeline,
        layer_name,
        scored_data_nb,
        scored_data_rn,
        data_nb,
        data_rn,
        n_periods,
        aging_rules_rn,
        aging_rules_nb=None,
        **kwargs,
    ):
        """Build the optimizer, deriving the per-policy decision space from a
        single-table ``RatingTablesLayer`` registered on ``pipeline``.

        The overlay layer must hold exactly **one** rating table with a **single**
        key column (the per-policy id) — that is what makes it an *individual*
        overlay rather than a rating-factor one. The live ``pipeline`` is stored;
        the forward simulation deep-copies it, writes the swept constant multiplier
        into the per-policy table, and scores each period's book.

        Parameters mirror
        :meth:`PriceOptimizerRatingFactorMultiPeriod.from_pipeline`:
        ``scored_data_nb`` / ``scored_data_rn`` are ``pipeline.predict(...)`` output
        (used to read the layer's key column and seed the decision space);
        ``data_nb`` / ``data_rn`` / ``n_periods`` / ``aging_rules_*`` / ``**kwargs``
        are forwarded to ``__init__`` (the raw baskets, horizon config, and column
        names).
        """
        parsed = cls._parse_rating_layer(pipeline, layer_name, scored_data_rn)

        # An individual overlay is one table keyed on the per-policy id. Reject the
        # rating-factor shape (multiple factors / composite keys) with a clear error so
        # the per-policy deployment stays unambiguous. Validated up front — before the
        # decision space is built — because the per-policy derivation below needs the
        # single factor name and key column.
        factor_names = tuple(parsed["initial_multipliers"].keys())
        if len(factor_names) != 1:
            raise ValueError(
                "PriceOptimizerIndividualMultiPeriod expects a single per-policy overlay "
                f"table (one rating factor); layer '{layer_name}' has {list(factor_names)}. "
                "Use PriceOptimizerRatingFactorMultiPeriod for multi-factor overlays."
            )
        factor = factor_names[0]
        key_cols = parsed["layer_schema"][factor]["key_cols"]
        if len(key_cols) != 1:
            raise ValueError(
                "the per-policy overlay table must have a single key column (the policy id); "
                f"layer '{layer_name}' table '{factor}' is keyed on {key_cols}."
            )
        key_col = key_cols[0]

        obj = cls(
            data_nb=data_nb,
            data_rn=data_rn,
            n_periods=n_periods,
            aging_rules_rn=aging_rules_rn,
            aging_rules_nb=aging_rules_nb,
            **kwargs,
        )

        # Per-policy decisions price the **in-force** (renewal) book; new business is
        # rule-priced (design going_concern §4). Under runoff the whole seed book
        # (period-0 NB + RN) is optimized per policy; under going-concern only the
        # in-force renewal book is per-policy — the NB stream (period-0 NB + every
        # injected cohort) is held at the baseline layer default and excluded from the
        # per-policy harvest (Phase 1: a valuation frame, not yet an NB price lever).
        target = parsed["layer_meta"]["target_table"]
        going_concern = obj.parameters.new_business_mode == "going_concern"

        # Derive the per-policy decision space from the scored in-force book (like the
        # single-period PriceOptimizerIndividual) instead of requiring every policy to be
        # pre-seeded as a table row: every in-force policy id becomes a tunable category,
        # warm-started from the table's current value if present, else the layer default —
        # so a swept multiplier reaches every policy rather than the fixed default. Under
        # runoff the period-0 NB ids are tuned too; under going-concern they are deliberately
        # excluded (the NB stream is rule-priced, and NB/in-force ids must stay disjoint).
        # _encode_keys keeps these labels byte-identical to the table categories.
        seed_labels = _encode_keys(data_rn[target], key_cols)
        if not going_concern:
            seed_labels = np.concatenate([seed_labels, _encode_keys(data_nb[target], key_cols)])
        cats = parsed["initial_multipliers"][factor]
        default_value = parsed["default_multipliers"][factor]
        for label in seed_labels:
            cats.setdefault(label, default_value)

        # Materialize the derived categories as actual overlay-table rows: the engine's
        # _apply_overlay writes the swept multiplier vector positionally onto the table in
        # category_orders order, so every category must be a physical row. Rebuilt from the
        # (now augmented) category→value map, preserving order; the key column carries the
        # policy ids. (No-op in shape when the table was already fully pre-seeded.)
        overlay_layer = pipeline.layers[layer_name]
        overlay_layer.rating_tables[factor] = pd.DataFrame(
            {key_col: list(cats.keys()), "value": list(cats.values())}
        )

        n_rows = len(scored_data_rn[target])
        decision = obj._compute_decision_space(
            parsed["rating_factors"],
            parsed["initial_multipliers"],
            parsed["default_multipliers"],
            n_rows=n_rows,
        )
        for key, value in decision.items():
            setattr(obj.input, key, value)

        # Going-concern needs NB and in-force ids disjoint: an NB quote sharing an
        # in-force risk_number would be priced at that policy's renewal multiplier rather
        # than the baseline NB price (injected cohorts are made disjoint by the engine's
        # @nbt re-keying when id_columns is supplied — recommended for id-keyed pipelines).
        if going_concern:
            collide = sorted(map(str, set(data_nb[target][key_col]) & set(data_rn[target][key_col])))
            if collide:
                raise ValueError(
                    f"new-business and in-force policy ids overlap (e.g. {collide[:5]}); under "
                    "going_concern an NB quote must not share an in-force policy id."
                )

        # The objective / feedback columns the optimizer reads must exist in the scored
        # data under the names given. This is an easy slip with namespaced pipelines, where
        # the default bare names (e.g. "previous_gwp") do not match the scored columns (e.g.
        # "risk.previous_gwp") — and a wrong previous_gwp_col silently breaks the price-change
        # feedback (the rule is no longer excluded from the demographic trajectory and the
        # feedback is written to a column the demand model never reads). Fail loudly here.
        # sample_weight_col is intentionally not required (the engine defaults a missing weight
        # to 1.0 in _simulate_horizon).
        scored_cols = set(scored_data_rn[target].columns)
        required = {
            "gwp_col": obj.parameters.gwp_col,
            "previous_gwp_col": obj.parameters.previous_gwp_col,
            "conversion_col": obj.parameters.conversion_col,
            "profit_col": obj.parameters.profit_col,
            "cost_col": obj.parameters.cost_col,
        }
        missing = {param: col for param, col in required.items() if col not in scored_cols}
        if missing:
            raise ValueError(
                f"these *_col(s) are not columns of the scored data['{target}']: {missing}. "
                f"The names must match the pipeline's scored columns "
                f"(available: {sorted(scored_cols)}). For a namespaced pipeline pass e.g. "
                f"previous_gwp_col='{target}.previous_gwp'."
            )

        obj.pipeline = pipeline
        obj._layer_schema = parsed["layer_schema"]
        obj._layer_meta = parsed["layer_meta"]
        obj._policy_factor = factor_names[0]
        obj._policy_key_col = key_cols[0]
        return obj

    # ------------------------------------------------------------------ #
    # Going-concern partition (design going_concern §4 / §8.5 / §10.3)
    # ------------------------------------------------------------------ #

    def _partition_ids(self):
        """Going-concern: split the horizon's stable ``__policy_id`` space into the
        **in-force** block (the period-0 renewal cohort, which keeps its birth ids as it
        ages) and the **new-business stream** (period-0 NB + every injected cohort), read
        from the cached demographic trajectory's period-0 frame. The per-policy decision
        and harvest are restricted to the in-force ids (design going_concern §4 / §10.3);
        the NB stream is rule-priced at the baseline layer default. Returns
        ``(inforce_ids, nb_ids)`` as sets of ``__policy_id``."""
        target = self._layer_meta["target_table"]
        period0 = self._demographic_trajectory()[0][target]
        cohort = period0[_COHORT_COL].to_numpy()
        ids = period0[_POLICY_ID_COL].to_numpy()
        return set(ids[cohort == "rn"]), set(ids[cohort == "nb"])

    def _going_concern_split(self, agg):
        """Partition a whole-book scored horizon ``agg`` into in-force vs NB-stream NPV
        and volume (design going_concern §8.5), for the going-concern reporting columns.
        The decision optimizes only the in-force block; the NB stream is valued at the
        baseline lever it is held at."""
        inforce_ids, _ = self._partition_ids()
        df = agg.df
        is_if = df[_POLICY_ID_COL].isin(inforce_ids).to_numpy()
        npv = df["expected_npv"].to_numpy(dtype=float)
        vol = (
            df[self.parameters.sample_weight_col].to_numpy(dtype=float)
            * df[self.parameters.conversion_col].to_numpy(dtype=float)
        )
        npv_if, npv_nb = float(npv[is_if].sum()), float(npv[~is_if].sum())
        vol_if, vol_nb = float(vol[is_if].sum()), float(vol[~is_if].sum())
        return {
            "npv_inforce": npv_if, "npv_nb": npv_nb, "npv_whole_book": npv_if + npv_nb,
            "volume_inforce": vol_if, "volume_nb": vol_nb, "volume_whole_book": vol_if + vol_nb,
        }

    # ------------------------------------------------------------------ #
    # The precompute (design §7.1)
    # ------------------------------------------------------------------ #

    def _build_lifetime_arrays(self, grid, constraint_metric="lifetime_average", n_jobs=1):
        """Build the per-policy lifetime ``(N, K)`` arrays — the multi-period
        analogue of :func:`simulate_adjustment_grid`.

        For each constant grid multiplier ``k`` the per-policy overlay is held at
        ``k`` for every renewal and the horizon is forward-simulated via the
        inherited engine (aging / survivorship / ``previous_gwp`` feedback / NPV).
        Per policy we harvest:

        - ``lifetime_npv`` — **per unit of birth exposure**:
          ``(Σ_t w_{i,t}·p_{i,t}·(G_{i,t}-C_{i,t})/(1+r)^(t+δ)) / w_{i,0}``. Dividing
          by the birth weight is what makes the Lagrangian per-row argmax exact
          (design §6) and reduces it to the single-period ``expected_profit`` at
          ``n_periods == 1, r == 0``;
        - ``lifetime_conversion`` — the chosen horizon conversion summary (§9.1);
        - reporting ``lifetime_gwp`` / ``lifetime_volume`` (also per-exposure);
        - ``birth_weights`` ``w_{i,0}`` and the per-policy key (``risk_number``) for
          the constraint averaging and deployment.

        Exact under S1: because policies do not interact, sweeping one global
        constant ``k`` reads off every policy's true lifetime outcome *for that
        policy using ``k``*, and the per-row optimum may mix different ``k`` freely.
        """
        if not hasattr(self, "pipeline"):
            raise RuntimeError("_build_lifetime_arrays requires an optimizer built via from_pipeline().")
        grid = self._coerce_1d(grid, "grid")
        if grid.shape[0] == 0:
            raise ValueError("grid must be non-empty.")
        if np.any(grid <= 0):
            raise ValueError("grid multipliers must all be positive.")
        if constraint_metric not in _CONSTRAINT_METRICS:
            raise ValueError(
                f"constraint_metric must be one of {list(_CONSTRAINT_METRICS)}; got {constraint_metric!r}."
            )
        if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs == 0:
            raise ValueError("n_jobs must be a non-zero int (1 = sequential, -1 = all cores).")

        w_col = self.parameters.sample_weight_col
        conv_col = self.parameters.conversion_col
        gwp_col = self.parameters.gwp_col
        r = self.parameters.discount_rate
        delta = self.parameters.timing_offset
        key_col = self._policy_key_col

        npv_cols, conv_cols, gwp_cols, vol_cols = [], [], [], []
        policy_index = None
        birth_weights = None
        risk_keys = None

        # The x-independent demographic trajectory is shared by every sweep, so build
        # it once up front; the (optionally parallel) sims below then only read its
        # cache. Each sim deep-copies the pipeline, so the K sweeps are independent
        # (the ThreadPoolExecutor pattern from simulate_adjustment_grid, design §12).
        self._demographic_trajectory()

        # Going-concern: restrict the per-policy harvest to the in-force renewal book.
        # New business (period-0 + every injected cohort) is rule-priced at the baseline
        # layer default and excluded — which also fixes the period-0-centric birth-weight
        # reindex (an NB-born policy has no period-0 row → NaN weight). Under runoff
        # ``inforce_ids`` stays None and every policy is harvested, exactly as before
        # (design going_concern §10.3).
        inforce_ids = None
        if self.parameters.new_business_mode == "going_concern":
            inforce_ids, _ = self._partition_ids()

        def _scored_frame(k):
            return self._simulate_and_aggregate(np.full(self.input.param_total, float(k))).df

        if n_jobs == 1:
            frames = [_scored_frame(k) for k in grid]
        else:
            max_workers = n_jobs if n_jobs > 0 else None
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                frames = list(executor.map(_scored_frame, [float(k) for k in grid]))

        for df in frames:
            if inforce_ids is not None:
                df = df[df[_POLICY_ID_COL].isin(inforce_ids)]
            ids = df[_POLICY_ID_COL].to_numpy()
            w = df[w_col].to_numpy(dtype=float)
            p = df[conv_col].to_numpy(dtype=float)
            disc = (1.0 + r) ** (df[_PERIOD_COL].to_numpy(dtype=float) + delta)

            frame = pd.DataFrame({
                _POLICY_ID_COL: ids,
                "npv": df["expected_npv"].to_numpy(dtype=float),   # w·p·(G-C)/disc
                "wp": w * p,                                       # expected renewals (undiscounted)
                "w": w,
                "wp_disc": w * p / disc,                           # discounted expected renewals
                "egwp": w * p * df[gwp_col].to_numpy(dtype=float) / disc,  # discounted expected premium
            })
            grouped = frame.groupby(_POLICY_ID_COL, sort=True)
            sums = grouped.sum()

            if policy_index is None:
                policy_index = sums.index
                # Birth weight and the per-policy key are x-independent (the period-0
                # exposure and the policy id), so capture them once.
                period0 = df[df[_PERIOD_COL] == 0]
                birth_weights = (
                    period0.set_index(_POLICY_ID_COL)[w_col].reindex(policy_index).to_numpy(dtype=float)
                )
                risk_keys = period0.set_index(_POLICY_ID_COL)[key_col].reindex(policy_index).to_numpy()

            sums = sums.reindex(policy_index)
            total_npv = sums["npv"].to_numpy(dtype=float)
            sum_wp = sums["wp"].to_numpy(dtype=float)
            sum_w = sums["w"].to_numpy(dtype=float)
            sum_wp_disc = sums["wp_disc"].to_numpy(dtype=float)
            sum_egwp = sums["egwp"].to_numpy(dtype=float)

            npv_cols.append(_per_exposure(total_npv, birth_weights))
            gwp_cols.append(_per_exposure(sum_egwp, birth_weights))
            vol_cols.append(_per_exposure(sum_wp, birth_weights))

            if constraint_metric == "lifetime_average":
                conv_cols.append(np.where(sum_w > 0, sum_wp / sum_w, 0.0))
            elif constraint_metric == "total_discounted":
                conv_cols.append(_per_exposure(sum_wp_disc, birth_weights))
            else:  # "year1" — first-period retention
                y1 = (
                    df[df[_PERIOD_COL] == 0].set_index(_POLICY_ID_COL)[conv_col]
                    .reindex(policy_index).to_numpy(dtype=float)
                )
                conv_cols.append(y1)

        return {
            "lifetime_npv": np.column_stack(npv_cols),
            "lifetime_conversion": np.column_stack(conv_cols),
            "lifetime_gwp": np.column_stack(gwp_cols),
            "lifetime_volume": np.column_stack(vol_cols),
            "birth_weights": birth_weights,
            "risk_keys": risk_keys,
            "policy_index": np.asarray(policy_index),
        }

    # ------------------------------------------------------------------ #
    # Public surface, mirroring the other optimizers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_grid(grid, bounds):
        """The price-point grid: ``grid`` verbatim if given, else a linspace across
        ``bounds`` at the default resolution."""
        if grid is not None:
            return np.asarray(grid, dtype=float)
        if not (isinstance(bounds, tuple) and len(bounds) == 2):
            raise TypeError("bounds must be a (lo, hi) tuple when grid is not given.")
        lo, hi = float(bounds[0]), float(bounds[1])
        if not (0 < lo <= hi):
            raise ValueError("bounds must satisfy 0 < lo <= hi.")
        return np.linspace(lo, hi, _DEFAULT_GRID_SIZE)

    @staticmethod
    def _adjust_objective(lifetime_npv, grid, enbp_ceiling, regularizer):
        """Apply the compliance / regularization adjustments to the per-exposure
        lifetime-NPV objective before the per-row argmax (design §9.3):

        - **regularizer** — subtract a per-multiplier penalty (e.g. L1/L2 on
          ``log k`` from :func:`make_log_multiplier_penalty`) from every policy's
          column, nudging the overlay toward 1.0;
        - **ENBP / GIPP ceiling** — forbid multipliers above ``enbp_ceiling`` by
          setting those columns to ``-inf`` so the argmax can never pick them. The
          compliant set is enforced *exactly*, with no penalty weight to tune (the
          individual-optimization advantage, design §9.3 / §11). The mask is the
          same for every policy, so each row keeps a non-empty feasible set.
        """
        obj = np.array(lifetime_npv, dtype=float, copy=True)
        if regularizer is not None:
            penalty = np.asarray(regularizer(grid), dtype=float)
            if penalty.shape != grid.shape:
                raise ValueError(
                    f"regularizer must return one penalty per grid multiplier (shape {grid.shape}); "
                    f"got {penalty.shape}."
                )
            obj = obj - penalty[None, :]
        if enbp_ceiling is not None:
            forbidden = grid > float(enbp_ceiling) + 1e-9
            if forbidden.all():
                raise ValueError(
                    f"enbp_ceiling={enbp_ceiling} forbids every grid multiplier "
                    f"({grid.min():.4g}..{grid.max():.4g}); raise the ceiling or lower the grid."
                )
            obj[:, forbidden] = -np.inf
        return obj

    @staticmethod
    def _cohort(arrays, decimals=9):
        """Bucket policies sharing an identical lifetime response (design §12).

        Policies whose ``(lifetime_npv, lifetime_conversion, lifetime_gwp,
        lifetime_volume)`` rows agree (to ``decimals`` places; ``None`` for
        bit-identical) respond identically to every grid multiplier, so they pick the
        same optimum and collapse into one weighted cell. Because the objective and
        constraint are weight-sums this is **exact** at the grouping precision — the
        cohorted solve reproduces the full-resolution per-policy optimum — and is the
        speed-up that makes large books cheap (e.g. when a tree-based demand model
        buckets inputs into identical predictions). The default 9-dp rounding absorbs
        the per-exposure division's floating-point noise so feature-identical policies
        with *different weights* still merge; lower ``decimals`` for coarser
        (approximate) cohorting, validating the NPV gap. Returns ``(cohorted_arrays,
        inverse)`` mapping each policy to its cohort so the optimum expands back per
        policy; each cohort keeps its first member's exact (un-rounded) response.
        """
        npv, conv = arrays["lifetime_npv"], arrays["lifetime_conversion"]
        gwp, vol = arrays["lifetime_gwp"], arrays["lifetime_volume"]
        k = npv.shape[1]
        key = np.concatenate([npv, conv, gwp, vol], axis=1)
        grouped = key if decimals is None else np.round(key, decimals)
        _, idx, inverse = np.unique(grouped, axis=0, return_index=True, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        rep = key[idx]  # representative (first-member) exact response per cohort
        cohort_w = np.zeros(idx.shape[0], dtype=float)
        np.add.at(cohort_w, inverse, arrays["birth_weights"])
        cohorted = {
            "lifetime_npv": rep[:, :k],
            "lifetime_conversion": rep[:, k:2 * k],
            "lifetime_gwp": rep[:, 2 * k:3 * k],
            "lifetime_volume": rep[:, 3 * k:4 * k],
            "birth_weights": cohort_w,
        }
        return cohorted, inverse

    def _make_inner(self, grid, constraint_metric, enbp_ceiling, regularizer, cohort=False, cohort_decimals=9, n_jobs=1):
        """Build the lifetime arrays and the delegate :class:`PriceOptimizerIndividual`
        with the ENBP-masked / regularized objective. Shared by :meth:`optimize` and
        :meth:`calculate_efficient_frontier` so the arrays + adjustments are built once.

        With ``cohort=True`` the policies are first bucketed into their ``M`` distinct
        lifetime-response rows (:meth:`_cohort`); ``inverse`` maps every policy back to
        its cohort for deployment (``None`` without cohorting). ``n_jobs`` parallelises
        the ``K`` horizon sweeps.
        """
        arrays = self._build_lifetime_arrays(grid, constraint_metric=constraint_metric, n_jobs=n_jobs)
        work, inverse = (self._cohort(arrays, cohort_decimals) if cohort else (arrays, None))
        objective = self._adjust_objective(work["lifetime_npv"], grid, enbp_ceiling, regularizer)
        inner = PriceOptimizerIndividual(
            adjustment_factors=grid,
            objective_array=objective,
            constraint_array=work["lifetime_conversion"],
            sample_weights=work["birth_weights"],
            reporting_arrays={
                "lifetime_gwp": work["lifetime_gwp"],
                "lifetime_volume": work["lifetime_volume"],
            },
        )
        return inner, arrays, inverse

    def baseline(self, multiplier=1.0, *, constraint_metric="lifetime_average"):
        """Evaluate the horizon at a fixed UNIFORM per-policy ``multiplier`` WITHOUT
        optimizing — by default ``1.0`` (do-nothing: today's book re-simulated over
        ``n_periods`` with no demand adjustment). Pure: no solver, no logging, no
        progress side effects.

        Use ``baseline()["conversion"]`` to derive a realistic ``target_constraint``
        for :meth:`optimize`: it is the lifetime, exposure-weighted conversion the
        constraint actually measures — aggregated *exactly* as
        :class:`PriceOptimizerIndividual` does (the birth-weighted average of the
        per-policy lifetime conversions, **not** the engine's row-pooled
        ``agg.constraint``) — and, being the do-nothing point inside the default
        ``bounds`` / ``enbp_ceiling``, an achievable equality target. A one-period
        renewal-book mean overstates it (renewals convert higher than new business,
        and a snapshot misses the horizon blend).

        Pass ``constraint_metric`` to match the metric you will optimize against
        (``"lifetime_average"`` (default), ``"year1"``, ``"total_discounted"``);
        pass ``multiplier`` to score any other uniform overlay.

        Mirrors :meth:`PriceOptimizerRatingFactorMultiPeriod.baseline`. Returns
        ``{"conversion", "npv", "total_volume", "summary", "per_policy", "per_period",
        "output_dataframe"}`` — where ``npv`` is the **total** book lifetime NPV
        (``Σ_i NPV_i``), ``summary`` a one-row frame whose columns match
        :meth:`optimize`'s summary (so a baseline and an optimized run line up
        column-for-column), ``per_policy`` the per-policy do-nothing response keyed on
        the policy id, ``per_period`` the per-period horizon breakdown, and
        ``output_dataframe`` the scored horizon frame (policy×period rows, as RF baseline).
        """
        arrays = self._build_lifetime_arrays(
            np.array([float(multiplier)]), constraint_metric=constraint_metric
        )
        w = arrays["birth_weights"]
        w_total = float(np.sum(w))
        if w_total <= 0:
            raise ValueError(
                "baseline needs a positive total birth exposure (the sum of birth weights is 0)."
            )

        conv = arrays["lifetime_conversion"][:, 0]
        npv_pe = arrays["lifetime_npv"][:, 0]
        gwp_pe = arrays["lifetime_gwp"][:, 0]
        vol_pe = arrays["lifetime_volume"][:, 0]

        # Birth-weighted portfolio aggregates — the exact quantities the inner
        # PriceOptimizerIndividual reports, so baseline()["conversion"] is a
        # reproducible optimize() target and the summaries are directly comparable.
        conversion = float(np.average(conv, weights=w))
        objective = float(np.average(npv_pe, weights=w))   # per-exposure NPV (matches inner)
        gwp_avg = float(np.average(gwp_pe, weights=w))
        vol_avg = float(np.average(vol_pe, weights=w))
        total_npv = float(np.sum(w * npv_pe))              # Σ_i NPV_i (interpretable headline)
        total_volume = float(np.sum(w * vol_pe))

        # Same enriched columns (and order) as optimize()'s summary, so a baseline and
        # an optimized run line up column-for-column. constraint_gap is 0 (the do-nothing
        # target is its own conversion).
        summary = pd.DataFrame({
            "target_constraint": [conversion],
            "constraint": [conversion],
            "objective": [objective],
            "price_multiplier": [float(multiplier)],
            "lifetime_gwp": [gwp_avg],
            "lifetime_volume": [vol_avg],
            "npv": [total_npv],
            "total_volume": [total_volume],
            "constraint_metric": [constraint_metric],
            "constraint_gap": [0.0],
        })
        per_policy = pd.DataFrame({
            self._policy_key_col: arrays["risk_keys"],
            "value": float(multiplier),
            "lifetime_conversion": conv,
            "lifetime_npv": npv_pe,
            "lifetime_gwp": gwp_pe,
            "lifetime_volume": vol_pe,
            "birth_weight": w,
        })
        # Per-period horizon breakdown of the do-nothing book (mirrors the rating-factor
        # baseline's per_period_summary): re-score at the uniform overlay and reduce per
        # period.
        agg = self._simulate_and_aggregate(np.full(self.input.param_total, float(multiplier)))
        # Going-concern: same whole-book valuation columns as optimize()'s summary, so a
        # baseline and an optimized run still line up column-for-column (design §8.5).
        if self.parameters.new_business_mode == "going_concern":
            for col, val in self._going_concern_split(agg).items():
                summary[col] = val
        per_period = self._per_period_summary(agg)
        return {
            "conversion": conversion,
            "npv": total_npv,
            "total_volume": total_volume,
            "summary": summary,
            "per_policy": per_policy,
            "per_period": per_period,
            "output_dataframe": agg.df,   # scored horizon frame (policy×period), as RF baseline
        }

    # ------------------------------------------------------------------ #
    # Optimize: shared scaffold (_run_optimize) + the per-class inner solve
    # (_solve_core). The per-period path optimizer overrides _solve_core only.
    # ------------------------------------------------------------------ #

    def _run_optimize(self, target_constraint, *, save_to_log, core_kwargs):
        """Shared ``optimize`` scaffold for every individual multi-period optimizer.

        Delegates the inner solve to :meth:`_solve_core` (the lifetime λ-bisection
        here; subclasses may override it), then enriches the summary with the horizon
        KPIs, re-scores the book at the deployed per-policy optimum (one post-solve forward sim) for
        ``output_dataframe`` / ``per_period_summary``, and optionally logs to
        ``self.output``.

        ``_solve_core(target_constraint, **core_kwargs)`` must return
        ``(optima, summary_row_df, arrays)``: ``optima`` the deployable per-policy
        ``(N,)`` multiplier aligned to ``arrays["risk_keys"]``; ``summary_row_df`` a
        one-row frame carrying at least ``objective`` / ``constraint`` /
        ``lifetime_volume``; ``arrays`` carrying ``birth_weights`` / ``risk_keys`` /
        ``policy_index``. ``core_kwargs`` must include ``constraint_metric``.
        """
        optima, summary_core, arrays = self._solve_core(target_constraint, **core_kwargs)

        # Enrich the inner summary with the horizon KPIs the rating-factor optimizer
        # reports, from the birth-weighted totals: npv = Σ_i NPV_i, total_volume =
        # Σ lifetime volume, plus the constrained metric and the realised gap.
        w_total = float(np.sum(arrays["birth_weights"]))
        summary = summary_core.copy()
        summary["npv"] = summary["objective"] * w_total
        summary["total_volume"] = summary["lifetime_volume"] * w_total
        summary["constraint_metric"] = core_kwargs["constraint_metric"]
        summary["constraint_gap"] = summary["constraint"] - target_constraint
        # Re-score the book at the deployed per-policy optimum (one post-solve forward
        # sim, matching the rating-factor optimizer). For the lifetime optimizer this is
        # the chosen constant multiplier; for the path optimizer it is the deployed
        # period-0 action (design §8 Option 1 — the static this-year table), so
        # output_dataframe is always the scored horizon frame of what actually deploys
        # (policy×period rows). The deployable per-policy table is to_per_policy_table().
        overlay = self._per_policy_overlay(optima, arrays["risk_keys"])
        agg = self._simulate_and_aggregate(self._flatten(overlay))
        # Going-concern: surface the whole-book valuation (in-force optimized + the NB
        # stream held at baseline) alongside the optimized in-force objective (design
        # §8.5). The headline npv/total_volume stay the in-force optimized block; the
        # *_whole_book / *_nb columns add the new-business stream for analysis.
        if self.parameters.new_business_mode == "going_concern":
            for col, val in self._going_concern_split(agg).items():
                summary[col] = val
        result = {"optimal_multipliers": optima, "summary": summary, "output_dataframe": agg.df}

        self._last_arrays = arrays
        if save_to_log:
            self.output.optimal_multipliers = optima
            self.output.risk_keys = arrays["risk_keys"]
            self.output.policy_index = arrays["policy_index"]
            self.output.output_optimized = agg.df
            self.output.summary = summary
            self.output.per_period_summary = self._per_period_summary(agg)
            self.output.optimization_scoring_history = pd.concat(
                [self.output.optimization_scoring_history, summary],
                ignore_index=True,
            )
        return result

    def _solve_core(self, target_constraint, *, grid, constraint_metric, bounds,
                    enbp_ceiling, regularizer, cohort, cohort_decimals, n_jobs,
                    lagrangian_lambda, tol, max_iter):
        """Lifetime inner solve: build the per-policy lifetime ``(N, K)`` arrays and run
        :class:`PriceOptimizerIndividual`'s λ-bisection over them (delegated). The path
        subclass overrides this with a per-period dynamic program. Returns
        ``(optima(N,), summary_row_df, arrays)`` per the :meth:`_run_optimize` contract.
        """
        grid = self._resolve_grid(grid, bounds)
        inner, arrays, inverse = self._make_inner(
            grid, constraint_metric, enbp_ceiling, regularizer,
            cohort=cohort, cohort_decimals=cohort_decimals, n_jobs=n_jobs,
        )
        result = inner.optimize(
            target_constraint=target_constraint,
            tol=tol,
            max_iter=max_iter,
            lagrangian_lambda=lagrangian_lambda,
            save_to_log=True,
        )
        # Expand the per-cohort optimum back to one multiplier per policy (a no-op
        # without cohorting) so the result + deployment are always per policy.
        optima = result["optimal_multipliers"]
        if inverse is not None:
            optima = optima[inverse]
        self._inner = inner
        return optima, result["summary"], arrays

    def optimize(
        self,
        target_constraint,
        *,
        grid=None,
        constraint_metric="lifetime_average",
        bounds=(0.9, 1.1),
        enbp_ceiling=1.0,
        regularizer=None,
        cohort=False,
        cohort_decimals=9,
        n_jobs=1,
        lagrangian_lambda=(-1e3, 1e3),
        tol=1e-4,
        max_iter=50,
        save_to_log=True,
    ):
        """Optimize the per-policy lifetime multipliers against a portfolio
        conversion target.

        Builds the lifetime arrays (design §7.1), applies the compliance /
        regularization adjustments, then runs :class:`PriceOptimizerIndividual`'s
        λ-bisection over them (delegated). Returns
        ``{"optimal_multipliers", "output_dataframe", "summary"}`` — where
        ``optimal_multipliers`` is the per-policy ``(N,)`` chosen multiplier (aligned
        with :meth:`to_per_policy_table`'s policy id) and ``output_dataframe`` is the
        **scored horizon frame** at that optimum (policy×period rows, as the
        rating-factor optimizer reports — re-scored by one post-solve sim, also stored
        on ``self.output.output_optimized`` with the per-period breakdown on
        ``self.output.per_period_summary``).

        Compliance-tight defaults (design §11 — make the compliant strategy the easy
        path):

        - ``enbp_ceiling`` (default ``1.0``) caps every renewal multiplier at the
          Equivalent-New-Business-Price ceiling, enforced *exactly* as a grid mask
          (design §9.3) — the GIPP price-walking guard, **on by default**. Pass
          ``None`` to disable (e.g. an unconstrained benchmark) or another scalar to
          move the ceiling.
        - ``bounds`` default to a tight ``(0.9, 1.1)``.
        - ``regularizer`` (default ``None``) is an optional per-multiplier penalty
          callable (see :func:`make_log_multiplier_penalty`) subtracted from each
          policy's per-exposure lifetime NPV, nudging the overlay toward 1.0 and
          damping over-fitting to per-policy demand noise.

        Performance (design §12): ``n_jobs`` parallelises the ``K`` horizon sweeps
        (``-1`` = all cores); ``cohort=True`` collapses policies with an identical
        lifetime response into weighted cells before the solve — *exact*, and a large
        speed-up on books with repeated profiles.
        """
        return self._run_optimize(
            target_constraint,
            save_to_log=save_to_log,
            core_kwargs=dict(
                grid=grid,
                constraint_metric=constraint_metric,
                bounds=bounds,
                enbp_ceiling=enbp_ceiling,
                regularizer=regularizer,
                cohort=cohort,
                cohort_decimals=cohort_decimals,
                n_jobs=n_jobs,
                lagrangian_lambda=lagrangian_lambda,
                tol=tol,
                max_iter=max_iter,
            ),
        )

    def calculate_efficient_frontier(
        self,
        target_constraint_range,
        *,
        grid=None,
        constraint_metric="lifetime_average",
        bounds=(0.9, 1.1),
        enbp_ceiling=1.0,
        regularizer=None,
        cohort=False,
        cohort_decimals=9,
        n_jobs=1,
        lagrangian_lambda=(-1e3, 1e3),
        tol=1e-4,
        max_iter=50,
    ):
        """Sweep a range of conversion targets. The lifetime arrays (and the ENBP /
        regularization adjustments) are built **once** and every target is then a
        pure-NumPy λ-bisection (design §12). ``enbp_ceiling`` / ``regularizer`` /
        ``cohort`` / ``n_jobs`` behave as in :meth:`optimize`; the frontier rows are
        portfolio-level, so cohorting needs no per-policy expansion here."""
        grid = self._resolve_grid(grid, bounds)
        inner, arrays, _ = self._make_inner(
            grid, constraint_metric, enbp_ceiling, regularizer,
            cohort=cohort, cohort_decimals=cohort_decimals, n_jobs=n_jobs
        )
        frontier = inner.calculate_efficient_frontier(
            target_constraint_range, tol=tol, max_iter=max_iter, lagrangian_lambda=lagrangian_lambda
        )
        self._inner = inner
        self._last_arrays = arrays
        self.output.efficient_frontier = frontier
        return frontier

    # ------------------------------------------------------------------ #
    # Deployment (design §3.3 / §7.3)
    # ------------------------------------------------------------------ #

    def to_per_policy_table(self):
        """Deploy the optimum as a static per-policy rating table: a DataFrame of
        the policy key column (e.g. ``risk_number``) plus the chosen ``value``,
        ready to drop into the overlay layer (exactly as single-period individual)."""
        if not hasattr(self.output, "optimal_multipliers"):
            raise RuntimeError("No optimum available; run optimize() first.")
        return pd.DataFrame({
            self._policy_key_col: self.output.risk_keys,
            "value": np.asarray(self.output.optimal_multipliers, dtype=float),
        })

    def to_rating_tables(self, optimal_multipliers=None):
        """Not the deployment path for the individual optimizer — the optimum is
        per *policy*, not per rating-factor *category*. Use
        :meth:`to_per_policy_table` instead (which also backs ``to_layer``)."""
        raise NotImplementedError(
            "PriceOptimizerIndividualMultiPeriod deploys per-policy multipliers; "
            "use to_per_policy_table() instead of to_rating_tables() / to_layer()."
        )

    def _per_policy_overlay(self, multipliers, risk_keys):
        """Adapt a per-policy ``(N,)`` multiplier array (aligned with ``risk_keys``) to
        the ``{factor: {category: multiplier}}`` overlay the decision space / engine
        expect: every overlay category seeded at its layer value, the optimized policies
        overridden by their chosen multiplier (so ``_flatten`` sees a complete overlay).
        Used to re-simulate the horizon at the optimum (the per-period summary in
        :meth:`optimize`) and to stress-test it (:meth:`monte_carlo_npv`)."""
        factor = self._policy_factor
        overlay = dict(self.input.initial_multipliers[factor])
        for key, mult in zip(risk_keys, np.asarray(multipliers, dtype=float)):
            overlay[key] = float(mult)
        return {factor: overlay}

    def monte_carlo_npv(self, optimal_multipliers=None, **kwargs):
        """Post-hoc Monte-Carlo of the horizon NPV under the chosen per-policy overlay
        (design §10.2 / §17.5), reusing the inherited engine sampler.

        The deterministic :meth:`optimize` reports one *expected* NPV by carrying a
        fractional survivorship weight; this samples renew/lapse per policy per period
        to yield the NPV **distribution** and tail risk (VaR/CVaR). With
        ``optimal_multipliers=None`` it stress-tests the last :meth:`optimize` result:
        the per-policy ``(N,)`` optimum is adapted (via :meth:`_per_policy_overlay`) to
        the ``{factor: {risk_number: multiplier}}`` dict the engine expects. Pass an
        explicit dict to evaluate a hand-picked overlay. ``**kwargs`` (``n_sims``,
        ``seed``, ``percentiles``, ``var_alpha``, …) pass straight through to the engine.
        """
        if optimal_multipliers is None:
            stored = getattr(self.output, "optimal_multipliers", None)
            if stored is None:
                raise RuntimeError(
                    "monte_carlo_npv needs a per-policy optimum; run optimize() first "
                    "or pass an explicit {factor: {risk_number: multiplier}} overlay."
                )
            optimal_multipliers = self._per_policy_overlay(stored, self.output.risk_keys)
        return super().monte_carlo_npv(optimal_multipliers, **kwargs)

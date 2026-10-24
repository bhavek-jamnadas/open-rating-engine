"""Shared multi-period horizon-simulation engine.

Extracted — with **no behaviour change** — from
:mod:`multi_period_price_optimizer` so the rating-factor multi-period optimizer
and the (forthcoming) per-policy individual optimizer inherit **one** engine
rather than each carrying its own copy (design
``docs/design/multi_period_individual_optimizer.md`` §13–§14). The engine owns:

- two-basket aging (``_age_one_step`` / ``_fresh_nb`` / ``_age_book_trajectory``)
  via forge's shared :func:`apply_aging`, and the cached x-independent demographic
  trajectory (``_demographic_trajectory``);
- demand-overlay injection (``_apply_overlay``) and the forward simulation with
  survivorship by weight-carrying and the ``previous_gwp ← gwp`` feedback
  (``_simulate_horizon``), reduced to NPV / volume / lifetime conversion
  (``_simulate_and_aggregate``);
- post-hoc Monte-Carlo survivorship risk (``_build_mc_inputs`` /
  ``_simulate_mc_npv`` / ``monte_carlo_npv``).

Building both optimizers on this base guarantees a strategy validated in either
ages, survives, and discounts identically to the other and to the app when it
scores the deployed overlay (design goal G3) — there is no second copy to drift.

The engine extends :class:`_RatingFactorDecisionSpace` so its forward simulation
can reuse the inherited decision-space helpers (``_unflatten`` / ``_flatten`` /
``_effective_multiplier``). It is **not instantiated directly**: subclasses build
``self.input`` / ``self.parameters`` / ``self.output`` (the basket data, horizon
config, and column names the methods below read) and add their own solve on top.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..data_processing.data_aging import apply_aging
from ._decision_space import _RatingFactorDecisionSpace


# Internal tagging columns added to the simulated book.
_PERIOD_COL = "__period"
_COHORT_COL = "__cohort"  # "nb" (fresh new business) | "rn" (carried renewal book)
_POLICY_ID_COL = "__policy_id"  # stable per-policy id (assigned at birth) for Monte-Carlo pivots

# New-business inflow framings (design §10.3). "runoff" values today's book and
# its future renewals only (no new NB after period 0); "going_concern" injects a
# fresh un-aged NB cohort every period.
_NEW_BUSINESS_MODES = ("runoff", "going_concern")


def _npv_distribution_summary(samples, percentiles, var_alpha):
    """Reduce an array of Monte-Carlo NPV samples to summary + tail risk metrics.

    Returns ``(stats, pct, var, cvar)`` where ``stats`` is a dict
    (``mean``/``std``/``min``/``max``), ``pct`` maps each requested percentile to
    its NPV level, ``var`` is the ``var_alpha``-quantile NPV (a profit level — the
    "we are ``1-var_alpha`` confident NPV is at least this" floor) and ``cvar`` is
    the mean of samples at or below it (expected shortfall). By construction
    ``cvar <= var <= median <~ mean`` for a left tail.
    """
    n = len(samples)
    stats = {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples, ddof=1)) if n > 1 else 0.0,
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
    }
    pct = {(int(p) if float(p).is_integer() else float(p)): float(np.percentile(samples, p)) for p in percentiles}
    var = float(np.quantile(samples, var_alpha))
    tail = samples[samples <= var]
    cvar = float(tail.mean()) if tail.size else var
    return stats, pct, var, cvar


class _MultiPeriodHorizonEngine(_RatingFactorDecisionSpace):
    """Horizon-simulation engine shared by the multi-period optimizers.

    Holds the aging / overlay / forward-simulation / NPV / Monte-Carlo machinery
    that is independent of *how* a candidate overlay ``x`` is chosen. Subclasses
    supply the decision space (via :class:`_RatingFactorDecisionSpace` /
    ``from_pipeline``) and the solve that drives ``x`` — the rating-factor
    optimizer with a dense ``scipy`` objective, the individual optimizer with a
    per-policy lifetime-array precompute + Lagrangian decomposition.

    Every method here reads only instance state the subclass sets up
    (``self.input`` baskets, ``self.parameters`` horizon config + column names,
    ``self.pipeline``, ``self._layer_meta``, ``self._demographic_aging_rules``,
    ``self._demo_trajectory_cache``, ``self.output``) plus inherited
    decision-space helpers — so the engine is behaviour-identical wherever it is
    mixed in.
    """

    def __init__(
        self,
        data_nb,
        data_rn,
        n_periods,
        aging_rules_rn,
        aging_rules_nb=None,
        *,
        new_business_mode="runoff",
        discount_rate=0.10,
        timing_offset=0.5,
        nb_growth_rate=0.0,
        sample_weight_col="sample_weights",
        conversion_col="conversion_rate",
        profit_col="reporting_expected_profit",
        gwp_col="gwp",
        cost_col="reporting_offered_technical_premium",
        previous_gwp_col="previous_gwp",
        id_columns=None,
    ):
        """Validate the baskets, aging rules, and horizon configuration shared by
        every multi-period optimizer and set up the engine state the simulation
        methods read: ``self.input`` (baskets; ``from_pipeline`` adds the
        decision-space attributes), ``self.parameters`` (horizon config + column
        names), ``self.output``, and the cached x-independent demographic
        trajectory (``_demographic_aging_rules`` / ``_demo_trajectory_cache``).

        Subclasses call ``super().__init__(...)`` and add their solver-specific
        state on top (e.g. the rating-factor optimizer appends the objective and
        regularizer to ``self.parameters``); the per-policy individual optimizer
        needs nothing more.
        """
        data_nb = self._validate_basket(data_nb, "data_nb")
        data_rn = self._validate_basket(data_rn, "data_rn")
        id_columns = self._validate_id_columns(id_columns, data_nb)

        if not isinstance(n_periods, int) or isinstance(n_periods, bool) or n_periods < 1:
            raise ValueError("n_periods must be an integer >= 1.")

        aging_rules_rn = self._validate_aging_rules(aging_rules_rn, "aging_rules_rn", required=True)
        aging_rules_nb = self._validate_aging_rules(aging_rules_nb, "aging_rules_nb", required=False)

        if new_business_mode not in _NEW_BUSINESS_MODES:
            raise ValueError(
                f"new_business_mode must be one of {list(_NEW_BUSINESS_MODES)}; got {new_business_mode!r}."
            )

        for nm, val in (
            ("discount_rate", discount_rate),
            ("timing_offset", timing_offset),
            ("nb_growth_rate", nb_growth_rate),
        ):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise TypeError(f"{nm} must be numeric.")
        if discount_rate <= -1:
            raise ValueError("discount_rate must be > -1 so the discount factor (1 + r) stays positive.")
        if nb_growth_rate <= -1:
            raise ValueError("nb_growth_rate must be > -1 so the growth factor (1 + g) stays positive.")

        # Baskets live on ``self.input``; ``from_pipeline`` adds the decision-space
        # attributes here. Aging/horizon configuration lives on ``self.parameters``.
        self.input = SimpleNamespace(data_nb=data_nb, data_rn=data_rn)
        self.parameters = SimpleNamespace(
            n_periods=n_periods,
            aging_rules_rn=aging_rules_rn,
            aging_rules_nb=aging_rules_nb,
            new_business_mode=new_business_mode,
            discount_rate=float(discount_rate),
            timing_offset=float(timing_offset),
            nb_growth_rate=float(nb_growth_rate),
            sample_weight_col=sample_weight_col,
            conversion_col=conversion_col,
            profit_col=profit_col,
            gwp_col=gwp_col,
            cost_col=cost_col,
            previous_gwp_col=previous_gwp_col,
            id_columns=id_columns,
        )
        self.output = SimpleNamespace(
            optimization_scoring_history=pd.DataFrame({
                "target_constraint": [],
                "constraint": [],
                "objective": [],
                "price_multiplier": [],
            })
        )

        # Performance split (design §13.1): the demographic aging trajectory is
        # x-independent and can be precomputed once; only the ``previous_gwp ← gwp``
        # feedback depends on x, so it is excluded here and threaded per
        # evaluation in ``_simulate_horizon``. ``_demo_trajectory_cache`` memoises
        # the trajectory the first time it is built.
        self._demographic_aging_rules = {
            name: cfg for name, cfg in aging_rules_rn.items()
            if cfg.get("source_column") != previous_gwp_col
        }
        self._demo_trajectory_cache = None

    # --------------------------------------------------------------------- #
    # Validation helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _validate_basket(basket, name):
        """Validate and defensively copy a pipeline-state basket.

        Returns a new ``{table: DataFrame}`` dict of copies so later aging never
        mutates the caller's frames.
        """
        if not isinstance(basket, dict) or not basket:
            raise TypeError(
                f"{name} must be a non-empty dict[str, pandas.DataFrame] (pipeline-state shape)."
            )
        cleaned = {}
        for table, df in basket.items():
            if not isinstance(table, str):
                raise TypeError(f"{name} keys must be table-name strings; got {type(table).__name__}.")
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"{name}['{table}'] must be a pandas DataFrame.")
            cleaned[table] = df.copy()
        if all(len(df) == 0 for df in cleaned.values()):
            raise ValueError(f"{name} contains no rows.")
        return cleaned

    @staticmethod
    def _validate_aging_rules(rules, name, *, required):
        """Validate the SourceColumn-style ``layer_configs`` aging rules.

        ``None`` becomes an empty dict (``apply_aging`` then no-ops), so a cohort
        with no rules is aged identically — i.e. not at all.
        """
        if rules is None:
            if required:
                raise TypeError(
                    f"{name} is required: SourceColumn-style layer_configs that apply_aging consumes."
                )
            return {}
        if not isinstance(rules, dict):
            raise TypeError(f"{name} must be a dict of SourceColumn-style layer configs (or None).")
        return rules

    @staticmethod
    def _validate_id_columns(id_columns, data_nb):
        """Validate the going-concern identifier re-keying spec and return a
        normalised copy.

        ``id_columns`` maps a logical key name to the list of ``(table, column)``
        places that key appears, so the same per-period token can be applied to
        every occurrence and keep foreign keys joinable. ``None`` becomes ``{}``
        (no re-keying). Each ``(table, column)`` must name a real column in
        ``data_nb`` so typos fail fast at construction.
        """
        if id_columns is None:
            return {}
        if not isinstance(id_columns, dict):
            raise TypeError("id_columns must be a dict {logical_key: [(table, column), ...]} or None.")
        out = {}
        for key, occurrences in id_columns.items():
            if not isinstance(key, str) or not key:
                raise TypeError("id_columns keys must be non-empty strings (logical key names).")
            if not isinstance(occurrences, (list, tuple)) or not occurrences:
                raise TypeError(f"id_columns['{key}'] must be a non-empty list of (table, column) pairs.")
            cleaned = []
            for occ in occurrences:
                if not (isinstance(occ, (list, tuple)) and len(occ) == 2):
                    raise TypeError(f"id_columns['{key}'] entries must be (table, column) pairs; got {occ!r}.")
                table, column = occ
                if not isinstance(table, str) or not table or not isinstance(column, str) or not column:
                    raise TypeError(f"id_columns['{key}'] entries must be (table, column) string pairs; got {occ!r}.")
                if table not in data_nb:
                    raise ValueError(
                        f"id_columns['{key}'] references table '{table}' not in data_nb {sorted(data_nb)}."
                    )
                if column not in data_nb[table].columns:
                    raise ValueError(
                        f"id_columns['{key}'] references column '{column}' not found on data_nb['{table}']."
                    )
                cleaned.append((table, column))
            out[key] = cleaned
        return out

    # --------------------------------------------------------------------- #
    # Aging mechanics
    # --------------------------------------------------------------------- #

    def _age_one_step(self, basket, aging_rules):
        """Advance a basket one year by applying the one-step ``aging_rules`` via
        forge's shared :func:`apply_aging`.

        The input ``basket`` is not mutated: each frame is copied before
        ``apply_aging`` writes the aged columns in place. ``aging_rules`` is a
        SourceColumn-style ``layer_configs`` dict; columns its expressions read
        (e.g. ``gwp`` for ``previous_gwp ← gwp``) must be present on the basket.
        """
        aged = {table: df.copy() for table, df in basket.items()}
        return apply_aging(aging_rules, aged)

    def _fresh_nb(self, period):
        """The new-business cohort injected in ``period``: a fresh, un-aged draw
        from the NB population (a copy of ``data_nb``).

        With no NB aging rules the cohort is identical every period (fully
        un-aged). If date-shift NB rules are supplied they are applied ``period``
        times so the fresh term lands in the right calendar year (design §9.1) —
        the demographics still come straight from ``data_nb``, never compounded
        from a prior period.
        """
        nb = {table: df.copy() for table, df in self.input.data_nb.items()}
        if self.parameters.aging_rules_nb:
            for _ in range(period):
                nb = apply_aging(self.parameters.aging_rules_nb, nb)
        return self._rekey_fresh_nb(nb, period)

    def _rekey_fresh_nb(self, nb, period):
        """Make a freshly-injected new-business cohort's identifiers globally
        unique by appending a per-period token to every configured id column.

        Going concern re-injects a copy of ``data_nb`` each period, so without
        this the fresh cohort shares ``policy_number`` / ``risk_number`` with the
        carried (graduated) copies of the same policies. Any policy→risk
        aggregation/join keyed on those ids would then attribute the fresh
        new-business attributes (e.g. ``transaction_type == 'NB'``) to the
        carried renewals. Applying the *same* token to every occurrence of a
        logical key keeps foreign keys joinable within the cohort while making
        the whole cohort distinct from carried business. Period 0 is left
        untouched (NB and RN ids are already disjoint there), which also
        preserves source-id traceability for the seed book.
        """
        if not self.parameters.id_columns or period == 0:
            return nb
        token = f"@nb{period}"
        for occurrences in self.parameters.id_columns.values():
            for table, column in occurrences:
                if table in nb and column in nb[table].columns:
                    nb[table][column] = nb[table][column].astype(str) + token
        return nb

    @staticmethod
    def _concat_baskets(nb_basket, rn_basket):
        """Concatenate two baskets table-by-table into one book for a single
        period's scoring. Tables present in only one basket pass through; tables
        in both are row-concatenated (new business first, then the carried book).
        """
        tables = {}
        ordered = (*nb_basket, *(t for t in rn_basket if t not in nb_basket))
        for name in ordered:
            parts = [b[name] for b in (nb_basket, rn_basket) if name in b]
            tables[name] = pd.concat(parts, ignore_index=True)
        return tables

    def _age_book_trajectory(self, aging_rules=None, cohort_target=None):
        """Build the per-period book trajectory by the two-basket rule (§9.1).

        Period 0 is the seed book — new business ⊕ in-force renewals, both
        un-aged. Each subsequent period ages the *whole* previous book one step
        (the carried renewal book: last period's converted NB graduates to ``RN``
        and ages alongside the prior renewals), then — only in
        ``"going_concern"`` mode — re-injects a fresh, un-aged new-business
        cohort. In ``"runoff"`` mode no new business is written after period 0.

        ``aging_rules`` defaults to the full carried-book rules; the performance
        path passes the demographic-only subset so the trajectory is
        x-independent (design §13.1). ``cohort_target``, if given, tags that
        table's rows ``__cohort = "nb" | "rn"`` (so the forward sim threads the
        feedback / survivorship onto renewals only) and a stable ``__policy_id``
        assigned at birth (so Monte-Carlo can pivot per policy across periods —
        ids ride through ``apply_aging`` untouched, so a carried policy keeps its
        birth id as it ages).

        Returns a list of ``n_periods`` baskets (each ``dict[str, DataFrame]``).
        """
        if aging_rules is None:
            aging_rules = self.parameters.aging_rules_rn

        next_id = [0]  # running birth counter -> unique, stable per-policy ids

        def _newborn(basket, cohort):
            """Tag a freshly-entering cohort with its cohort label and new ids."""
            if cohort_target is not None and cohort_target in basket:
                df = basket[cohort_target].copy()
                df[_COHORT_COL] = cohort
                df[_POLICY_ID_COL] = np.arange(next_id[0], next_id[0] + len(df))
                next_id[0] += len(df)
                basket[cohort_target] = df
            return basket

        def _carry(basket):
            """Re-label the aged carry as renewal, keeping each policy's birth id."""
            if cohort_target is not None and cohort_target in basket:
                df = basket[cohort_target].copy()
                df[_COHORT_COL] = "rn"
                basket[cohort_target] = df
            return basket

        rn_seed = _newborn({tbl: df.copy() for tbl, df in self.input.data_rn.items()}, "rn")
        baskets = [self._concat_baskets(_newborn(self._fresh_nb(0), "nb"), rn_seed)]
        for t in range(1, self.parameters.n_periods):
            aged_carry = _carry(self._age_one_step(baskets[t - 1], aging_rules))
            if self.parameters.new_business_mode == "going_concern":
                baskets.append(self._concat_baskets(_newborn(self._fresh_nb(t), "nb"), aged_carry))
            else:  # runoff — no new business written after period 0
                baskets.append(aged_carry)
        return baskets

    def _demographic_trajectory(self):
        """Cached, x-independent demographic book trajectory (design §13.1).

        Built once with the **demographic-only** aging rules (the
        ``previous_gwp ← gwp`` feedback excluded) and tagged with ``__cohort`` so
        renewals are identifiable. The per-evaluation forward sim threads the
        x-dependent ``previous_gwp`` and survivorship weights onto this fixed
        backbone, so aging is paid once rather than on every objective call.
        """
        if self._demo_trajectory_cache is None:
            self._demo_trajectory_cache = self._age_book_trajectory(
                aging_rules=self._demographic_aging_rules,
                cohort_target=self._layer_meta["target_table"],
            )
        return self._demo_trajectory_cache

    # --------------------------------------------------------------------- #
    # Overlay injection + forward simulation
    # --------------------------------------------------------------------- #

    def _apply_overlay(self, x, pipeline):
        """Write the per-category multipliers ``x`` into the demand layer's
        rating tables on ``pipeline`` (mutated in place — the caller passes a
        deep copy), then return the pipeline ready to ``predict``.

        Each rating table's ``value`` column is replaced positionally in
        ``category_orders`` order, which is exactly the table's row order
        (``_parse_rating_layer`` built the categories from the table rows), so
        category *c*'s multiplier lands on table row *c*. Rows whose category is
        a fixed default take ``default_contrib`` via the table's
        ``default_values``.
        """
        layer = pipeline.layers[self._layer_meta["name"]]
        multipliers = self._unflatten(x)
        for factor in self.input.factor_names:
            cats = self.input.category_orders[factor]
            layer.rating_tables[factor]["value"] = np.array(
                [multipliers[factor][c] for c in cats], dtype=float
            )
            if factor in self.input.default_contrib:
                layer.default_values[factor] = float(self.input.default_contrib[factor])
        return pipeline

    def _simulate_horizon(self, x):
        """Forward-simulate the horizon under overlay ``x`` and return the
        concatenated per-period scored target table (tagged with ``__period``).

        Performance structure (design §13): the x-independent demographic book
        trajectory is precomputed once (:meth:`_demographic_trajectory`) and the
        single shared overlay ``μ(x)`` is injected into **one** deep-copied
        pipeline and reused for every period. Per period only the x-dependent tail
        is threaded onto the cached book — carried renewals inherit
        ``previous_gwp`` from the premium charged last period (the feedback
        hand-off) and a survivorship weight scaled by conversion (design §10.1),
        while fresh new business keeps its seed ``previous_gwp`` and weight 1 —
        then the book is scored.
        """
        if not hasattr(self, "pipeline"):
            raise RuntimeError("_simulate_horizon requires an optimizer built via from_pipeline().")

        target = self._layer_meta["target_table"]
        w_col = self.parameters.sample_weight_col
        conv_col = self.parameters.conversion_col
        gwp_col = self.parameters.gwp_col
        pgwp_col = self.parameters.previous_gwp_col

        trajectory = self._demographic_trajectory()                  # x-independent, cached
        overlaid = self._apply_overlay(x, deepcopy(self.pipeline))   # μ(x) injected once, reused

        scored_periods = []
        prev_gwp = None       # last period's scored gwp (all rows, in carry order)
        survivorship = None   # last period's conversion × weight (all rows)
        for t, demo in enumerate(trajectory):
            book = {tbl: df.copy() for tbl, df in demo.items()}
            tgt = book[target]
            if w_col not in tgt.columns:
                tgt[w_col] = 1.0
            if t > 0 and self.parameters.nb_growth_rate:
                # Going-concern volume growth: scale this period's fresh
                # new-business cohort by (1 + g)^t (design §10.3). Carried
                # renewals already embed prior growth via survivorship.
                nb_mask = (tgt[_COHORT_COL] == "nb").to_numpy()
                if nb_mask.any():
                    tgt.loc[nb_mask, w_col] = (
                        tgt.loc[nb_mask, w_col] * (1.0 + self.parameters.nb_growth_rate) ** t
                    )
            if t > 0:
                # The carried renewals are the aged previous-period book, in the
                # same row order, so the feedback / survivorship align positionally.
                rn_mask = (tgt[_COHORT_COL] == "rn").to_numpy()
                tgt.loc[rn_mask, pgwp_col] = prev_gwp
                tgt.loc[rn_mask, w_col] = survivorship

            scored = overlaid.predict(book)
            scored_target = scored[target].copy()
            scored_target[_PERIOD_COL] = t
            scored_periods.append(scored_target)

            prev_gwp = scored[target][gwp_col].to_numpy()
            survivorship = scored[target][conv_col].to_numpy() * scored[target][w_col].to_numpy()

        return pd.concat(scored_periods, ignore_index=True)

    def _simulate_and_aggregate(self, x):
        """Run :meth:`_simulate_horizon` and reduce it to NPV / volume /
        lifetime-conversion plus the per-row ``expected_npv``-annotated frame."""
        df = self._simulate_horizon(x)
        w = df[self.parameters.sample_weight_col].to_numpy(dtype=float)
        conv = df[self.parameters.conversion_col].to_numpy(dtype=float)
        profit = df[self.parameters.profit_col].to_numpy(dtype=float)
        period = df[_PERIOD_COL].to_numpy(dtype=float)

        discount = (1.0 + self.parameters.discount_rate) ** (period + self.parameters.timing_offset)
        expected_npv = w * profit / discount

        npv = float(np.sum(expected_npv))
        volume = float(np.sum(w * conv))
        weight_total = float(np.sum(w))
        constraint = volume / weight_total if weight_total > 0 else 0.0

        df = df.assign(expected_npv=expected_npv)
        return SimpleNamespace(
            df=df, npv=npv, volume=volume, weight_total=weight_total, constraint=constraint
        )

    def _per_period_summary(self, agg):
        """Per-period breakdown (conversion, volume, discounted NPV, mean
        overlay multiplier) — the per-period rows the design's return shape
        promises (§14). Shared by both optimizers: each passes the ``agg`` from a
        forward sim at its chosen overlay (RF at the SLSQP optimum, Individual at
        the per-policy λ optimum / a uniform baseline)."""
        w_col = self.parameters.sample_weight_col
        conv_col = self.parameters.conversion_col
        out_col = self._layer_meta["output_column"]

        rows = []
        for period, g in agg.df.groupby(_PERIOD_COL):
            w = g[w_col].to_numpy(dtype=float)
            conv = g[conv_col].to_numpy(dtype=float)
            wsum = w.sum()
            rn = g[g[_COHORT_COL] == "rn"]
            nb = g[g[_COHORT_COL] == "nb"]
            rn_w = rn[w_col].to_numpy(dtype=float)
            rows.append({
                "period": int(period),
                "conversion": float(np.average(conv, weights=w)) if wsum > 0 else float("nan"),
                "volume": float(np.sum(w * conv)),
                "weight": float(wsum),
                # Book composition (design §10.3): fresh new-business inflow vs the
                # carried renewal stock entering the period.
                "new_business_weight": float(nb[w_col].sum()),
                "renewal_weight": float(rn_w.sum()),
                "npv": float(g["expected_npv"].sum()),
                "price_multiplier": float(np.average(g[out_col], weights=w)) if wsum > 0 else float("nan"),
                # Renewal-only overlay — the quantity the ENBP ceiling bounds (§12).
                "renewal_price_multiplier": (
                    float(np.average(rn[out_col], weights=rn_w)) if rn_w.sum() > 0 else float("nan")
                ),
            })
        return pd.DataFrame(rows)

    # --------------------------------------------------------------------- #
    # Monte-Carlo survivorship (post-hoc risk reporting, design §10.2)
    # --------------------------------------------------------------------- #

    def _build_mc_inputs(self, x):
        """From ONE deterministic ``_simulate_horizon(x)`` build the per-policy ×
        per-period matrices Monte-Carlo needs: conversion ``P[i,t]``, discounted
        offered margin ``OFF_over_disc[i,t] = (gwp - cost) / (1+r)^(t+δ)``, and the
        birth-weight vector ``weight0[i]``.

        These are deterministic for fixed ``x`` (aging + ``previous_gwp`` feedback
        are deterministic; conversion is per-policy with no cross-policy coupling),
        so MC re-samples Bernoulli survival on top of them without re-running
        ``predict`` per simulation.
        """
        df = self._simulate_horizon(x)
        w_col = self.parameters.sample_weight_col
        conv_col = self.parameters.conversion_col
        offered = df[self.parameters.gwp_col].to_numpy(dtype=float) - df[self.parameters.cost_col].to_numpy(dtype=float)
        df = df.assign(__offered=offered)

        n_periods = self.parameters.n_periods
        disc = (1.0 + self.parameters.discount_rate) ** (np.arange(n_periods) + self.parameters.timing_offset)

        if self.parameters.new_business_mode == "going_concern":
            # Rows grow as fresh NB is injected, so align per policy by stable id.
            P = df.pivot(index=_POLICY_ID_COL, columns=_PERIOD_COL, values=conv_col).reindex(columns=range(n_periods))
            OFF = df.pivot(index=_POLICY_ID_COL, columns=_PERIOD_COL, values="__offered").reindex(columns=range(n_periods))
            # Birth weight = the weight when each policy first appears.
            weight0 = (
                df.sort_values(_PERIOD_COL).groupby(_POLICY_ID_COL)[w_col].first().reindex(P.index).to_numpy(dtype=float)
            )
            # Before a policy is born: p→1 (identity in the survival product) and
            # offered→0 (no contribution), so the inclusive cumprod stays correct.
            P = np.where(np.isnan(P.to_numpy(dtype=float)), 1.0, P.to_numpy(dtype=float))
            OFF = np.where(np.isnan(OFF.to_numpy(dtype=float)), 0.0, OFF.to_numpy(dtype=float))
        else:
            # Runoff: a constant book aged in place — rows align positionally.
            per_period = [df[df[_PERIOD_COL] == t] for t in range(n_periods)]
            P = np.column_stack([g[conv_col].to_numpy(dtype=float) for g in per_period])
            OFF = np.column_stack([g["__offered"].to_numpy(dtype=float) for g in per_period])
            weight0 = per_period[0][w_col].to_numpy(dtype=float)  # all born at period 0

        return P, OFF / disc[None, :], weight0

    @staticmethod
    def _simulate_mc_npv(P, off_over_disc, weight0, n_sims, rng, chunk_size):
        """Vectorised Bernoulli-survival NPV sampler.

        Per sim, draw ``B ~ Bernoulli(P)`` per policy/period, take the **inclusive**
        running product ``S = Π_{s≤t} B`` (a policy realises period ``t``'s margin
        only if it converted every period up to and including ``t``), and sum
        ``weight0 · S · off_over_disc``. Chunked over sims to bound the
        ``(chunk, N, T)`` array.
        """
        samples = np.empty(n_sims, dtype=float)
        done = 0
        while done < n_sims:
            c = min(chunk_size, n_sims - done)
            survive = rng.random((c, P.shape[0], P.shape[1])) < P[None, :, :]
            surv_prod = np.cumprod(survive, axis=2)  # inclusive Π_{s<=t}
            contrib = weight0[None, :, None] * surv_prod * off_over_disc[None, :, :]
            samples[done:done + c] = contrib.sum(axis=(1, 2))
            done += c
        return samples

    def monte_carlo_npv(
        self,
        optimal_multipliers=None,
        *,
        n_sims=1000,
        seed=None,
        percentiles=(5, 25, 50, 75, 95),
        var_alpha=0.05,
        chunk_size=200,
        return_samples=True,
        save_to_output=False,
    ):
        """Post-hoc Monte-Carlo of the horizon NPV under a fixed overlay (design §10.2).

        The deterministic ``optimize`` reports one *expected* NPV by carrying a
        fractional survivorship weight; this stress-tests a chosen overlay by
        sampling renew/lapse per policy per period (Bernoulli on conversion) over
        ``n_sims`` runs, yielding the NPV **distribution** and tail risk
        (VaR/CVaR). It is a reporting tool — it is **not** used inside the
        optimiser (the objective stays smooth/deterministic).

        Because the overlay is fixed, each policy's per-period conversion and
        offered margin are deterministic, so this runs the pipeline once and
        re-samples survival vectorised on top. By construction the **mean NPV
        converges to the deterministic NPV** (design §16.2); the spread/tails are
        the new information.

        Parameters
        ----------
        optimal_multipliers : dict[str, dict[str, float]], optional
            Overlay to evaluate, ``{factor: {category: multiplier}}``. Defaults to
            the last ``optimize`` result (``self.output.optimal_multipliers``).
        n_sims : int
            Number of Monte-Carlo runs.
        seed : int, optional
            Seeds ``numpy.random.default_rng`` for reproducibility (design R5).
        percentiles : tuple[float]
            Percentile levels (in ``[0, 100]``) to report.
        var_alpha : float
            Tail probability for VaR/CVaR (e.g. ``0.05`` → 5% VaR).
        chunk_size : int
            Simulations per vectorised batch (bounds peak memory).
        return_samples : bool
            If ``False``, omit the raw ``samples`` array from the result.
        save_to_output : bool
            If ``True``, also store the result on ``self.output.monte_carlo``.

        Returns
        -------
        SimpleNamespace
            ``samples`` (``np.ndarray`` of NPVs or ``None``), ``summary`` (one-row
            ``DataFrame``), ``percentiles`` (dict), ``var``, ``cvar``,
            ``deterministic_npv``.
        """
        if not hasattr(self, "pipeline"):
            raise RuntimeError("monte_carlo_npv requires an optimizer built via from_pipeline().")
        if optimal_multipliers is None:
            optimal_multipliers = getattr(self.output, "optimal_multipliers", None)
        if optimal_multipliers is None:
            raise RuntimeError(
                "monte_carlo_npv requires optimal_multipliers; run optimize() first or pass optimal_multipliers=..."
            )
        if not isinstance(n_sims, int) or isinstance(n_sims, bool) or n_sims < 1:
            raise ValueError("n_sims must be an integer >= 1.")
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
            raise ValueError("chunk_size must be an integer >= 1.")
        if not isinstance(var_alpha, (int, float)) or isinstance(var_alpha, bool) or not (0.0 < var_alpha < 1.0):
            raise ValueError("var_alpha must be a number in (0, 1).")
        percentiles = tuple(percentiles)
        if not percentiles or not all(
            isinstance(p, (int, float)) and not isinstance(p, bool) and 0 <= p <= 100 for p in percentiles
        ):
            raise ValueError("percentiles must be a non-empty sequence of numbers in [0, 100].")

        x = self._flatten(optimal_multipliers)
        deterministic_npv = self._simulate_and_aggregate(x).npv
        P, off_over_disc, weight0 = self._build_mc_inputs(x)
        rng = np.random.default_rng(seed)
        samples = self._simulate_mc_npv(P, off_over_disc, weight0, n_sims, rng, chunk_size)

        stats, pct, var, cvar = _npv_distribution_summary(samples, percentiles, var_alpha)
        summary = pd.DataFrame({
            "mean": [stats["mean"]],
            "std": [stats["std"]],
            "min": [stats["min"]],
            "max": [stats["max"]],
            "deterministic_npv": [deterministic_npv],
            "mean_minus_deterministic": [stats["mean"] - deterministic_npv],
            **{f"p{k}": [v] for k, v in pct.items()},
            f"var_{var_alpha}": [var],
            f"cvar_{var_alpha}": [cvar],
            "n_sims": [n_sims],
            "seed": [seed],
            "new_business_mode": [self.parameters.new_business_mode],
        })

        result = SimpleNamespace(
            samples=(samples if return_samples else None),
            summary=summary,
            percentiles=pct,
            var=var,
            cvar=cvar,
            deterministic_npv=deterministic_npv,
        )
        if save_to_output:
            self.output.monte_carlo = result
        return result

"""Multi-period rating-factor price optimizer.

Built in phases per ``docs/design/multi_period_price_optimizer.md`` §18.

- **Phase 2 — Aging.** The two-basket aging loop: the carried renewal book is
  aged one step per period (via forge's shared ``apply_aging``) while the
  new-business cohort is re-injected un-aged (``_age_one_step`` /
  ``_age_book_trajectory``).
- **Phase 3 — Forward simulation (current).** A single shared demand overlay is
  driven into the pipeline each period (``_apply_overlay``), the horizon is
  forward-simulated with survivorship by weight-carrying and NPV discounting
  (``_simulate_horizon``), and the overlay is optimized against the NPV of
  expected profit subject to a lifetime conversion constraint (``optimize`` /
  ``calculate_efficient_frontier``). With ``n_periods == 1`` and ``r == 0`` this
  reduces to :class:`PriceOptimizerRatingFactor` (design §16.1).

Why the pre-compute trick is abandoned (design §4): renewal conversion depends on
*year-on-year price change*, so the premium charged this period feeds next
period's ``previous_gwp``; the objective is a function of the whole *path* of
decisions, not of a per-row scalar. The only correct evaluation is to forward-
simulate the horizon for each candidate ``x`` — running the real pipeline each
period rather than interpolating a static grid.

Why aging is a *loop* and not a parameterised shift (design §9.2): following the
one-step aging pattern, each aging rule expresses **one**
year of change and the per-period loop does the compounding, so multi-year
effects fall out without any ``** periods`` term. The optimizer ages a book
through the same ``apply_aging`` the app uses (§9.4), so a strategy validated
here reproduces exactly when the app scores it.
"""
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:  # tqdm is already a dependency (the single-period optimizer uses it); guard
    # so a missing/odd install degrades the progress bar to text rather than crashing.
    from tqdm.auto import tqdm  # `.auto` picks the notebook widget or console bar
except Exception:  # pragma: no cover - exercised only when tqdm is unavailable
    tqdm = None

from ._horizon_engine import (
    _COHORT_COL,
    _MultiPeriodHorizonEngine,
    _PERIOD_COL,
)

__all__ = ["PriceOptimizerRatingFactorMultiPeriod", "make_log_multiplier_regularizer"]


# NPV-of-profit objective variants (design §11.1).
_OBJECTIVE_TYPES = ("total_expected_profit", "average_expected_profit")

# Per-period conversion constraint kinds (design §11.2). Also the public vocabulary
# for the lifetime ``constraint_kind`` ("target" == equality, "floor" == >=,
# "ceiling" == <=); each maps to the internal "eq"/"ineq" spec kind.
_PER_PERIOD_KINDS = ("target", "floor", "ceiling")

# Aggregate metrics a pluggable objective or lifetime constraint can target. Each
# maps the per-x simulation namespace from ``_simulate_and_aggregate`` (fields
# ``npv``, ``volume``, ``weight_total``, ``constraint``) to a scalar, so the
# objective and the lifetime constraint draw from one shared menu. A user may pass
# any of these names or a callable ``agg -> float`` (the escape hatch).
_AGG_METRICS = {
    "total_expected_profit": lambda agg: agg.npv,
    "average_expected_profit": lambda agg: agg.npv / agg.volume if agg.volume > 0 else 0.0,
    "total_volume": lambda agg: agg.volume,
    "conversion": lambda agg: agg.constraint,
    "total_weight": lambda agg: agg.weight_total,
}


def make_log_multiplier_regularizer(l1_penalty=0.0, l2_penalty=0.0):
    """Build a ``regularizer`` callable that penalizes the log of each decision
    multiplier (design §11.3): ``Σ_k l1·|log μ_k| + l2·(log μ_k)²`` over every
    category of every factor.

    Pass the result as ``regularizer=...`` to :class:`PriceOptimizerMultiPeriod`
    (or :class:`PriceOptimizerRatingFactor`) to keep the overlay gentle and close
    to 1.0 — L1 encourages exact-1.0 (no movement) sparsity, L2 shrinks large
    moves. The penalty is symmetric in ``log μ`` so a markup and the reciprocal
    discount are penalized equally.
    """
    if l1_penalty < 0 or l2_penalty < 0:
        raise ValueError("l1_penalty and l2_penalty must be non-negative.")

    def _regularizer(multipliers):
        total = 0.0
        for cats in multipliers.values():
            for value in cats.values():
                log_m = np.log(value)
                total += l1_penalty * abs(log_m) + l2_penalty * log_m * log_m
        return total

    return _regularizer


class _ProgressReporter:
    """Live readout for the ``optimize`` solve loop so a long run is visibly
    progressing rather than apparently frozen.

    ``scipy.optimize.minimize`` cannot report a total up front, so the bar is an
    **indeterminate** counter that ticks once per **forward simulation** — the
    expensive unit, and the truest "still running?" pulse, because SLSQP runs many
    simulations *inside* one iteration (finite-difference gradients) during which a
    per-iteration bar would sit still. A coarse iteration counter (from
    ``minimize``'s ``callback``) rides in the bar description.

    mode:
      ``"bar"``  — a tqdm bar with live ``obj``/``conv``/``tgt``/``npv`` postfix.
      ``"text"`` — one printed stats line per solver iteration (log-file friendly).
    """

    def __init__(self, mode, *, target, raw_objective, weight_col, out_col,
                 constraint_name="conv", constraint_fn=None, label="optimize"):
        self.mode = mode
        self.target = target
        self._raw_objective = raw_objective
        self._weight_col = weight_col
        self._out_col = out_col
        # The constrained quantity shown next to ``tgt``. Defaults reproduce the
        # legacy conversion readout byte-for-byte ("conv=<agg.constraint>"); a
        # non-conversion metric supplies its own label + extractor.
        self._constraint_name = constraint_name
        self._constraint_fn = constraint_fn if constraint_fn is not None else (lambda agg: agg.constraint)
        self._n_iter = 0
        self._last_agg = None
        self._penalty = None  # latest w·Σr² from the penalty-method objective (if any)
        # tqdm-unavailability is resolved to "text" before construction, so by here
        # a "bar" mode guarantees tqdm is importable.
        self._bar = tqdm(total=None, desc=label, unit="sim", leave=True) if mode == "bar" else None

    def _postfix(self, agg):
        """Build the bar postfix dict. ``pen`` (the constraint penalty added to the
        minimised objective) appears only once a penalty has been reported — so
        SLSQP, which enforces constraints exactly with no penalty term, omits it."""
        d = {
            "obj": f"{self._raw_objective(agg):.4g}",
            self._constraint_name: f"{self._constraint_fn(agg):.4f}",
            "tgt": f"{self.target:.4f}",
            "npv": f"{agg.npv:.4g}",
        }
        if self._penalty is not None:
            d["pen"] = f"{self._penalty:.4g}"
        return d

    def tick(self, agg):
        """Called on every real forward simulation (a ``_simulate_cached`` miss)."""
        self._last_agg = agg
        if self._bar is not None:
            self._bar.set_postfix(self._postfix(agg), refresh=False)
            self._bar.update(1)  # refreshes, showing the postfix just set

    def set_penalty(self, penalty):
        """Record the constraint penalty (``w·Σr²``) the penalty-method objective
        added at the latest evaluated ``x`` and refresh the bar so it shows against
        the current sim's stats. Display-only — never affects the optimum."""
        self._penalty = penalty
        if self._bar is not None and self._last_agg is not None:
            self._bar.set_postfix(self._postfix(self._last_agg), refresh=True)

    def iteration(self, xk=None, *args, **kwargs):
        """``scipy`` ``callback``: fires once per solver iteration. Tolerant of
        scipy passing either the parameter vector or an ``OptimizeResult`` (only
        the latest simulated stats are used, not the argument)."""
        self._n_iter += 1
        if self._bar is not None:
            self._bar.set_description(f"optimize (iter {self._n_iter})")
            return
        agg = self._last_agg
        if self.mode == "text" and agg is not None:
            line = (
                f"iter {self._n_iter}  "
                f"npv={agg.npv:.6g}  "
                f"obj={self._raw_objective(agg):.6g}  "
                f"{self._constraint_name}={self._constraint_fn(agg):.4f}  "
                f"tgt={self.target:.4f}  "
                f"pmult={self._pmult(agg):.4f}"
            )
            if self._penalty is not None:
                line += f"  pen={self._penalty:.6g}"
            print(line)

    def _pmult(self, agg):
        """Weighted-average realised overlay multiplier (mirrors ``_build_summary``)."""
        df = agg.df
        w = df[self._weight_col].to_numpy(dtype=float)
        if w.sum() <= 0:
            return float("nan")
        return float(np.average(df[self._out_col], weights=w))

    def close(self):
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def _resolve_progress(progress):
    """Normalise the ``progress`` argument to ``None`` / ``"bar"`` / ``"text"``.

    ``True`` ⇒ ``"bar"``; ``False``/``None`` ⇒ silent; a ``"bar"`` request with
    tqdm unavailable degrades to ``"text"`` (with a warning) rather than failing.
    """
    if progress is None or progress is False:
        return None
    if progress is True:
        progress = "bar"
    if progress not in ("bar", "text"):
        raise ValueError(f"progress must be one of True, False, 'bar', 'text'; got {progress!r}.")
    if progress == "bar" and tqdm is None:
        warnings.warn("tqdm is not available; falling back to progress='text'.", stacklevel=2)
        return "text"
    return progress


class PriceOptimizerRatingFactorMultiPeriod(_MultiPeriodHorizonEngine):
    """Optimize a deployable demand overlay against the multi-period NPV of
    expected profit, modelling renewal price-change feedback, aging, and
    survivorship by forward-simulating the horizon.

    The horizon-simulation engine (aging, overlay injection, survivorship, the
    ``previous_gwp`` feedback, NPV, and Monte-Carlo risk) is inherited from
    :class:`_MultiPeriodHorizonEngine`, which in turn builds on
    :class:`_RatingFactorDecisionSpace` for the decision-variable machinery
    (flatten/unflatten, bounds, ``from_pipeline`` layer parsing,
    ``to_rating_tables`` / ``to_layer`` deployment). This class adds the dense
    ``scipy`` objective/constraint solve on top, in place of the single-period
    grid interpolation.

    Construct via :meth:`from_pipeline` (which derives the decision space from a
    multiplicative ``RatingTablesLayer`` and stores the pipeline the horizon is
    scored on). The bare ``__init__`` validates and stores the baskets, aging
    rules, and horizon configuration; it does not wire the decision space, so
    ``optimize`` requires ``from_pipeline``.

    Parameters
    ----------
    data_nb, data_rn : dict[str, pandas.DataFrame]
        New-business and in-force/renewal baskets in pipeline-state shape (the
        same ``{table: DataFrame}`` mapping ``pipeline.predict`` consumes).
        ``data_rn`` seeds the carried renewal book.
    n_periods : int
        Number of renewal periods to simulate (``>= 1``).
    aging_rules_rn : dict[str, dict]
        One-step aging rules for the **carried book**, as SourceColumn-style
        ``layer_configs`` that :func:`apply_aging` consumes (design §9.4).
    aging_rules_nb : dict[str, dict], optional
        Aging rules for the new-business cohort. Normally ``None`` (NB is
        re-injected un-aged); supply date-shift-only rules to land each period's
        fresh NB term in the right calendar year (design §9.1).
    new_business_mode : {"runoff", "going_concern"}, keyword-only
        New-business inflow framing (design §10.3). Defaults to ``"runoff"``.
    discount_rate, timing_offset : float, keyword-only
        NPV discount rate ``r`` and mid-period timing offset ``δ`` (design §7).
    nb_growth_rate : float, keyword-only
        Per-period growth of the fresh new-business cohort under
        ``"going_concern"`` (design §10.3). Period ``t``'s injected NB volume is
        scaled by ``(1 + nb_growth_rate) ** t`` (``0.0`` ⇒ steady-state). No
        effect under ``"runoff"`` (no NB after period 0). Premium *inflation* is
        a separate lever — express it as a one-year aging rule on the base
        premium rather than here.
    objective_type : str or callable, keyword-only
        The quantity to **maximise**. A name from the metric menu
        (``"total_expected_profit"`` — discounted **sum** of expected profit,
        portfolio growth; ``"average_expected_profit"`` — per-converted-policy
        discounted average, margin focus; plus ``"total_volume"``,
        ``"conversion"``, ``"total_weight"``) or a callable ``agg -> float`` on the
        simulation namespace (fields ``npv``, ``volume``, ``weight_total``,
        ``constraint``) for anything off-menu. Defaults to
        ``"total_expected_profit"``; any ``regularizer`` is subtracted from it.
    regularizer : callable, optional, keyword-only
        ``multipliers_dict -> float`` penalty subtracted from the objective each
        evaluation (e.g. L1/L2 on ``log(multiplier)``); mirrors the
        single-period ``regularizer`` hook.
    sample_weight_col, conversion_col, profit_col, gwp_col, cost_col : str, keyword-only
        Names of the scored columns the objective reads on the overlay's target
        table: the survivorship weight, the conversion/retention probability,
        the per-row **expected** profit (offered profit × conversion), and —
        for reporting only — the offered premium and technical cost.
    """

    _SUPPORTED_METHODS = ("SLSQP", "L-BFGS-B", "Nelder-Mead")

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
        objective_type="total_expected_profit",
        regularizer=None,
        sample_weight_col="sample_weights",
        conversion_col="conversion_rate",
        profit_col="reporting_expected_profit",
        gwp_col="gwp",
        cost_col="reporting_offered_technical_premium",
        previous_gwp_col="previous_gwp",
        id_columns=None,
    ):
        # The shared horizon engine validates and stores the baskets, aging
        # rules, horizon config, and column names (and builds the cached
        # x-independent demographic trajectory).
        super().__init__(
            data_nb,
            data_rn,
            n_periods,
            aging_rules_rn,
            aging_rules_nb,
            new_business_mode=new_business_mode,
            discount_rate=discount_rate,
            timing_offset=timing_offset,
            nb_growth_rate=nb_growth_rate,
            sample_weight_col=sample_weight_col,
            conversion_col=conversion_col,
            profit_col=profit_col,
            gwp_col=gwp_col,
            cost_col=cost_col,
            previous_gwp_col=previous_gwp_col,
            id_columns=id_columns,
        )

        # The objective metric and regularizer are specific to the dense scipy
        # solve this class bolts onto the shared engine; append them to the
        # engine-built ``self.parameters``.
        objective_fn, objective_label = self._resolve_metric(objective_type, "objective_type")
        if regularizer is not None and not callable(regularizer):
            raise TypeError("regularizer must be None or callable.")
        self.parameters.objective_type = objective_type
        self.parameters.objective_fn = objective_fn
        self.parameters.objective_label = objective_label
        self.parameters.regularizer = regularizer

    # --------------------------------------------------------------------- #
    # Construction from a pipeline
    # --------------------------------------------------------------------- #

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
        """Build a multi-period optimizer whose decision space is derived from a
        multiplicative ``RatingTablesLayer`` registered on ``pipeline``.

        The layer is parsed (shared with the single-period optimizer) from the
        scored **renewal** book; new business shares the same overlay. The live
        ``pipeline`` is stored — the forward simulation deep-copies it, writes the
        candidate overlay into its rating tables, and scores each period's book.

        Parameters
        ----------
        pipeline : PyfuncPricingPipeline
            Pipeline holding the demand-overlay layer; scored each period.
        layer_name : str
            Name of the multiplicative ``RatingTablesLayer`` to optimize.
        scored_data_nb, scored_data_rn : dict[str, pandas.DataFrame]
            ``pipeline.predict(...)`` output for the NB and RN baskets — used to
            read the layer's key column(s) and seed the decision space.
        data_nb, data_rn, n_periods, aging_rules_rn, aging_rules_nb
            Forwarded to ``__init__`` (the raw baskets and aging configuration).
        **kwargs
            Forwarded to ``__init__`` (``new_business_mode``, ``discount_rate``,
            ``objective_type``, column names, etc.).
        """
        parsed = cls._parse_rating_layer(pipeline, layer_name, scored_data_rn)

        obj = cls(
            data_nb=data_nb,
            data_rn=data_rn,
            n_periods=n_periods,
            aging_rules_rn=aging_rules_rn,
            aging_rules_nb=aging_rules_nb,
            **kwargs,
        )

        target = parsed["layer_meta"]["target_table"]
        n_rows = len(scored_data_rn[target])
        decision = obj._compute_decision_space(
            parsed["rating_factors"],
            parsed["initial_multipliers"],
            parsed["default_multipliers"],
            n_rows=n_rows,
        )
        for key, value in decision.items():
            setattr(obj.input, key, value)

        obj.pipeline = pipeline
        obj._layer_schema = parsed["layer_schema"]
        obj._layer_meta = parsed["layer_meta"]
        return obj

    # --------------------------------------------------------------------- #
    # Forward-sim caching + objective / constraint (the scipy solve)
    # --------------------------------------------------------------------- #

    def _simulate_cached(self, x):
        """Cache the most recent simulation keyed on the exact ``x`` so the
        objective and constraint (called at the same ``x`` by SLSQP) score the
        horizon once, not twice."""
        x = np.asarray(x, dtype=float)
        key = x.tobytes()
        if getattr(self, "_sim_key", None) != key:
            self._sim = self._simulate_and_aggregate(x)
            self._sim_key = key
            # Pulse the progress readout once per *real* forward sim (cache misses
            # only) — objective + constraint share a cache hit at the same x, so
            # this counts true work, not solver bookkeeping. No-op when off.
            progress = getattr(self, "_progress", None)
            if progress is not None:
                progress.tick(self._sim)
        return self._sim

    @staticmethod
    def _resolve_metric(spec, what):
        """Resolve an objective/constraint metric ``spec`` to ``(fn, label)``.

        ``spec`` is either a name from :data:`_AGG_METRICS` or a callable
        ``agg -> float`` (the escape hatch for anything off-menu). ``what`` names
        the parameter so the two call sites raise distinct messages. The returned
        ``label`` is the name (for a string) or the callable's ``__name__`` (or
        ``"custom"`` for a lambda) — used for the summary's reporting columns.
        """
        if callable(spec):
            return spec, getattr(spec, "__name__", "custom")
        if isinstance(spec, str) and spec in _AGG_METRICS:
            return _AGG_METRICS[spec], spec
        if isinstance(spec, str):
            raise ValueError(
                f"{what} must be one of {sorted(_AGG_METRICS)} or a callable; got {spec!r}."
            )
        raise TypeError(
            f"{what} must be a metric name (str) or a callable agg->float; got {type(spec).__name__}."
        )

    def _raw_objective(self, agg):
        return self.parameters.objective_fn(agg)

    def _objective_fn(self, x):
        agg = self._simulate_cached(x)
        score = self._raw_objective(agg)
        if self.parameters.regularizer is not None:
            score = score - float(self.parameters.regularizer(self._unflatten(x)))
        return -score

    def _constraint_value(self, x):
        return self._simulate_cached(x).constraint

    def _period_conversion_value(self, x, period):
        """Weighted-average conversion within a single period (design §11.2)."""
        df = self._simulate_cached(x).df
        sub = df[df[_PERIOD_COL] == period]
        w = sub[self.parameters.sample_weight_col].to_numpy(dtype=float)
        if w.sum() <= 0:
            return 0.0
        conv = sub[self.parameters.conversion_col].to_numpy(dtype=float)
        return float(np.average(conv, weights=w))

    def _period_renewal_overlay_value(self, x, period):
        """Weighted-average realised demand-overlay multiplier over the *renewal*
        rows of a single period — the renewal price relative to the technical
        (new-business-equivalent) premium, which the ENBP ceiling bounds (§12).

        Returns 0.0 for a period with no renewal rows, so the ``ceiling - value``
        residual is trivially satisfied there.
        """
        df = self._simulate_cached(x).df
        out_col = self._layer_meta["output_column"]
        sub = df[(df[_PERIOD_COL] == period) & (df[_COHORT_COL] == "rn")]
        w = sub[self.parameters.sample_weight_col].to_numpy(dtype=float)
        if w.sum() <= 0:
            return 0.0
        return float(np.average(sub[out_col].to_numpy(dtype=float), weights=w))

    # --------------------------------------------------------------------- #
    # Constraint assembly
    # --------------------------------------------------------------------- #

    def _validate_per_period_targets(self, per_period_targets):
        """Validate ``{period: (kind, value)}`` per-period conversion constraints
        and return a normalised copy (``value`` cast to float)."""
        if per_period_targets is None:
            return {}
        if not isinstance(per_period_targets, dict):
            raise TypeError("per_period_targets must be a dict {period: (kind, value)} or None.")
        out = {}
        for period, spec in per_period_targets.items():
            if not isinstance(period, int) or isinstance(period, bool) or not (0 <= period < self.parameters.n_periods):
                raise ValueError(
                    f"per_period_targets period {period!r} must be an int in [0, {self.parameters.n_periods})."
                )
            if not (isinstance(spec, tuple) and len(spec) == 2):
                raise TypeError(f"per_period_targets[{period}] must be a (kind, value) tuple.")
            kind, value = spec
            if kind not in _PER_PERIOD_KINDS:
                raise ValueError(
                    f"per_period_targets[{period}] kind must be one of {list(_PER_PERIOD_KINDS)}; got {kind!r}."
                )
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"per_period_targets[{period}] value must be numeric.")
            out[period] = (kind, float(value))
        return out

    def _build_constraint_specs(self, target_constraint, metric_fn, constraint_kind, enbp_ceiling, per_period_targets):
        """Assemble the constraint set as ``(kind, residual_fn)`` pairs, where the
        convention is ``residual == 0`` (``"eq"``) or ``residual >= 0`` (``"ineq"``).

        SLSQP consumes these as constraint dicts directly; the penalty methods
        fold them into squared / squared-hinge penalties. Comprises the lifetime
        constraint on ``metric_fn`` — ``constraint_kind`` is ``"target"``
        (equality), ``"floor"`` (``metric >= target``), or ``"ceiling"``
        (``metric <= target``) — any per-period conversion targets/floors/ceilings,
        and — if set — a per-period ENBP ceiling on the renewal overlay.
        ``metric_fn`` reads the shared per-x sim cache, so it costs no extra
        simulation alongside the objective evaluated at the same ``x``.
        """
        if constraint_kind == "floor":
            specs = [("ineq", lambda x, t=target_constraint, m=metric_fn: m(self._simulate_cached(x)) - t)]
        elif constraint_kind == "ceiling":
            specs = [("ineq", lambda x, t=target_constraint, m=metric_fn: t - m(self._simulate_cached(x)))]
        else:  # "target" == equality
            specs = [("eq", lambda x, t=target_constraint, m=metric_fn: m(self._simulate_cached(x)) - t)]

        for period, (kind, value) in per_period_targets.items():
            if kind == "target":
                specs.append(("eq", lambda x, p=period, v=value: self._period_conversion_value(x, p) - v))
            elif kind == "floor":
                specs.append(("ineq", lambda x, p=period, v=value: self._period_conversion_value(x, p) - v))
            else:  # ceiling
                specs.append(("ineq", lambda x, p=period, v=value: v - self._period_conversion_value(x, p)))

        if enbp_ceiling is not None:
            ceiling = float(enbp_ceiling)
            for period in range(self.parameters.n_periods):
                specs.append((
                    "ineq",
                    lambda x, p=period, c=ceiling: c - self._period_renewal_overlay_value(x, p),
                ))
        return specs

    # --------------------------------------------------------------------- #
    # Public optimization surface
    # --------------------------------------------------------------------- #

    def baseline(self, x=None):
        """Evaluate the horizon at a fixed overlay WITHOUT optimizing — by
        default the current / do-nothing decision vector
        (``self.input.initial_flat``), i.e. today's book re-simulated over
        ``n_periods`` at today's prices.

        Use ``baseline()["conversion"]`` to derive a realistic
        ``target_constraint`` for :meth:`optimize`: it is the lifetime,
        exposure-weighted, new-business-inclusive conversion the constraint
        actually measures, and — being the do-nothing point (``optimize``'s
        default warm start) inside ``bounds`` — an achievable equality target. A
        renewal-book mean overstates it (renewals convert higher than new
        business, and a one-period snapshot misses the horizon blend).

        Pass ``x`` to score any candidate overlay instead. Pure: no solver, no
        logging, no progress side effects. Returns ``{"conversion", "npv",
        "total_volume", "summary", "per_period_summary", "output_dataframe"}``.
        """
        if not hasattr(self, "pipeline"):
            raise RuntimeError("baseline requires an optimizer built via from_pipeline().")
        if x is None:
            x = self.input.initial_flat
        else:
            x = np.asarray(x, dtype=float)
            if x.shape != (self.input.param_total,):
                raise ValueError(
                    f"x must be a 1-D vector of length {self.input.param_total}; got shape {x.shape}."
                )
        agg = self._simulate_and_aggregate(x)
        # No solver here, so stand in a trivially-successful result for the
        # shared summary builder (target == self ⇒ constraint_gap == 0).
        result = SimpleNamespace(success=True, message="baseline (no optimization)", nit=0)
        return {
            "conversion": agg.constraint,
            "npv": agg.npv,
            "total_volume": agg.volume,
            "summary": self._build_summary(agg, agg.constraint, "baseline", result),
            "per_period_summary": self._per_period_summary(agg),
            "output_dataframe": agg.df,
        }

    def optimize(
        self,
        target_constraint,
        constraint_metric="conversion",
        constraint_kind="target",
        bounds=(0.9, 1.1),
        method="SLSQP",
        constraint_penalty_weight=100000.0,
        optimizer_options=None,
        enbp_ceiling=None,
        per_period_targets=None,
        x0=None,
        save_to_log=True,
        progress="bar",
    ):
        """Optimize the demand overlay to maximise the objective (``objective_type``,
        default horizon NPV of expected profit) subject to a lifetime constraint on
        ``constraint_metric`` relative to ``target_constraint``.

        ``constraint_metric`` selects the constrained quantity — a name from the
        metric menu (``"conversion"`` (default), ``"total_volume"``,
        ``"total_expected_profit"``, ``"average_expected_profit"``,
        ``"total_weight"``) or a callable ``agg -> float`` — and ``constraint_kind``
        its sense: ``"target"`` (``== target_constraint``, default), ``"floor"``
        (``>= target_constraint``), or ``"ceiling"`` (``<= target_constraint``).
        E.g. ``constraint_metric="total_volume", constraint_kind="floor"`` keeps
        lifetime volume at or above ``target_constraint``. The metric is evaluated
        on the same cached forward sim as the objective, so it adds no extra cost.

        ``x0`` warm-starts the solver from a flat decision vector (e.g. the
        single-period interpolated optimum, design §13.4); it defaults to the
        layer's initial multipliers. A good warm start cuts iterations sharply
        because each forward-sim evaluation is expensive.

        Additional constraints (design §11–§12):

        - ``per_period_targets`` — ``{period: (kind, value)}`` where ``kind`` is
          ``"target"`` (equality), ``"floor"`` (``>=``), or ``"ceiling"``
          (``<=``) on that period's weighted-average conversion (e.g.
          ``{0: ("floor", 0.85)}`` protects year-1 retention).
        - ``enbp_ceiling`` — a per-period upper bound on the realised renewal
          demand multiplier, i.e. renewals are not priced above the
          new-business-equivalent premium (UK GIPP; ``1.0`` ⇒ renewal price
          ≤ technical premium). Applies to renewal rows only.

        ``progress`` controls the live solve readout (each forward simulation is
        expensive, so a long run benefits from a heartbeat): ``"bar"`` (default) a
        tqdm progress bar that ticks per forward simulation with live objective and
        the constrained metric in the postfix; ``"text"`` one stats line per solver
        iteration (log-file friendly); ``False`` (or ``None``) silent. ``True`` is
        an alias for ``"bar"``.

        Mirrors :meth:`PriceOptimizerRatingFactor.optimize`: SLSQP enforces the
        constraints exactly (equality / inequality dicts); the gradient-free /
        bounded methods fold them into squared / squared-hinge penalties. The
        penalty path's ``constraint_penalty_weight * residual**2`` is
        scale-sensitive: for a large-magnitude ``constraint_metric`` (e.g.
        ``"total_volume"`` ~ 1e3) raise ``constraint_penalty_weight`` to match the
        residual magnitude, or prefer SLSQP (scale-free for the hard constraint).
        Returns ``{"optimal_multipliers", "output_dataframe", "summary", "result"}``.
        """
        if not hasattr(self, "pipeline"):
            raise RuntimeError("optimize requires an optimizer built via from_pipeline().")
        if not isinstance(target_constraint, (int, float)) or isinstance(target_constraint, bool):
            raise TypeError("target_constraint must be a numeric value.")
        if constraint_kind not in _PER_PERIOD_KINDS:
            raise ValueError(
                f"constraint_kind must be one of {list(_PER_PERIOD_KINDS)}; got {constraint_kind!r}."
            )
        constraint_fn, constraint_label = self._resolve_metric(constraint_metric, "constraint_metric")
        if method not in self._SUPPORTED_METHODS:
            raise ValueError(
                f"method must be one of {list(self._SUPPORTED_METHODS)}; got {method!r}."
            )
        if not isinstance(constraint_penalty_weight, (int, float)) or isinstance(constraint_penalty_weight, bool):
            raise TypeError("constraint_penalty_weight must be a numeric value.")
        if constraint_penalty_weight < 0:
            raise ValueError("constraint_penalty_weight must be non-negative.")
        if optimizer_options is not None and not isinstance(optimizer_options, dict):
            raise TypeError("optimizer_options must be None or a dict.")
        if enbp_ceiling is not None and (not isinstance(enbp_ceiling, (int, float)) or isinstance(enbp_ceiling, bool)):
            raise TypeError("enbp_ceiling must be None or a numeric value.")
        per_period_targets = self._validate_per_period_targets(per_period_targets)
        progress_mode = _resolve_progress(progress)

        if x0 is None:
            x0 = self.input.initial_flat
        else:
            x0 = np.asarray(x0, dtype=float)
            if x0.shape != (self.input.param_total,):
                raise ValueError(
                    f"x0 must be a 1-D vector of length {self.input.param_total}; got shape {x0.shape}."
                )

        flat_bounds = self._build_bounds(bounds)
        specs = self._build_constraint_specs(
            target_constraint, constraint_fn, constraint_kind, enbp_ceiling, per_period_targets
        )

        if method == "SLSQP":
            constraints = [{"type": kind, "fun": fun} for kind, fun in specs]
            fn = self._objective_fn
        else:
            constraints = ()
            weight = float(constraint_penalty_weight)

            def fn(x, specs=specs, w=weight):
                penalty = 0.0
                for kind, residual in specs:
                    r = residual(x)
                    # eq -> r**2; ineq (want r >= 0) -> squared-hinge on violation.
                    penalty += r * r if kind == "eq" else min(r, 0.0) ** 2
                p = getattr(self, "_progress", None)
                if p is not None:
                    p.set_penalty(w * penalty)  # the term added to the minimised objective
                return self._objective_fn(x) + w * penalty

        reporter = None
        callback = None
        if progress_mode is not None:
            reporter = _ProgressReporter(
                progress_mode,
                target=target_constraint,
                raw_objective=self._raw_objective,
                weight_col=self.parameters.sample_weight_col,
                out_col=self._layer_meta["output_column"],
                constraint_name=("conv" if constraint_metric == "conversion" else constraint_label),
                constraint_fn=constraint_fn,
            )
            # `_simulate_cached` pulses this on each real forward sim (cache miss).
            self._progress = reporter
            callback = reporter.iteration

        try:
            result = minimize(
                fn,
                x0=x0,
                method=method,
                bounds=flat_bounds,
                constraints=constraints,
                options=optimizer_options or {},
                callback=callback,
            )
        finally:
            # Always tear down the bar and clear the hook, even if minimize raised
            # or was interrupted — never leak a stale bar or a dangling _progress.
            if reporter is not None:
                reporter.close()
            self._progress = None

        if not result.success:
            warnings.warn(f"{method} did not converge: {result.message}")

        x_opt = result.x
        optimal_multipliers = self._unflatten(x_opt)
        agg = self._simulate_and_aggregate(x_opt)
        output_df = agg.df
        summary = self._build_summary(
            agg, target_constraint, method, result,
            constraint_fn=constraint_fn, constraint_label=constraint_label, constraint_kind=constraint_kind,
        )

        if save_to_log:
            self.output.optimal_array = x_opt
            self.output.optimal_multipliers = optimal_multipliers
            self.output.output_optimized = output_df
            self.output.per_period_summary = self._per_period_summary(agg)
            self.output.optimization_scoring_history = pd.concat([
                self.output.optimization_scoring_history,
                pd.DataFrame({
                    "target_constraint": [target_constraint],
                    "constraint": [agg.constraint],
                    "objective": [self._raw_objective(agg)],
                    "price_multiplier": [summary["price_multiplier"].iloc[0]],
                }),
            ])

        return {
            "optimal_multipliers": optimal_multipliers,
            "output_dataframe": output_df,
            "summary": summary,
            "result": result,
        }

    def _build_summary(self, agg, target_constraint, method, result,
                       constraint_fn=None, constraint_label="conversion", constraint_kind="target"):
        """One-row lifetime summary mirroring the single-period optimizer's
        shape, extended with the horizon NPV and realised overlay multiplier.

        ``constraint_fn`` / ``constraint_label`` / ``constraint_kind`` describe the
        constrained quantity; defaults reproduce the legacy conversion-equality
        summary (``constraint_gap`` measured against ``agg.constraint``).
        ``constraint_gap = realized_metric - target_constraint`` regardless of
        kind — for a ``floor`` a non-negative gap means satisfied, for a
        ``ceiling`` a non-positive gap means satisfied. ``constraint`` (conversion)
        and ``total_volume`` stay reported KPIs whichever metric is constrained.
        """
        df = agg.df
        w = df[self.parameters.sample_weight_col].to_numpy(dtype=float)
        out_col = self._layer_meta["output_column"]
        price_multiplier = float(np.average(df[out_col], weights=w)) if w.sum() > 0 else float("nan")
        realized = constraint_fn(agg) if constraint_fn is not None else agg.constraint

        data = {
            "target_constraint": [target_constraint],
            "constraint": [agg.constraint],
            "objective": [self._raw_objective(agg)],
            "price_multiplier": [price_multiplier],
            "npv": [agg.npv],
            "total_volume": [agg.volume],
            "objective_type": [self.parameters.objective_label],
            "method": [method],
            "constraint_metric": [constraint_label],
            "constraint_kind": [constraint_kind],
            "constraint_gap": [realized - target_constraint],
            "result_success": [bool(result.success)],
            "result_message": [str(result.message)],
            "result_nit": [int(getattr(result, "nit", -1))],
        }
        return pd.DataFrame(data)

    # `_per_period_summary` now lives on the shared `_MultiPeriodHorizonEngine`
    # (both optimizers inherit it); the call in `optimize` resolves via inheritance.

    def calculate_efficient_frontier(
        self,
        target_constraint_range,
        constraint_metric="conversion",
        constraint_kind="target",
        bounds=(0.9, 1.1),
        method="SLSQP",
        constraint_penalty_weight=100000.0,
        optimizer_options=None,
        enbp_ceiling=None,
        per_period_targets=None,
        progress="bar",
    ):
        """Optimize across a range of lifetime constraint targets, returning one
        summary row per achievable target (the multi-period efficient frontier).
        ``constraint_metric`` / ``constraint_kind`` (defaulting to the conversion
        equality) select what each value in ``target_constraint_range`` targets,
        and are held fixed across the sweep, as are any ``enbp_ceiling`` /
        ``per_period_targets``.

        ``progress`` mirrors :meth:`optimize` but tracks the *sweep*: ``"bar"``
        (default) one outer tqdm bar over the targets; ``"text"`` one stats line
        per target; ``False`` silent. The inner per-target solves are always run
        silently so bars do not stack."""
        if not isinstance(target_constraint_range, (list, tuple, np.ndarray)):
            raise TypeError("target_constraint_range must be a list, tuple, or NumPy array.")
        if len(target_constraint_range) == 0:
            raise ValueError("target_constraint_range cannot be empty.")
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in target_constraint_range):
            raise TypeError("All values in target_constraint_range must be numeric.")
        progress_mode = _resolve_progress(progress)

        iterator = target_constraint_range
        if progress_mode == "bar":
            iterator = tqdm(target_constraint_range, desc="Efficient frontier", unit="target", leave=True)

        summary_rows = []
        for target in iterator:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = self.optimize(
                        target_constraint=target,
                        constraint_metric=constraint_metric,
                        constraint_kind=constraint_kind,
                        bounds=bounds,
                        method=method,
                        constraint_penalty_weight=constraint_penalty_weight,
                        optimizer_options=optimizer_options,
                        enbp_ceiling=enbp_ceiling,
                        per_period_targets=per_period_targets,
                        save_to_log=False,
                        progress=False,  # the outer sweep owns the readout
                    )
                if result["summary"]["result_success"].iloc[0]:
                    summary_rows.append(result["summary"])
                    if progress_mode == "text":
                        s = result["summary"].iloc[0]
                        print(
                            f"target {target:.4f}  npv={s['npv']:.6g}  "
                            f"conv={s['constraint']:.4f}  pmult={s['price_multiplier']:.4f}"
                        )
                else:
                    print(f"Warning: {method} did not converge at target {target:.4f}.")
            except Exception as exc:
                print(f"Warning: optimization failed at target {target:.4f}: {exc}")

        if not summary_rows:
            raise RuntimeError("No valid results were produced. All targets may have failed.")

        self.output.efficient_frontier = pd.concat(summary_rows, ignore_index=True)
        return self.output.efficient_frontier

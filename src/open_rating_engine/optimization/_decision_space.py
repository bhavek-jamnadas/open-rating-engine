"""Shared decision-variable machinery for rating-factor price optimizers.

``_RatingFactorDecisionSpace`` owns everything about the *decision space* — the
per-category multipliers an optimizer tunes — independent of how the objective is
evaluated. The single-period :class:`PriceOptimizerRatingFactor` evaluates its
objective by interpolating a precomputed ``(N, K)`` grid; the multi-period
optimizer evaluates it by forward-simulating a horizon. Both flatten/unflatten
the same decision vector, parse the same kind of multiplicative
``RatingTablesLayer``, and deploy the optimum via the same
``to_rating_tables`` / ``to_layer``, so that logic lives here once.

Contract for subclasses:

- populate ``self.input`` (a ``SimpleNamespace``) with at least the keys returned
  by :meth:`_compute_decision_space` (``factor_names``, ``category_orders``,
  ``row_category_indices``, ``default_contrib``, ``param_offsets``,
  ``param_total``, ``initial_flat``, ``initial_multipliers``);
- populate ``self.output`` (a ``SimpleNamespace``); ``optimize`` sets
  ``self.output.optimal_multipliers``;
- ``from_pipeline`` builders set ``self._layer_schema`` / ``self._layer_meta``
  (both produced by :meth:`_parse_rating_layer`) before
  :meth:`to_rating_tables` / :meth:`to_layer` are used.
"""
import ast

import numpy as np
import pandas as pd


_RESERVED_SUMMARY_COLS = ("target_constraint", "constraint", "objective", "price_multiplier")

# Separator used to fold a multi-column rating-table key into a single hashable
# label (the optimizer keys decision variables by 1-D category labels). The ASCII
# unit separator does not appear in categorical rating-factor values.
_KEY_SEP = "\x1f"


def _encode_keys(frame, key_cols):
    """Fold one or more key columns of ``frame`` into a 1-D array of labels.

    A single key column is returned verbatim (preserving dtype, so single-column
    tables behave exactly as before). Multiple key columns are joined with
    ``_KEY_SEP`` after stringifying, producing one composite label per row.
    """
    if len(key_cols) == 1:
        return frame[key_cols[0]].to_numpy()
    return frame[key_cols].astype(str).agg(_KEY_SEP.join, axis=1).to_numpy()


def _formula_product_terms(formula):
    """Return the set of table names multiplied in ``formula`` if it is a pure
    product of ``<table>['value']`` terms, else ``None``.

    The optimizer combines rating tables multiplicatively, so auto-derivation
    from a layer is only valid when the formula is omitted (implicit product) or
    is exactly the product of its tables' ``value`` columns, e.g.
    ``a['value'] * b['value']``. Any other operator/structure returns ``None``.
    This validates the product *structure* only; the caller checks the referenced
    names against the layer's actual rating tables.
    """
    try:
        tree = ast.parse(formula, mode="eval").body
    except SyntaxError:
        return None

    names = set()

    def _check(node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return _check(node.left) and _check(node.right)
        # Leaf must be exactly <name>['value'].
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "value"
        ):
            names.add(node.value.id)
            return True
        return False

    return names if _check(tree) else None


class _RatingFactorDecisionSpace:
    """Decision-variable handling shared by the single- and multi-period
    rating-factor optimizers: flatten/unflatten, effective-multiplier expansion,
    bounds building, ``RatingTablesLayer`` parsing, and deployment.

    Not instantiated directly; subclasses build ``self.input`` / ``self.output``
    and add an objective evaluator.
    """

    @staticmethod
    def _coerce_1d(arr, name):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a 1-D array.")
        return arr

    @staticmethod
    def _coerce_2d(arr, name):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 2-D array.")
        return arr

    def _compute_decision_space(self, rating_factors, initial_multipliers, default_multipliers, n_rows):
        """Validate the decision-space inputs and flatten them into the per-factor
        bookkeeping the optimizer works with.

        Pure with respect to ``self`` (it runs before ``self.input`` exists): it
        reads only its arguments and returns a dict of the eight decision-space
        attributes the subclass merges into ``self.input``. ``n_rows`` is the
        number of data rows each factor's ``rating_factors`` vector must match.
        """
        if not isinstance(initial_multipliers, dict) or not initial_multipliers:
            raise TypeError("initial_multipliers must be a non-empty dict[str, dict[str, float]].")
        if not isinstance(rating_factors, dict) or not rating_factors:
            raise TypeError("rating_factors must be a non-empty dict[str, np.ndarray].")
        if set(rating_factors) != set(initial_multipliers):
            raise ValueError(
                f"rating_factors keys {sorted(rating_factors)} must match "
                f"initial_multipliers keys {sorted(initial_multipliers)}."
            )

        factor_names = tuple(initial_multipliers.keys())
        collisions = [f for f in factor_names if f in _RESERVED_SUMMARY_COLS]
        if collisions:
            raise ValueError(
                f"Factor name(s) {collisions} collide with reserved summary columns {_RESERVED_SUMMARY_COLS}."
            )

        if default_multipliers is not None:
            if not isinstance(default_multipliers, dict):
                raise TypeError("default_multipliers must be a dict[str, float] or None.")
            unknown = set(default_multipliers) - set(factor_names)
            if unknown:
                raise ValueError(
                    f"default_multipliers keys {sorted(unknown)} are not rating factors {list(factor_names)}."
                )
            for k, v in default_multipliers.items():
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    raise TypeError(f"default_multipliers['{k}'] must be numeric.")
        # Per-factor fixed contribution for rows whose category is not a decision
        # variable. Factors absent here keep the strict behaviour (raise on miss).
        default_contrib = {f: float(default_multipliers[f]) for f in (default_multipliers or {})}

        category_orders = {}
        row_category_indices = {}
        param_offsets = {}
        initial_flat_parts = []
        offset = 0
        for factor in factor_names:
            cat_dict = initial_multipliers[factor]
            if not isinstance(cat_dict, dict) or not cat_dict:
                raise TypeError(f"initial_multipliers['{factor}'] must be a non-empty dict.")
            cats = tuple(cat_dict.keys())
            category_orders[factor] = cats

            row_vec = np.asarray(rating_factors[factor])
            if row_vec.ndim != 1 or row_vec.shape[0] != n_rows:
                raise ValueError(
                    f"rating_factors['{factor}'] must be a 1-D array of length {n_rows}; got shape {row_vec.shape}."
                )

            mapping = pd.Series(np.arange(len(cats)), index=cats)
            idx = mapping.reindex(row_vec).to_numpy()  # float; NaN where unmatched
            unmatched = np.isnan(idx)
            if unmatched.any():
                missing = sorted(set(pd.unique(row_vec[unmatched]).tolist()), key=str)
                if factor not in default_contrib:
                    raise KeyError(
                        f"rating_factors['{factor}'] contains categories not in "
                        f"initial_multipliers['{factor}']: {missing}. Pass "
                        f"default_multipliers['{factor}'] to map unmatched rows to a fixed default."
                    )
                idx = np.where(unmatched, -1.0, idx)
            row_category_indices[factor] = idx.astype(np.int64)

            param_offsets[factor] = offset
            offset += len(cats)
            initial_flat_parts.append(np.asarray(list(cat_dict.values()), dtype=float))

        return {
            "factor_names": factor_names,
            "category_orders": category_orders,
            "row_category_indices": row_category_indices,
            "default_contrib": default_contrib,
            "param_offsets": param_offsets,
            "param_total": offset,
            "initial_flat": np.concatenate(initial_flat_parts),
            "initial_multipliers": initial_multipliers,
        }

    def _unflatten(self, x):
        out = {}
        for factor in self.input.factor_names:
            offset = self.input.param_offsets[factor]
            cats = self.input.category_orders[factor]
            out[factor] = {cat: float(x[offset + i]) for i, cat in enumerate(cats)}
        return out

    def _flatten(self, multipliers):
        """Inverse of :meth:`_unflatten`: pack a ``{factor: {category: value}}``
        mapping back into the flat decision vector, ordered by ``param_offsets`` /
        ``category_orders``. Raises a clear error naming any missing factor or
        category (e.g. when stress-testing a hand-specified overlay)."""
        x = np.empty(self.input.param_total, dtype=float)
        for factor in self.input.factor_names:
            try:
                cats_values = multipliers[factor]
            except (KeyError, TypeError):
                raise KeyError(f"multipliers is missing factor '{factor}'.") from None
            offset = self.input.param_offsets[factor]
            for i, cat in enumerate(self.input.category_orders[factor]):
                try:
                    x[offset + i] = float(cats_values[cat])
                except (KeyError, TypeError):
                    raise KeyError(
                        f"multipliers['{factor}'] is missing category '{cat}'."
                    ) from None
        return x

    def _factor_contribution(self, factor, slice_, idx=None):
        """Per-row multiplier from one factor: the chosen variable, or the fixed
        default for rows whose category is not a decision variable (idx == -1).

        ``idx`` (the row→category index map) defaults to the factor's fixed
        single-period map; the multi-period optimizer passes a per-period map
        recomputed from the aged book.
        """
        if idx is None:
            idx = self.input.row_category_indices[factor]
        if factor in self.input.default_contrib:
            safe = np.where(idx >= 0, idx, 0)
            return np.where(idx >= 0, slice_[safe], self.input.default_contrib[factor])
        return slice_[idx]

    def _effective_multiplier(self, x, row_category_indices=None):
        """Product over factors of each row's chosen multiplier.

        ``row_category_indices`` defaults to the fixed single-period map; the
        multi-period optimizer passes the aged map for the current period. The
        number of rows is taken from that map (not from any objective array), so
        the base stays independent of how the objective is evaluated.
        """
        if row_category_indices is None:
            row_category_indices = self.input.row_category_indices
        factor_names = self.input.factor_names
        n_rows = len(row_category_indices[factor_names[0]])
        eff = np.ones(n_rows)
        for factor in factor_names:
            offset = self.input.param_offsets[factor]
            n_cats = len(self.input.category_orders[factor])
            slice_ = x[offset:offset + n_cats]
            eff *= self._factor_contribution(factor, slice_, row_category_indices[factor])
        return eff

    def _build_bounds(self, bounds):
        if isinstance(bounds, tuple) and len(bounds) == 2:
            lo, hi = bounds
            if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float))):
                raise TypeError("bounds tuple must be (numeric, numeric).")
            return [(float(lo), float(hi))] * self.input.param_total
        if isinstance(bounds, dict):
            unexpected = set(bounds) - set(self.input.factor_names)
            if unexpected:
                raise ValueError(
                    f"bounds keys {sorted(unexpected)} are not rating factors {list(self.input.factor_names)}."
                )
            result = []
            for factor in self.input.factor_names:
                b = bounds.get(factor, (0.8, 1.2))
                if not (isinstance(b, tuple) and len(b) == 2):
                    raise TypeError(f"bounds['{factor}'] must be a (lo, hi) tuple.")
                result.extend([b] * len(self.input.category_orders[factor]))
            return result
        raise TypeError("bounds must be a (lo, hi) tuple or a dict[factor, (lo, hi)].")

    @staticmethod
    def _parse_rating_layer(pipeline, layer_name, scored_data):
        """Derive decision-space inputs plus deploy metadata from a multiplicative
        ``RatingTablesLayer`` registered on ``pipeline``.

        Each rating table becomes one rating factor; its key column(s) define the
        categories (folded into a single label for multi-column joins). Decision
        variables are the table's rows, seeded from their current ``value``; data
        rows whose key combination is absent from a table take that table's
        ``default_values`` entry (fixed, not optimized).

        Returns ``{rating_factors, initial_multipliers, default_multipliers,
        layer_schema, layer_meta}``: the first three feed the constructor, the
        last two are assigned to ``self._layer_schema`` / ``self._layer_meta`` for
        :meth:`to_rating_tables` / :meth:`to_layer`.
        """
        if not hasattr(pipeline, "layers") or layer_name not in pipeline.layers:
            raise KeyError(f"layer '{layer_name}' not found in pipeline")
        layer = pipeline.layers[layer_name]
        for attr in ("rating_tables", "default_values", "target_table"):
            if not hasattr(layer, attr):
                raise TypeError(
                    f"layer '{layer_name}' is not a RatingTablesLayer (missing '{attr}')."
                )

        formula = getattr(layer, "formula", None)
        if formula is not None:
            terms = _formula_product_terms(formula)
            tables = set(layer.rating_tables)
            if terms is None:
                raise ValueError(
                    f"layer '{layer_name}' formula {formula!r} is not a pure product of "
                    f"\"<table>['value']\" terms. from_pipeline only supports multiplicative "
                    f"rating-table layers; build rating_factors/initial_multipliers manually instead."
                )
            if terms != tables:
                raise ValueError(
                    f"layer '{layer_name}' formula multiplies {sorted(terms)}, but its "
                    f"rating_tables keys are {sorted(tables)}. Every rating table must appear "
                    f"exactly once as a \"<table>['value']\" term, and the names used in the "
                    f"formula must match the rating_tables keys. "
                    f"In rating_tables but not the formula: {sorted(tables - terms)}; "
                    f"in the formula but not rating_tables: {sorted(terms - tables)}."
                )

        target = layer.target_table
        if not isinstance(scored_data, dict) or target not in scored_data:
            raise KeyError(
                f"scored_data must be a dict containing the layer target table '{target}'; "
                f"pass the output of pipeline.predict(...)."
            )
        df = scored_data[target]

        rating_factors = {}
        initial_multipliers = {}
        default_multipliers = {}
        schema = {}
        for tname, tdf in layer.rating_tables.items():
            key_cols = [c for c in tdf.columns if c != "value"]
            missing_cols = [c for c in key_cols if c not in df.columns]
            if missing_cols:
                raise KeyError(
                    f"rating table '{tname}' key columns {missing_cols} are missing from "
                    f"scored_data['{target}']; pass the scored pipeline output."
                )
            table_labels = _encode_keys(tdf, key_cols)
            rating_factors[tname] = _encode_keys(df, key_cols)
            initial_multipliers[tname] = {
                lbl: float(val) for lbl, val in zip(table_labels, tdf["value"].to_numpy())
            }
            default_multipliers[tname] = float(layer.default_values[tname])
            schema[tname] = {
                "key_cols": key_cols,
                "decode": {
                    lbl: tuple(row)
                    for lbl, row in zip(table_labels, tdf[key_cols].to_numpy())
                },
            }

        layer_meta = {
            "name": layer.name,
            "target_table": target,
            "output_column": layer.output_column,
            "formula": formula,
            "default_values": dict(layer.default_values),
            "dependencies": list(layer.dependencies),
        }
        return {
            "rating_factors": rating_factors,
            "initial_multipliers": initial_multipliers,
            "default_multipliers": default_multipliers,
            "layer_schema": schema,
            "layer_meta": layer_meta,
        }

    def to_rating_tables(self, optimal_multipliers=None):
        """Decode optimized multipliers into ``{table: DataFrame}`` with the
        original key column(s) plus a ``value`` column.

        Requires this optimizer to have been built via ``from_pipeline``.
        Defaults to the last ``optimize`` result.
        """
        if not hasattr(self, "_layer_schema"):
            raise RuntimeError("to_rating_tables requires an optimizer built via from_pipeline().")
        if optimal_multipliers is None:
            optimal_multipliers = getattr(self.output, "optimal_multipliers", None)
        if optimal_multipliers is None:
            raise RuntimeError("No optimal_multipliers available; run optimize() first.")

        tables = {}
        for tname, mults in optimal_multipliers.items():
            key_cols = self._layer_schema[tname]["key_cols"]
            decode = self._layer_schema[tname]["decode"]
            rows = [
                {**dict(zip(key_cols, decode[label])), "value": float(mult)}
                for label, mult in mults.items()
            ]
            tables[tname] = pd.DataFrame(rows, columns=key_cols + ["value"])
        return tables

    def to_layer(self, name=None):
        """Rebuild a ``RatingTablesLayer`` from the optimized multipliers, reusing
        the source layer's formula, defaults, dependencies, target table and
        output column. Requires construction via ``from_pipeline``.
        """
        if not hasattr(self, "_layer_meta"):
            raise RuntimeError("to_layer requires an optimizer built via from_pipeline().")
        from open_rating_engine import RatingTablesLayer

        meta = self._layer_meta
        return RatingTablesLayer(
            name=name or meta["name"],
            target_table=meta["target_table"],
            output_column=meta["output_column"],
            rating_tables=self.to_rating_tables(),
            formula=meta["formula"],
            default_values=meta["default_values"],
            dependencies=meta["dependencies"],
        )

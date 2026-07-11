# Code Quality Report & Roadmap — "Operation Increase Code Quality"

Goal: enforce **black**, **flake8**, and **mypy** in CI, matching the style guide
already in use in the `optimizers` repo. This is delivered as a **sequence of small,
independently-reviewable PRs** rather than one large drop.

## Style guide (copied from `optimizers`)

| Tool   | Setting in `optimizers`                                              | Applied here |
|--------|---------------------------------------------------------------------|--------------|
| black  | Defaults (line length 88), no `[tool.black]`; CI runs `black --check .` | Same. We additionally pin `target-version = ["py311"]` to silence a spurious AST-safety warning (see below). |
| flake8 | `max-line-length = 120`, `extend-ignore = ["E203"]`, `exclude` build dirs, via `Flake8-pyproject` | Copied verbatim. |
| mypy   | *Not actually configured* in `optimizers` (only a stale `.mypy_cache`). | Introduced fresh here as a lenient baseline, then ratcheted. |

Note the intentional black/flake8 split: black formats to 88 columns, flake8's
`E501` only trips at 120, so the two never fight. `E203` is ignored because black
and pycodestyle disagree on whitespace around slice colons.

## Baseline gap (measured against `main`)

| Check  | Result on `main` before this work |
|--------|-----------------------------------|
| black  | **6 of 19 files** would be reformatted — CI's existing `black --check .` step is currently **red**. |
| flake8 (120 / E203) | **34 violations**: unused imports (F401, mostly the re-export `__init__.py`), 6× line-too-long (E501), 5× f-string-without-placeholder (F541), 1× unused local (F841), `== True/False` comparisons (E712), stray operator spacing (E221/E221). |
| mypy (lenient, `--ignore-missing-imports`) | **6 errors in 2 files** (`pvat.py`, `ivatmeans.py`): a redefinition, a `None`-typed variable reassigned to a real type, and a `Union` that is indexed/iterated without narrowing. Several look like latent bugs, not just annotation noise. |

Good news: the gap is small. `src/` is only ~1.1k LOC and mypy in non-strict mode
already nearly passes.

## The PR sequence

Each PR is green-on-merge and does one thing.

1. **`black` — make the tree clean + enforce (THIS PR).**
   Reformat the 6 offending files (pure formatting, no logic change), pin
   `[tool.black] target-version = ["py311"]`, and land this report. CI's existing
   `black --check .` step goes green. Nothing else changes.

2. **`flake8` — introduce config + fix the 34 violations + gate in CI.**
   Copy `optimizers`' `[tool.flake8]` block, add `Flake8-pyproject` to the `dev`
   extra, add a `flake8` CI step. Fixes are mechanical: `__all__` (or `# noqa: F401`)
   on the re-export `__init__.py`, wrap/soften long lines, drop dead f-strings and
   the unused local, use `is`/truthiness instead of `== True/False`.

3. **`mypy` — introduce a lenient baseline + fix the 6 errors + gate in CI.**
   Add a gentle `[tool.mypy]` (`ignore_missing_imports = true`, non-strict), add
   `mypy` to the `dev` extra and a CI step. Fix the 6 real errors. Since these touch
   control flow (the `Union[IvatMeansResult, list[...]]` returns), each gets a quick
   behavioral check.

4. **(Optional follow-ups) ratchet mypy strictness.**
   Once green, tighten incrementally — `disallow_untyped_defs` per-module, then
   `strict` — one module per PR so review stays small. This is where the real
   type-safety value is; steps 1–3 just establish the enforcement scaffold.

## Why this order

black first because it is the loudest (CI is already red on it) and the safest
(zero semantic change) — it unblocks everything. flake8 second because its fixes
are still mechanical. mypy last because its fixes are the only ones that touch
runtime behavior, so it deserves the most careful review, and its strictness is a
dial we can keep turning after the scaffold exists.

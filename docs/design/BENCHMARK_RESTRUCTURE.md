# Benchmark restructure: datasets × protocols × status

Status: design proposal, 2026-07. Companion to
`ARCHITECTURE_RECOMMENDATIONS.md` (R2/R6/R7); this document is the concrete
implementation target for the benchmark consolidation. Written to be
executed as a sequence of small PRs by any contributor.

## Why the current system is hard to understand

Verified confusions, with evidence:

1. **One dataset, three opaque keys.** `plasma.constellaration`,
   `plasma.constellaration_paper`, `plasma.constellaration_multioutput` are
   the SAME dataset (ConStellaration, Cadena et al. 2025; 26,897 samples,
   canonical split NPZ) under three *protocols* (10k-subsample 90→1 quick
   check; the paper's §A.4 twelve-independent-models protocol; joint
   90→12). Nothing in the key or `--list` output says so; the paper-protocol
   runner is a private function inside `leaderboard.py`
   (`_run_constellaration_paper_benchmark`, line ~1531) — a dataset-specific
   protocol implemented inside a display/aggregation module.
2. **Near-duplicate names for different things.** `pde.burgers_1d`
   (64-point toy, inline FD solver, hermetic) vs `pdebench.burgers_1d`
   (1,024-point PDEBench HDF5, multi-GB download). Users cannot tell which
   one a result refers to without reading loaders.
3. **Invisible status.** PDEBench (3 keys) and TheWell (3 keys) have real
   loaders but require multi-GB downloads / the optional `the-well`
   package; on a fresh machine they have zero results and no cache, yet
   `--list` shows them exactly like verified benchmarks. Six other keys
   (ctr23.*, plasma.cmod_density_limit) are `None`-runner placeholders.
   Three different "not really available yet" situations, all rendered
   identically.
4. **Definition scattered across four places.** For one benchmark: loader +
   model list (`leaderboard.py`), runner (`tasks.py` or `leaderboard.py`),
   threshold (`_THRESHOLDS` dict in `leaderboard.py`), descriptive metadata
   (`metadata.yaml`), HPO space (`hpo.py`). Understanding one benchmark
   means reading ~4 files; `leaderboard.py` is 1,933 lines because it
   absorbed everything.
5. **Category drift.** `sequence.lorenz63` is categorized "tabular" in the
   legacy tier mapping, "Time Series / Forecasting" in metadata capability,
   and `sequence.` by prefix.

## Target model: `suite / dataset / protocol` + explicit status

**A benchmark = dataset spec × protocol spec.** One YAML file per dataset
under `surge/benchmarks/specs/`, containing its protocols:

```yaml
# surge/benchmarks/specs/fusion/constellaration.yaml
dataset:
  id: fusion.constellaration
  title: ConStellaration (QI stellarator boundaries)
  description: >-
    Map 90 VMEC boundary Fourier coefficients to ideal-MHD equilibrium
    metrics for quasi-isodynamic stellarator design.
  citation: {text: "Cadena et al. (2025) arXiv:2506.19583",
             doi: 10.48550/arXiv.2506.19583}
  source: {loader: huggingface, name: proxima-fusion/constellaration,
           sha256: <content digest>, size: 210MB, license: CC-BY-4.0}
  schema: {inputs: 90, outputs: 12, n: 26897,
           features_doc: true}          # feature docs live in the spec
  split: {strategy: canonical, file: split_n26897_seed42_test0.2.npz,
          sha256: ...}
protocols:
  quick:        {task: regression, outputs: [minor_radius], subsample: 10000,
                 metrics: [r2, rmse], gate: {metric: r2, ">=": 0.90,
                 basis: "SURGE baseline"}}
  paper:        {task: regression, style: one-model-per-output,
                 metrics: [r2_mean, r2_min], gate: {metric: r2_mean,
                 ">=": 0.97, basis: "Cadena et al. §A.4"}}
  multioutput:  {task: regression, style: joint,
                 metrics: [r2_mean, nrmse_mean], gate: {metric: r2_mean,
                 ">=": 0.85, basis: "SURGE baseline"}}
status: verified          # verified | implemented | needs-data | proposed
requires: []              # e.g. [the-well], [h5py]
```

Benchmark ids become `fusion.constellaration:paper` (dataset id +
protocol). Single-protocol datasets omit the suffix.

**Suites** (top-level folders and the only categories anywhere):
`smoke/` (hermetic, CI), `tabular/`, `vision/`, `field/` (operator
learning: toy PDEs, PDEBench, TheWell), `sequence/`, `fusion/` (M3D-C1,
QLKNN, ConStellaration, C-Mod). The toy/real Burgers confusion is resolved
by naming: `field.burgers_toy` vs `field.pdebench_burgers1d`.

**Status is first-class and honest.**

| status | meaning | `--list` rendering | leaderboard |
|---|---|---|---|
| `verified` | ran end-to-end in CI or has committed reference results | plain | shown |
| `implemented` | code complete, not yet exercised on reference hardware | `[unverified]` | shown w/ badge |
| `needs-data` | code complete; dataset requires download/optional pkg not present | `[needs data: 6.5 GB / pip install the-well]` | hidden until data exists |
| `proposed` | spec only, no runner | `[proposed]` | never shown |

`surge bench fetch <id>` downloads + sha-verifies the dataset (prints size
first); `surge bench status` shows the local cache/result state per
benchmark. Runners never download implicitly — today's silent multi-GB
first-run download becomes an explicit, resumable step.

**Execution is generic.** One runner: spec → `DataBundle` (loader registry)
→ canonical/declared split → `SurrogateEngine` → metrics by name → run
manifest in `benchmark_reports/`. Protocol styles (`one-model-per-output`,
`joint`, rollout for sequences) are small strategy classes in
`surge/benchmarks/protocols.py` — the ConStellaration paper protocol moves
out of `leaderboard.py` into a ~60-line reusable `OneModelPerOutput`
protocol that PDEBench/TheWell multi-channel tasks can also use.
`leaderboard.py` shrinks to aggregation + rendering only.

**Docs and CLI generated from specs.** The `--list` tree, the leaderboard
cards (descriptions, citations, gates, feature docs), and a
`docs/benchmarks.md` reference page are all rendered from
`specs/**/*.yaml`. `metadata.yaml` (2026-07 extraction) is the seed content
and is absorbed into the specs, then deleted.

## Migration table (all 37 current keys)

| current key | new id | status |
|---|---|---|
| synthetic.regression_1d / multioutput_2d / classification_binary, classification.flow_regime | smoke.* (unchanged names) | verified |
| tabular.{diabetes, california_housing, concrete_strength, energy_efficiency, airfoil_noise, yacht_dynamics, superconductor, iris, wine, breast_cancer, digits} | tabular.* | verified |
| classification.covertype, classification.plasma_stability | tabular.covertype, fusion.plasma_stability | verified |
| multioutput.scm20d | tabular.scm20d:joint | verified |
| sequence.lorenz63 | sequence.lorenz63:rollout | verified |
| pde.burgers_1d | field.burgers_toy | verified |
| pdebench.{burgers_1d, darcy_2d, shallow_water_2d} | field.pdebench_{burgers1d, darcy2d, shallowwater2d} | **implemented → needs-data** (6.5 GB total; no local cache/results anywhere yet) |
| thewell.{gray_scott, turbulence_2d, mhd} | field.thewell_{grayscott, turbulence2d, mhd} | **needs-data** (`pip install the-well` + download; never run) |
| vision.{mnist, cifar10} | vision.* | verified (mnist), implemented (cifar10 — only aborted local runs) |
| fusion.m3dc1_sample | fusion.m3dc1_sample | needs-data (HDF5 not distributed) |
| plasma.qlknn_transport | fusion.qlknn_transport | verified |
| plasma.constellaration{,_paper,_multioutput} | fusion.constellaration:{quick,paper,multioutput} | verified |
| plasma.cmod_density_limit | fusion.cmod_density_limit | proposed (data cached, runner None) |
| ctr23.* (5 keys) | tabular.ctr23_* | proposed |

Old keys remain as aliases for one release (results files carry both ids in
the manifest) so existing `benchmark_reports/` trees keep aggregating.

## Science-benchmark specifics

- **ConStellaration**: keep the canonical split NPZ as the split contract
  (sha in spec); the `paper` gate cites §A.4 (0.97) and the current best
  (mean R² ≈ 0.93 residual MLP) stays visible as a not-yet-passing entry —
  that gap is scientific signal, not failure noise.
- **PDEBench**: spec pins exact file (e.g. `1D_Burgers_Sols_Nu0.01.hdf5`,
  darus.uni-stuttgart.de file id), sha, size, and the operator-model
  compatibility list; `fetch` supports partial suites. First verified run
  should be recorded in CI-adjacent "reference results" so status can
  flip to `verified`.
- **TheWell**: `requires: [the-well]`; spec records well dataset name and
  the field-channel subset used; same fetch/status flow. Until someone
  runs them on real hardware they are `needs-data`, and the leaderboard
  stops implying otherwise.

## PR sequence for implementation (each independently shippable)

1. **Spec schema + loader**: `specs/` YAML schema (jsonschema-validated),
   parser, and absorption of `metadata.yaml`; `--list` gains status badges.
   No behavior change.
2. **Status + fetch**: `surge bench status|fetch` with sha-verified
   downloads; runners refuse to download implicitly.
3. **Protocol classes**: extract `OneModelPerOutput`, `JointMultiOutput`,
   `SequenceRollout` from `leaderboard.py`/`tasks.py`; ConStellaration's
   three keys become one spec + three protocols (aliases kept).
4. **Engine-based runner** for `smoke` + `tabular` suites (R2), manifest
   output (R7); delete per-benchmark bespoke runners as parity is proven.
5. **field/ suite migration** (toy Burgers rename, PDEBench, TheWell) on
   the generic runner; record first reference results.
6. **Docs generation** (`docs/benchmarks.md` from specs) + delete
   `_THRESHOLDS`, `metadata.yaml`, and the legacy tier mapping.

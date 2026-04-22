---
name: Bug report
about: Report a defect in SURGE
title: "[bug] "
labels: ["bug"]
assignees: []
---

<!--
If this is a security issue please DO NOT file it here. Use the private
channel in SECURITY.md instead.
-->

## What happened

<!-- A clear, one-paragraph description of the bug. -->

## Minimal reproducer

```python
# Smallest standalone snippet that triggers the bug.
```

If the bug needs a dataset, either attach a synthetic CSV (under ~1 MB)
or explain how to generate one.

## Expected behaviour

<!-- What you thought should happen. -->

## Actual behaviour

<!-- Copy/paste the full traceback or the [surge.fit] ... banner plus any
log lines that precede the failure. Include `logging.basicConfig(level='INFO')`
output when relevant. -->

## Environment

- SURGE version: `python -c "import surge; print(surge.__version__)"`
- Python version: `python --version`
- OS: (e.g. `Linux 6.1`, `macOS 14`, `Cray shasta_c`)
- Install method: `pip install -e .`, `pip install surge-ml`, wheel, ...
- Relevant optional deps: `torch`, `gpflow`, `optuna`, `onnxruntime` (and
  their versions)

## Anything else

<!-- Related issues, recent PRs, first version where the bug appeared, ... -->

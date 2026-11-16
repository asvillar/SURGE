# Security Policy

## Reporting a Vulnerability

If you believe you have found a security vulnerability in SURGE, please
**do not open a public GitHub issue**. Instead, report it privately to the
maintainer so we can evaluate and fix the issue before public disclosure.

**Contact:** <asvillar@princeton.edu>

Please include, to the extent possible:

- A description of the vulnerability and its potential impact.
- Steps to reproduce, ideally with a minimal example.
- The SURGE version (`python -c "import surge; print(surge.__version__)"`)
  and Python / OS information.
- Any relevant logs, tracebacks, or artifacts.

We aim to acknowledge reports within **5 working days** and will keep the
reporter informed of progress. After a fix is released we will credit the
reporter in the release notes, unless the reporter prefers otherwise.

## Supported Versions

Security fixes land on the default branch and the latest released tag
(`0.1.0` is the first release). Older pre-release snapshots are not
supported.

| Version      | Supported |
| ------------ | --------- |
| `main`       | yes       |
| `0.1.x`      | yes |
| older snapshots | no     |

## Loading files and downloading data

- Model artifacts (`.joblib`, `.pt`) and pickle datasets (`.pkl`) are Python
  pickles and execute code when loaded. Load only files you created or trust.
- Benchmark datasets are downloaded over HTTPS from the cited public sources
  and are not checksum-verified in this release.
- `SURGE_INSECURE_SSL=1` and `SURGE_USE_SYSTEM_CA=1` are explicit opt-ins that
  change certificate handling.

## Scope

This policy covers the SURGE Python package and its associated CI
workflows. Vulnerabilities in third-party dependencies (PyTorch,
scikit-learn, NumPy, ONNX Runtime, etc.) should be reported to their
respective maintainers; feel free to CC the SURGE maintainer if the
issue materially affects SURGE users.

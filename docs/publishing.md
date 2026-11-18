# Publish MDPForge on PyPI

The prepared package version is `0.2.1` in `pyproject.toml`; the existing Git tag `v0.2.0` is kept. This preparation does not upload a release. The publication workflow runs manually and defaults to building artifacts only. PyPI uploads require a tag matching the package version.

## Check a release locally

From a clean checkout in the `benchmark` Conda environment:

```sh
conda run -n benchmark python -m pip install build twine
conda run -n benchmark python -m build --outdir artifacts/tmp/release/dist
conda run -n benchmark python -m twine check --strict artifacts/tmp/release/dist/*
```

Use an empty output directory for each release candidate. Inspect the source archive and wheel: they must include the three model collections' data, manifests and licenses. The project code is MIT; the existing package license expression also covers bundled CC-BY-4.0 and GPL-2.0-only sources. Keep those notices and files in the distributions.

Run the repository checks in `AGENTS.md` before releasing, then try the demonstration in `notebooks/compare_vi_pi_mpi.ipynb` from an installed wheel. The wheel's `notebooks` extra supplies JupyterLab, IPython's kernel and pandas; notebooks themselves remain in the GitHub repository.

## Configure publishing once

Create accounts on [PyPI](https://pypi.org/) and [TestPyPI](https://test.pypi.org/). Configure a pending Trusted Publisher for `mdpforge` on each service, using these exact values:

| Setting | Value |
| --- | --- |
| GitHub owner | `OrsoF` |
| Repository | `mdpforge` |
| Workflow filename | `publish.yml` |
| Environment | `testpypi` on TestPyPI; `pypi` on PyPI |

Create matching GitHub environments named `testpypi` and `pypi` in the repository settings. For `pypi`, restrict deployment to release tags and require review when available. Authentication uses GitHub's short-lived identity token; no PyPI API token is stored in the repository. See [PyPI's Trusted Publishing instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/) for the initial project setup.

## Build, rehearse, publish

1. Commit the reviewed release files and set a version that has not already been published.
2. Create and push a matching release tag, such as `v0.2.1`.
3. In GitHub Actions, open **Build and publish Python package**, select that tag, leave `target` as `build-only`, and inspect the resulting distributions.
4. Run the workflow for the same tag with `target=testpypi`. Check its metadata, README and downloaded wheel on TestPyPI.
5. When ready to release publicly, run the workflow for the same tag with `target=pypi`. Then create the matching GitHub release with its changes and limitations.

After publication, check installation in a clean Python environment with `python -m pip install "mdpforge[notebooks]==0.2.1"`. Download the demonstration notebook from the same GitHub tag and run it. Later releases need a new version and tag; published files cannot be replaced with different content under the same filename.

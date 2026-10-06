# Development instructions

## Development environment setup

- Create a venv: `python -m venv .venv`
- Activate the venv
- This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage dependencies, install uv: `pip install uv`
- Install the dependencies: `uv sync --extra=cli`
- Install pre-commit: `pre-commit install`
- Run CLI: `generalize --help`
- Run tests: `pytest`

## Requirements changes

To update requirements, do `uv lock --upgrade-package <package>`.

To add requirements, do `uv add <package>` or `uv add <package> --dev` for development requirements and `uv add <package> --group lint` for linting requirements.

## Code style

Included `.code-workspace` has necessary options set (linting, formatting, tests, extensions) set for VS Code.

## Commit message style

Commit messages should follow [Conventional Commits notation](https://www.conventionalcommits.org/en/v1.0.0/#summary).

## New algorithm steps

To create a completely new algorithm:

- Create a new file in `src/geogenalg/application`
- Create a new class (inheriting BaseAlgorithm, or another existing algorithm)
- Override `_execute` with algorithm implementation logic
- Any code that could reasonably be reused in other contexts should be not be placed in an algorithm, but the other modules (geometry.py, continuity.py etc.)

Index handling:

- The algorithm should handle indexes in such a way that:
  - No duplicate indexes should exist in the output
  - An index should be deterministically created
  - In case multiple features are merged together, the new index should be hashed from the original features' indexes (see `dissolve_and_inherit_attributes()` and `hash_index_from_old_ids`)
  - In case features are split to multiple features, the new index should be hashed from a combination of the original index and then new geometry (see `explode_and_hash_id` and `hash_duplicate_indexes`)
  - In case it cannot reasonably be tracked how (and what from) a new feature was formed, the new index can be hashed purely from the new geometry (see `hash_index_from_geometry`)
  - In other cases (no merging or splitting) involved, features should keep their original index
- Once these requirements are met, you can add the `@supports_identity` decorator to the algorithm class

Geometry validity:

- Generally speaking, an algorithm should produce GEOS-valid and simple geometries
- However, depending on what an algorithm does it can be difficult preempt all cases of invalid geometries potentially forming
  - For this reason, there is no hard requirement or error cases for invalid geometries, only a warning is given
  - Additionally, for each algorithm there is an option to automatically attempt to repair any invalid and non-simple geometries (`repair_result_geometries`)
    - This is intended only as a quick fix to allow applications to still run algorithms
    - As they get discovered, these _should_ be fixed at in the algorithm itself

Testing:

- If you introduce new functions, write unit tests for them in a corresponding test module in `tests/unit`
- If an algorithm class has methods, write unit tests for them in a new file under `tests/unit/application`
- If an algorithm method needs benchmarking, write benchmark test functions in a new file under `tests/bench/unit/application`
- Place algorithm test data in `tests/testdata/algo`
- Create a fixture for test input data in `tests/conftest.py`
- Create a fixture for an algorithm instance used in tests in `tests/conftest.py`
- Define an integration test in `tests/integration/test_algorithms.py`
- Define an algorithm benchmark in `tests/bench/algo/test_algorithms_benchmark.py`

[Read more about the testing framework here.](tests/README.md)

## Release steps

When the branch is in a releasable state, trigger the `Create draft release` workflow from GitHub Actions. Pass the to-be-released version number as an input to the workflow.

Workflow creates two commits in the target branch, one with the release state and one with the post-release state. It also creates a draft release from the release state commit with auto-generated release notes. Check the draft release notes and modify those if needed. After the release is published, the tag will be created, release workflow will be triggered, and it publishes a new version to PyPI.

Note: if you created the release commits to a non-`main` branch (i.e. to a branch with an open pull request), only publish the release after the pull request has been merged to main branch. Change the commit hash on the draft release to point to the actual rebased commit on the main branch, instead of the now obsolete commit on the original branch. If the GUI dropdown selection won't show the new main branch commits, the release may need to be re-created manually to allow selecting the rebased commit hash.

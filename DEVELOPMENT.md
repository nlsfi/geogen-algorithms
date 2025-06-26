# Development instructions

## Development environment setup

- Create a venv: `python -m venv .venv`
- Activate the venv
- This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage dependencies, install uv: `pip install uv`
- Install the dependencies: `uv sync`
- Install pre-commit: `pre-commit install`
- Run tests: `pytest`

## Requirements changes

To update requirements, do `uv lock --upgrade-package <package>`.

To add requirements, do `uv add <package>` or `uv add <package> --dev` for development requirements and `uv add <package> --group lint` for linting requirements.

## Code style

Included `.code-workspace` has necessary options set (linting, formatting, tests, extensions) set for VS Code.

## Commit message style

Commit messages should follow [Conventional Commits notation](https://www.conventionalcommits.org/en/v1.0.0/#summary).

## Release steps

When the branch is in a releasable state, trigger the `Create draft release` workflow from GitHub Actions. Pass the to-be-released version number as an input to the workflow.

Workflow creates two commits in the target branch, one with the release state and one with the post-release state. It also creates a draft release from the release state commit with auto-generated release notes. Check the draft release notes and modify those if needed. After the release is published, the tag will be created, release workflow will be triggered, and it publishes a new version to PyPI.

Note: if you created the release commits to a non-`main` branch (i.e. to a branch with an open pull request), only publish the release after the pull request has been merged to main branch. Change the commit hash on the draft release to point to the actual rebased commit on the main branch, instead of the now obsolete commit on the original branch. If the GUI dropdown selection won't show the new main branch commits, the release may need to be re-created manually to allow selecting the rebased commit hash.

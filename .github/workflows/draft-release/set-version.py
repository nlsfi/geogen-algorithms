import datetime
import sys
from pathlib import Path

version = sys.argv[1]
date = datetime.datetime.now(tz=datetime.UTC).date().isoformat()  # noqa: SC200

# Version in Changelog file

changelog_file = Path("CHANGELOG.md")
error_msg = "Changelog not in correct format"

for line in changelog_file.read_text(encoding="utf-8").splitlines():
    if line.startswith("##"):
        if line != "## Unreleased":
            raise ValueError(error_msg)
        break
else:
    raise ValueError(error_msg)

link_line = (
    f"[{version}]: https://github.com/nlsfi/geogen-algorithms/releases/tag/v{version}\n"
)

changelog_file.write_text(
    changelog_file.read_text(encoding="utf-8").replace(
        "## Unreleased", f"## [{version}] - {date}", 1
    )
    + link_line,
    encoding="utf-8",
)

# Version in init file

init_file = Path("src/geogenalg/__init__.py")
error_msg = "Init file not in correct format"

init_line_to_replace = None

for line in init_file.read_text(encoding="utf-8").splitlines():
    if line.startswith("__version__ ="):
        init_line_to_replace = line
        break
else:
    raise ValueError(error_msg)

init_file.write_text(
    init_file.read_text(encoding="utf-8").replace(
        init_line_to_replace, f'__version__ = "{version}"', 1
    ),
    encoding="utf-8",
)

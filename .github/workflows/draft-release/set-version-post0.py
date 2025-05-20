import sys
from pathlib import Path

version = sys.argv[1]

# Version in Changelog file

changelog_file = Path("CHANGELOG.md")
changelog_text_to_modify = f"## [{version}]"
new_section = "## Unreleased\n\n"

changelog_file.write_text(
    changelog_file.read_text(encoding="utf-8").replace(
        changelog_text_to_modify, new_section + changelog_text_to_modify, 1
    ),
    encoding="utf-8",
)

# Version in init file

init_file = Path("src/geogenalg/__init__.py")
init_line_to_replace = f'__version__ = "{version}"'

init_file.write_text(
    init_file.read_text(encoding="utf-8").replace(
        init_line_to_replace, f'__version__ = "{version}.post0"', 1
    ),
    encoding="utf-8",
)

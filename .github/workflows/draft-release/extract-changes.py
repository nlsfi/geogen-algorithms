import sys
from pathlib import Path

version = sys.argv[1]

changelog_file = Path("CHANGELOG.md")
to_find = f"## [{version}]"

found = False
results = []

for line in changelog_file.read_text(encoding="utf-8").splitlines():
    if found and line.startswith("## "):
        break
    elif found:
        results.append(line)
    elif line.startswith(to_find):
        found = True

if len(results) == 0:
    error_msg = "No changes found"
    raise ValueError(error_msg)

if not results[0]:
    results = results[1:]
if not results[-1]:
    results = results[:-1]

sys.stdout.write("\n".join(results) + "\n")

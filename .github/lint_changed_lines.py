"""
report ruff violations that fall on lines this change added or modified

The package carries a large pre-existing violation count, so linting whole files would fail on
any edit to a legacy file and the gate would be routed around. Linting the added lines keeps the
check green today and still stops new debt.

Usage:
    python .github/lint_changed_lines.py <base-sha> <head-sha>

Exits 1 if any violation lands on an added line, 0 otherwise.
"""

# packages
import re
import subprocess
import sys
from typing import Dict, List, Set

HUNK = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@')
VIOLATION = re.compile(r'^(.+?):(\d+):\d+: (.+)$')


def added_lines(base: str,
                head: str,
                ) -> Dict[str, Set[int]]:
    """
    Line numbers added or modified per file, read from a zero-context diff.

    Args:
        base: base commit
        head: head commit

    Returns:
        mapping of file path to the set of added line numbers
    """
    diff = subprocess.run(['git', 'diff', '-U0', base, head, '--', '*.py'],
                          capture_output=True, text=True, check=True).stdout
    lines_by_file: Dict[str, Set[int]] = {}
    current = None
    for line in diff.splitlines():
        if line.startswith('+++ b/'):
            current = line[len('+++ b/'):]
            lines_by_file.setdefault(current, set())
        elif current is not None:
            match = HUNK.match(line)
            if match is not None:
                start, count = int(match.group(1)), int(match.group(2) or 1)
                lines_by_file[current].update(range(start, start + count))
    return lines_by_file


def main(base: str,
         head: str,
         ) -> int:
    """
    Run ruff on the changed files and keep only violations on added lines.

    Args:
        base: base commit
        head: head commit

    Returns:
        1 if any violation lands on an added line, 0 otherwise
    """
    lines_by_file = {path: lines for path, lines in added_lines(base=base, head=head).items()
                     if len(lines) > 0}
    if len(lines_by_file) == 0:
        print('no python lines added or modified')
        return 0

    result = subprocess.run(['ruff', 'check', '--output-format=concise', *lines_by_file],
                            capture_output=True, text=True)
    offending: List[str] = []
    for line in result.stdout.splitlines():
        match = VIOLATION.match(line)
        if match is not None and int(match.group(2)) in lines_by_file.get(match.group(1), set()):
            offending.append(line)

    print(f"linted {len(lines_by_file)} changed file(s); "
          f"{len(offending)} violation(s) on added lines")
    for line in offending:
        print(f"  {line}")
    return 1 if len(offending) > 0 else 0


if __name__ == '__main__':
    sys.exit(main(base=sys.argv[1], head=sys.argv[2]))

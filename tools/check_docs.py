"""Check the portable documentation standard without importing qis or running examples.

With no arguments, check adopted pages and report the remaining migration inventory. Use
``--files docs/topic.md`` for a revision batch, or ``--all`` to require complete adoption.
Code examples are excluded from prose checks. This is a source check, not a TeX renderer,
link checker, or numerical validator; those checks remain separate.
"""

import argparse
import re
from pathlib import Path
from typing import NamedTuple, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_URL = 'https://github.com/ArturSepp/QuantInvestStrats'
CITATION_URL = f'{PROJECT_URL}/blob/main/CITATION.cff'
PLACEHOLDER = '*[author / affiliation / date — placeholder]*'
ARTICLE_HEADINGS = (
    'Overview',
    'Inputs, notation, and assumptions',
    'Methodology',
    'Worked example',
    'Implementation in qis',
    'Interpretation and limitations',
    'See also',
    'References',
)
METHODOLOGY_PAGES = frozenset({
    'brinson_attribution.md', 'factsheets_and_reporting.md', 'frequency_convention_note.md',
    'fx_hedging_and_market_data.md', 'incomplete_and_mixed_frequency_data.md',
    'model_layer_attribution.md', 'performance_analytics_and_sharpe.md',
    'portfolio_backtesting.md', 'portfolio_breadth.md', 'private_asset_unsmoothing.md',
    'reproducibility.md', 'stress_testing.md', 'stress_testing_with_options.md',
    'tracking_error_and_risk.md',
    'turnover_conventions.md',
})
UTILITY_PAGES = frozenset({
    'documentation_standard.md', 'factsheets.md', 'gallery.md', 'index.md', 'install.md',
    'package_comparison.md', 'quickstart.md', 'REMOVED_5_0.md', 'software_design.md',
    'portfolio_stress.md',
})
# Adoption is explicit. Do not infer it from a byline or let new pages evade the inventory.
ADOPTED_PAGES = frozenset({
    'documentation_standard.md', 'index.md', 'portfolio_breadth.md',
    'frequency_convention_note.md', 'brinson_attribution.md',
    'performance_analytics_and_sharpe.md', 'tracking_error_and_risk.md',
    'portfolio_backtesting.md', 'turnover_conventions.md',
    'incomplete_and_mixed_frequency_data.md', 'private_asset_unsmoothing.md',
    'fx_hedging_and_market_data.md', 'stress_testing.md', 'stress_testing_with_options.md',
    'model_layer_attribution.md', 'reproducibility.md',
    'factsheets_and_reporting.md', 'factsheets.md', 'gallery.md',
    'install.md', 'quickstart.md', 'software_design.md', 'package_comparison.md',
    'REMOVED_5_0.md', 'portfolio_stress.md',
})
FENCE = re.compile(r'^ {0,3}(`{3,}|~{3,})(.*)$')
HEADING = re.compile(r'^(#{1,6})\s+(.+?)\s*#*\s*$')
LINK = re.compile(r'(?<!!)\[[^\]\n]+\]\((https://[^\s)]+)\)')
BYLINE = re.compile(r'^\*Author: .+ / Affiliation: .+ / Date: \d{4}-\d{2}-\d{2}\*$')


class Issue(NamedTuple):
    """One source-level documentation problem.

    Attributes:
        line: One-based source line, or one for a document-wide problem.
        message: Explanation of the violated convention.
    """

    line: int
    message: str


def prose_lines(text: str) -> tuple[list[tuple[int, str]], list[Issue], str]:
    """Extract prose while respecting YAML front matter and nested example fences.

    Args:
        text: Markdown source; no files are modified.

    Returns:
        Visible lines, syntax issues, and the front-matter body.
    """
    lines = text.splitlines()
    issues = []
    metadata = ''
    first = 0
    if lines and lines[0] == '---':
        closing = next((i for i in range(1, len(lines)) if lines[i] == '---'), None)
        if closing is None:
            return [], [Issue(1, 'Unclosed YAML front matter.')], ''
        metadata = '\n'.join(lines[1:closing])
        first = closing + 1
    visible = []
    fence = ''
    fence_line = 0
    math_line = 0
    comment = False
    for index in range(first, len(lines)):
        line = lines[index]
        number = index + 1
        matched = FENCE.match(line)
        if fence:
            if (matched and matched[1][0] == fence[0] and len(matched[1]) >= len(fence)
                    and not matched[2].strip()):
                fence = ''
            continue
        if math_line:
            if line.strip() == '$$':
                math_line = 0
                if index + 1 < len(lines) and lines[index + 1].strip():
                    issues.append(Issue(number, 'Put a blank line after display mathematics.'))
            continue
        # HTML comments must not satisfy a missing byline, section, or citation.
        if comment:
            if '-->' in line:
                comment = False
                line = line.split('-->', 1)[1]
            else:
                continue
        while '<!--' in line:
            before, after = line.split('<!--', 1)
            if '-->' in after:
                line = before + after.split('-->', 1)[1]
            else:
                line = before
                comment = True
        matched = FENCE.match(line)
        if matched:
            fence, fence_line = matched[1], number
            if re.match(r'^(?:\{math\}|math)(?:\s|$)', matched[2].strip()):
                issues.append(Issue(number, 'Use standalone $$ display blocks, not math fences.'))
            continue
        if line.startswith(('    ', '\t', '>')):
            continue  # indented code and quoted examples are not article structure
        # Inline code can demonstrate a delimiter without being a display expression.
        for code in re.finditer(r'(`+).*?\1', line):
            if re.search(r'\{(?:math|eq)\}$', line[:code.start()]):
                issues.append(Issue(number, 'Use dollar math and ordinary equation-section links.'))
        without_code = re.sub(r'(`+).*?\1', '', line)
        if '$$' in without_code:
            if line.strip() == '$$':
                math_line = number
                if index > 0 and lines[index - 1].strip():
                    issues.append(Issue(number, 'Put a blank line before display mathematics.'))
            else:
                issues.append(Issue(number, 'Put each display $$ delimiter on its own line.'))
            continue
        if re.search(r'\\[\[\]()]', without_code):
            issues.append(Issue(number, 'Use dollar delimiters for portable mathematics.'))
        visible.append((number, line))
    if fence:
        issues.append(Issue(fence_line, 'Unclosed code fence.'))
    if math_line:
        issues.append(Issue(math_line, 'Unclosed display mathematics.'))
    if comment:
        issues.append(Issue(len(lines), 'Unclosed HTML comment.'))
    return visible, issues, metadata


def check_document(text: str, *, methodology: bool) -> list[Issue]:
    """Check one article's metadata, structure, byline, references, and math source.

    Args:
        text: Complete Markdown source.
        methodology: Whether the full methodology section order is required.

    Returns:
        Source issues. An empty list means these source checks passed, not that TeX rendered.
    """
    visible, issues, metadata = prose_lines(text)
    description = re.search(r'(?m)^[ \t]+description:[ \t]*([^\n]*)', metadata)
    if not description or 'html_meta:' not in metadata or 'myst:' not in metadata:
        issues.append(Issue(1, 'Provide a myst.html_meta.description in front matter.'))
    elif description[1].strip() in ('', "''", '\"\"'):
        issues.append(Issue(1, 'The page description must not be empty.'))
    elif description[1].strip() in ('>', '>-', '|', '|-'):
        following = metadata[description.end():].strip()
        if not following:
            issues.append(Issue(1, 'The page description must not be empty.'))
    headings = [(number, len(match[1]), match[2]) for number, line in visible
                if (match := HEADING.match(line))]
    titles = [heading for heading in headings if heading[1] == 1]
    if len(titles) != 1 or not headings or headings[0][1] != 1:
        issues.append(Issue(1, 'Start with exactly one H1 title.'))
    previous = 0
    for number, level, _ in headings:
        if level > previous + 1:
            issues.append(Issue(number, 'Do not skip heading levels.'))
        previous = level
    title_line = titles[0][0] if titles else 0
    opening = [line for number, line in visible if title_line < number <= title_line + 12]
    if not any(line == PLACEHOLDER or BYLINE.fullmatch(line) for line in opening):
        issues.append(Issue(title_line or 1,
                            'Put the author/affiliation/date byline after the title.'))
    prose = '\n'.join(line for _, line in visible)
    links = set(LINK.findall(prose))
    if PROJECT_URL not in links and PROJECT_URL + '/' not in links:
        issues.append(Issue(1, 'Link to the qis project repository in article prose.'))
    if CITATION_URL not in links:
        issues.append(Issue(1, 'Link to the canonical qis CITATION.cff in article prose.'))
    if methodology:
        sections = [title for _, level, title in headings if level == 2]
        if sections != list(ARTICLE_HEADINGS):
            issues.append(Issue(1, 'Use each required methodology H2 once, in the standard order.'))
    return issues


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Validate adopted, selected, or all documentation pages and report pending migration.

    Args:
        argv: Optional command-line arguments, excluding the program name.

    Returns:
        Zero for passing selected pages, one for failed checks; argparse rejects invalid input.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--files', nargs='+', type=Path, help='Pages to validate now.')
    selection.add_argument('--all', action='store_true', help='Require every page to comply.')
    args = parser.parse_args(argv)
    docs = REPO_ROOT.joinpath('docs').resolve()
    discovered = {path.name for path in docs.glob('*.md')}
    inventory = METHODOLOGY_PAGES | UTILITY_PAGES
    errors = []
    if args.files:
        selected = set()
        for path in args.files:
            path = (REPO_ROOT / path).resolve()
            if path.parent != docs or path.suffix != '.md':
                parser.error(f'Expected a top-level docs/*.md page: {path}')
            selected.add(path.name)
    else:
        selected = discovered | inventory if args.all else set(ADOPTED_PAGES)
    for name in sorted(discovered - inventory):
        errors.append(f'docs/{name}:1: Add this page to the explicit documentation inventory.')
    for name in sorted(selected):
        path = docs / name
        if name not in inventory:
            if name not in discovered:
                errors.append(f'docs/{name}:1: Unknown documentation page.')
            continue
        if not path.is_file():
            errors.append(f'docs/{name}:1: Missing documentation page.')
            continue
        for issue in check_document(path.read_text(encoding='utf-8'),
                                    methodology=name in METHODOLOGY_PAGES):
            errors.append(f'docs/{name}:{issue.line}: {issue.message}')
    for error in errors:
        print(error)
    if errors:
        print(f'FAIL: {len(errors)} documentation issues.')
        return 1
    print(f'PASS: {len(selected)} selected documentation pages.')
    pending = sorted(inventory - ADOPTED_PAGES - selected)
    if not args.all and pending:
        print(f'PENDING (not validated): {len(pending)} pages: {", ".join(pending)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

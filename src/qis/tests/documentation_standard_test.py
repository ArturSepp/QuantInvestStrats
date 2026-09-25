"""Documentation regressions that a successful Sphinx build alone cannot detect."""

import runpy
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKER_PATH = REPO_ROOT / 'tools' / 'check_docs.py'
if not CHECKER_PATH.is_file():
    pytest.skip('Documentation tooling is not shipped in the wheel.', allow_module_level=True)

CHECKER = runpy.run_path(str(CHECKER_PATH))
CHECK = CHECKER['check_document']
PROJECT = 'https://github.com/ArturSepp/QuantInvestStrats'
AUTHOR_BYLINE = '*Author: [Artur Sepp](https://github.com/ArturSepp)*'
DATED_BYLINE = (AUTHOR_BYLINE[:-1] + ' / First recorded: [2026-09-06]('
                + PROJECT + '/commit/' + 'a' * 40 + ')*')
HEADER = f'''---
myst:
  html_meta:
    description: >-
      A reproducible analytical method implemented in qis.
---

# An analytical method

{DATED_BYLINE}

Implemented in [qis]({PROJECT}).
Software reference: [CITATION.cff]({PROJECT}/blob/main/CITATION.cff).
'''
SECTIONS = '''
## Overview

Define the method before its implementation.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Not applicable |
| Sampling grid | Supplied dates |
| Annualisation | None |
| Mean adjustment | None |
| Timing | Weights at each date |
| Output units | Count |
| qis default | Not applicable |

Let $p_i$ be an absolute capital share.

## Methodology

$$
N = \\frac{1}{\\sum_i p_i^2}.
$$

This count describes concentration.

## Worked example

Four equal shares have an effective count of four.

## Implementation in qis

Use the documented public entry point.

## Interpretation and limitations

Concentration alone does not establish independent risk.

## See also

Related risk methods.

## References

The software citation appears above.
'''


def test_complete_article_and_short_utility_page_pass():
    assert not CHECK(HEADER + SECTIONS, methodology=True)
    assert not CHECK(HEADER, methodology=False)
    assert CHECK(HEADER, methodology=True)


@pytest.mark.parametrize('before,after,message', [
    (DATED_BYLINE, '', 'byline'),
    (f'[qis]({PROJECT})', 'qis', 'project repository'),
    (f'[CITATION.cff]({PROJECT}/blob/main/CITATION.cff)', 'CITATION.cff', 'CITATION.cff'),
    ('    description: >-\n      A reproducible analytical method implemented in qis.',
     '', 'description'),
    ('    description: >-\n      A reproducible analytical method implemented in qis.',
     '    description: ""', 'description must not be empty'),
    ('## Methodology', '#### Methodology', 'heading levels'),
    ('## Worked example', '# Another title', 'one H1'),
    ('## Worked example', '## Overview', 'required methodology'),
    ('## Overview', '## Worked example', 'required methodology'),
    ('## Worked example', '## Extra section\n\n## Worked example', 'required methodology'),
    ('$$\nN = \\frac{1}{\\sum_i p_i^2}.\n$$',
     '```{math}\nN = \\frac{1}{\\sum_i p_i^2}.\n```', 'math fences'),
    ('$$\nN = \\frac{1}{\\sum_i p_i^2}.\n$$',
     '```{math} labelled-equation\nN = 4\n```', 'math fences'),
    ('$$\nN = \\frac{1}{\\sum_i p_i^2}.\n$$', '$$N = 4.$$', 'own line'),
    ('$$\n\nThis count', '$$\nThis count', 'blank line after'),
    ('$p_i$', '{math}`p_i`', 'ordinary equation-section links'),
    ('$p_i$', '\\(p_i\\)', 'portable mathematics'),
    ('| Convention | This article |', '| Setting | Value |', 'convention card'),
    ('| Timing | Weights at each date |\n', '', 'convention card rows'),
    ('| Output units |', '| Units |', 'convention card rows'),
    ('Let $p_i$', 'Let $p^\\intercal$', 'transpose'),
    ('Let $p_i$', 'Let $p^\\mathsf{T}$', 'transpose'),
    ('Let $p_i$', 'Let $\\mathrm{Var}(p)$', 'operatorname'),
    ('Let $p_i$', 'Let sqrt(p) and', 'not plain text'),
])
def test_article_defects_are_rejected(before, after, message):
    source = HEADER + SECTIONS
    assert before in source, 'The defect injection must actually change the fixture.'
    issues = CHECK(source.replace(before, after, 1), methodology=True)
    assert any(message in issue.message for issue in issues), issues


def test_code_and_comments_cannot_supply_missing_metadata_or_structure():
    hidden_header = '````markdown\n' + HEADER + '\n````\n'
    issues = CHECK('# Real title\n\n' + hidden_header + SECTIONS, methodology=True)
    assert any('byline' in issue.message for issue in issues)
    assert any('project repository' in issue.message for issue in issues)
    issues = CHECK(HEADER + '<!--\n' + SECTIONS + '\n-->', methodology=True)
    assert any('required methodology' in issue.message for issue in issues)


def test_nested_teaching_fences_do_not_trigger_article_math_errors():
    example = '\n````markdown\n# Example\n```{math}\nx = 1\n```\n$$x$$\n````\n'
    assert not CHECK(HEADER + SECTIONS + example, methodology=True)
    inline_example = '\nUse `$...$`, not `{math}` or `{eq}` roles or `\\(x\\)`.\n'
    assert not CHECK(HEADER + SECTIONS + inline_example, methodology=True)


@pytest.mark.parametrize('ending,message', [
    ('\n```python\nx = 1\n', 'Unclosed code'),
    ('\n$$\nx = 1\n', 'Unclosed display'),
    ('\n<!-- hidden\n', 'Unclosed HTML'),
])
def test_unclosed_source_blocks_are_rejected(ending, message):
    issues = CHECK(HEADER + SECTIONS + ending, methodology=True)
    assert any(message in issue.message for issue in issues)


def test_cli_distinguishes_pending_pages_from_all_pages(tmp_path, monkeypatch, capsys):
    docs = tmp_path / 'docs'
    docs.mkdir()
    globals_ = CHECKER['main'].__globals__
    monkeypatch.setitem(globals_, 'REPO_ROOT', tmp_path)
    inventory = CHECKER['METHODOLOGY_PAGES'] | CHECKER['UTILITY_PAGES']
    pending = 'install.md'
    adopted = inventory - {pending}
    monkeypatch.setitem(globals_, 'ADOPTED_PAGES', adopted)
    for name in inventory:
        source = HEADER + SECTIONS if name in CHECKER['METHODOLOGY_PAGES'] else HEADER
        if name not in adopted:
            source = '# Pending\n'
        (docs / name).write_text(source, encoding='utf-8')
    assert CHECKER['main']([]) == 0
    assert 'PENDING (not validated)' in capsys.readouterr().out
    assert CHECKER['main'](['--all']) == 1
    assert 'FAIL:' in capsys.readouterr().out
    (docs / pending).write_text(HEADER, encoding='utf-8')
    assert CHECKER['main'](['--all']) == 0
    assert 'PENDING' not in capsys.readouterr().out
    monkeypatch.setitem(globals_, 'ADOPTED_PAGES', inventory)
    assert CHECKER['main']([]) == 0
    assert 'PENDING' not in capsys.readouterr().out
    (docs / 'new_topic.md').write_text(HEADER, encoding='utf-8')
    assert CHECKER['main']([]) == 1
    assert 'explicit documentation inventory' in capsys.readouterr().out


def test_adopted_repository_pages_pass():
    assert CHECKER['main']([]) == 0


@pytest.mark.parametrize('byline', [
    AUTHOR_BYLINE,
    DATED_BYLINE,
    DATED_BYLINE.replace('2026-09-06', '2024-02-29'),
])
def test_linked_author_and_evidenced_date_pass(byline):
    """Allow confirmed authors without inventing an affiliation or uncommitted date."""
    assert not CHECK(HEADER.replace(DATED_BYLINE, byline), methodology=False)


@pytest.mark.parametrize('byline', [
    '*[author / affiliation / date — placeholder]*',
    AUTHOR_BYLINE.replace('[Artur Sepp](https://github.com/ArturSepp)', 'Artur Sepp'),
    AUTHOR_BYLINE.replace('https://github.com/', 'https://example.com/'),
    DATED_BYLINE.replace('2026-09-06', '2026-02-30'),
    DATED_BYLINE.replace('a' * 40, 'abcdef'),
    DATED_BYLINE.replace(PROJECT + '/commit/', PROJECT + '/tree/'),
    DATED_BYLINE.replace(PROJECT + '/commit/', 'https://example.com/commit/'),
])
def test_invalid_author_metadata_is_rejected(byline):
    """Reject placeholders, broken attribution and dates without a valid evidence link."""
    issues = CHECK(HEADER.replace(DATED_BYLINE, byline), methodology=False)
    assert any('byline' in issue.message for issue in issues)

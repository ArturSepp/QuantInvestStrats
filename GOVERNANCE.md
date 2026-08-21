# Governance, maintenance, and support

`qis` is an open-source research-software project for performance analytics, portfolio
backtesting, risk analysis, and factsheet reporting. This document states how decisions are made,
how the project is maintained, and what contributors and users can expect.

## Maintainer and decision model

Artur Sepp is the lead maintainer and currently has final responsibility for project scope,
technical decisions, releases, and repository administration. Development takes place through the
public GitHub repository. Contributors are encouraged to raise design questions in an issue before
starting a substantial change.

Decisions are normally discussed in issues and pull requests so their rationale remains public.
The lead maintainer may accept, request changes to, defer, or decline a proposal. There is currently
no steering committee or guaranteed maintainer succession process. A contributor who demonstrates
sustained technical judgement, constructive review, and commitment to the project may be invited
to take on review or maintenance responsibilities.

## Technical decision principles

Changes are evaluated against these priorities:

1. Numerical correctness and the absence of look-ahead in backtests.
2. Explicit return, frequency, annualisation, and Sharpe conventions.
3. Preservation of held-unit rebalancing semantics and public behavior unless a documented
   correction is required.
4. A core installation that remains testable offline and does not require proprietary data.
5. Backward compatibility of the exported API, with explicit changelog and version treatment for
   public-signature or behavioral changes.
6. Cross-platform maintainability, reproducible tests, and documentation that a reviewer can run
   without access to the maintainer's environment.

`qis` is the base analytics layer of the maintainer's package stack. Portfolio optimisation belongs
in `optimalportfolios`, generic factor estimation in `factorlasso`, and Bloomberg integration in
`bbg-fetch`. A proposal outside these boundaries may be useful but will normally be directed to
the appropriate project rather than duplicated here.

## Proposing and contributing changes

Small bug fixes and documentation improvements may go directly to a pull request. Open an issue
first for a new numerical convention, hard dependency, public API change, or change spanning
several subsystems.

The development commands and testing expectations are documented in
[CONTRIBUTING.md](CONTRIBUTING.md). Pull requests are reviewed for scope, numerical correctness,
tests, documentation, compatibility, and effects on published or reproducible results. A numerical
fix should include a test that fails under the defect and, where practical, an independent
reference calculation.

## Releases and compatibility

Releases are versioned, recorded in [CHANGELOG.md](CHANGELOG.md), and distributed through
[PyPI](https://pypi.org/project/qis/). Formal releases normally have matching package metadata, a
Git tag, and a GitHub Release. There is no fixed release cadence.

Supported Python versions and dependency floors are declared in `pyproject.toml` and exercised in
continuous integration. Public API or default changes require explicit changelog and version
treatment. When practical, a replacement is introduced before an API is removed. Urgent
correctness, compatibility, or security fixes may require a shorter deprecation period, which will
be documented.

Published numerical results are not changed merely to make a test pass. If a correction affects a
published convention or reproduction, the difference must be explained and the relevant
independent checks rerun.

## Support and sensitive reports

Use the public [issue tracker](https://github.com/ArturSepp/QuantInvestStrats/issues):

- use the bug-report form for reproducible defects;
- open a normal issue for usage or methodology questions;
- identify the publication and section for questions about a paper;
- use generated or public data rather than proprietary inputs.

Support is provided on a best-effort basis. The project has no service-level agreement, guaranteed
response time, or entitlement to individual portfolio advice.

GitHub issues are public. Do not post credentials, licensed datasets, client information, or other
sensitive material. For a security or privacy issue that should not be disclosed publicly, contact
the maintainer at `artursepp@gmail.com` with “qis” in the subject. This route does not create a
guaranteed response-time commitment.

## Conduct and disagreements

Be civil, assume good faith, and discuss the technical claim rather than the person. Harassment,
personal attacks, or publication of another person's private information are not acceptable.
Technical disagreement is welcome. If discussion reaches an impasse, the lead maintainer records
the decision and its scope; a declined proposal may be revisited when new evidence or a smaller
design is available.

## Changes to this policy

Governance changes use the same public issue and pull-request process as other project changes.
Material changes should explain why the current policy is insufficient and identify any new
responsibilities or promises they create.

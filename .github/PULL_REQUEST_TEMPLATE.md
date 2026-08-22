## What changed

Describe the problem and the smallest coherent change that solves it.

## Verification

List the exact commands run and their results.

## Checklist

- [ ] Tests cover the changed public behavior or defect.
- [ ] Point-in-time code uses no observations later than the decision time.
- [ ] Return, frequency, annualisation, and missing-data conventions remain explicit.
- [ ] No credentials, proprietary data, local paths, generated outputs, or agent reports are included.
- [ ] The changed-lines ruff gate and production docstring gate pass.
- [ ] User-visible changes are documented in `CHANGELOG.md` and relevant docs.
- [ ] New runtime dependencies or public-signature changes are called out explicitly.

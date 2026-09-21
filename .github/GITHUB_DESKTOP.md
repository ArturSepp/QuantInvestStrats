# Committing with GitHub Desktop

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Create a working branch, select the intended changes, and commit normally. The installed
hook checks the selected contents. Push the branch, open a pull request, and merge after
**Required checks** passes. Main is protected; a failing branch does not change main.

The [shared Desktop guide](https://github.com/ArturSepp/ArturSepp/blob/main/docs/github_desktop.md)
explains setup, repair messages, partial commits, and the longer local checks.
GitHub Desktop can bypass a local hook for a work-in-progress commit; remote checks still apply.
No hook automatically stages or changes files. External-link and live-dependency maintenance
runs are labelled separately from the required checks on a proposed change.

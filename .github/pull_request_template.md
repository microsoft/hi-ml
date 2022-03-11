Write a detailed description of your Pull Request (PR) here.

---

## Guidelines

Please follow the guidelines for PRs contained [here](/CONTRIBUTING.md). Checklist:

- [ ] Ensure that your PR is small, and implements one change
- [ ] Give your PR title one of the prefixes ENH, BUG, STYLE, DOC, DEL to indicate what type of change that is (see [CONTRIBUTING](/CONTRIBUTING.md))
- [ ] Link the correct GitHub issue for tracking
- [ ] Add unit tests for all functions that you introduced or modified
- [ ] Run automatic code formatting / linting on all files ("Format Document" Shift-Alt-F in VSCode)
- [ ] Ensure that documentation renders correctly in Sphinx (run Sphinx via `make html` in the `docs` folder)

## Change the default merge message

When completing your PR, you will be asked for a title and an optional extended description. By default, the extended description will be a concatenation of the individual
commit messages. Please DELETE/REPLACE that with a human readable extended description for non-trivial PRs.

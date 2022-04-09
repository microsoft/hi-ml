# Contribution and Coding Guidelines for HI-ML

## Design Phase

For non-trivial changes, please communicate your plans early on to reach maximum impact and to avoid duplicate work:

* What are planning you working on?
* Why is it needed for the project?
* Why is it needed in the repository?
* Ask the rest of the team if they have already worked on it or thought about it. This helps to get to know about existing code fragments that can simplify your task, and integrate better with existing code.

For non-trivial changes, we recommend to work with a team member to create a design for your work.
Depending on the size of the task at hand, a design can be anywhere between a short descrption of
your plan on Github, or a long document where you list your design options and invite feedback.

Your design partner will later also be well positioned to review your PR because they are familiar with your plan.

## Setting up your dev environment

Please see the detailed instructions [here](developers.md).

## Coding Guidelines

### Naming

Please follow the general Python rules for naming:

* Variables, function and method names should use `snake_case` (lower case with under scores separating words)
* Class names should follow `PascalCase` (first letter of each word capitalized).

In addition:

* To improve readability, functions or methods that return boolean values should follow a `is_...`, `has_...`, `use...` pattern, like `is_status_ok` instead of `ok`.

### Static Analysis and Linting

We use `flake8` as a linter, and `mypy` and pyright for static typechecking. Both tools run as part of the PR build, and must run without errors for a contribution to be accepted. `mypy` requires that all functions and methods carry type annotations, see [mypy documentation](https://mypy.readthedocs.io/en/latest/getting_started.html#function-signatures-and-dynamic-vs-static-typing)

We highly recommend to run all those tools _before_ pushing the latest changes to a PR. If you have `make` installed, you can run both tools in one go via `make check` (from the repository root folder)

### String Style

End sentences in docstrings with a period:

```pythong
def method(self, arg: int) -> None:
    """
    Method description, followed by an period.
    """
```

* End strings in error messages with a period.

### Documentation

We are using Sphinx for generating documentation. We recommend using a VSCode extension for auto-generating documentation templates, for example `njpwerner.autodocstring` (this extension is already pre-configured in VSCode's workspace settings file).

Common mistakes when writing docstrings:

* There must be a separating line between a function description and the documentation for its parameters.
* In multi-line parameter descriptions, continuations on the next line must be indented.
* Sphinx will merge the class description and the arguments of the constructor `__init__`. Hence, there is no need to
  write any text in the constructor docstring other than the constructor arguments.
* Use `>>>` to include code snippets. PyCharm will run intellisense on those to make authoring easier.
* To generate the Sphinx documentation on your dev machine, run `make html` in the `./docs` folder, and then
  open `./docs/build/html/index.html`

Example:

```python
class Foo:
    """
    This is the class description.

    The following block will be pretty-printed by Sphinx. Note the space between >>> and the code!
    
    Usage example:
        >>> from module import Foo
        >>> foo = Foo(bar=1.23)
    """

    ANY_ATTRIBUTE = "what_ever."
    """Document class attributes after the attribute."""

    def __init__(self, bar: float = 0.5) -> None:
        """
        :param bar: This is a description for the constructor argument.
            Long descriptions should be indented.
        """
        self.bar = bar

    def method(self, arg: int) -> None:
        """
        Method description, followed by an empty line. Argument names like ``arg`` are rendered nicely
        if enclosed in double backtick.

        This method can raise a :exc:`ValueError`.
        
        :param arg: This is a description for the method argument.
            Long descriptions should be indented.
        """
```

## Testing

### Why do we write unit tests?

If you think that there are sensitive parts in your code, where hidden errors can impact the project outcome, please take your time to protect parts of your code. As the team grows, so does the number of  users of the repository, and we may not be able to monitor changes on a frequent basis, but your tests can ensure that the expected behavior is preserved. We are not making it strictly mandatory, however users shall take the liability of untested code and its potential downstream impact.

We do not need to test every part of the code depending on the project constraints, e.g., quality and test-coverage vs learning and hypothesis-testing. However, we expect a justification in each PR for this decision and it should be reviewed by team members. Please regularly monitor the code coverage results that are posted as comments on each pull request.

In particular, please do write unit tests for anything that affects training/results such as data pre-processing, losses, and metrics.

### Testing Do's / Don'ts

* DO write unit tests for each new function or class that you add.
* DO extend unit tests for existing functions or classes if you change their core behaviour.
* DO try your best to write unit tests that are fast. Very often, this can be done by reducing data size to a minimum. Also, it is helpful to avoid long-running integration tests, but try to test at the level of the smallest involved function.
* DO ensure that your tests are designed in a way that they can pass on the local machine, even if they are relying on specific cloud features. If required, use `unittest.mock` to simulate the cloud features, and hence enable the tests to run successfully on your local machine.
* DO run all unit tests on your dev machine before submitting your changes. The test suite is designed to pass completely also outside of cloud builds.
* DO NOT rely only on the test builds in the cloud (i.e., run test locally before submitting). Cloud builds trigger AzureML runs on GPU machines that have a far higher CO2 footprint than your dev machine.
* When fixing a bug, the suggested workflow is to first write a unit test that shows the invalid behaviour, and only then start to code up the fix.

## What not to check in

* DO NOT check in files taken from or derived from private datasets.
* Avoid checking in large files (anything over 100kB). If you need to check in large files, consider adding them via Git LFS.

### Jupyter Notebooks

Notebooks can easily become obsolete over time and may not work with code changes. Also, testing them is generally difficult.

* If you are planning to use a notebook, please avoid checking them into the repository unless it is part of a project demo.
* If the notebook is used to document results, then please render it and place screenshots in a OneNote page.
* Notebooks are also an easy way of leaking sensitive data: They can for example contain images derived from private datasets.

## Review / Pull Requests

### Scope of Pull Requests

Pull Requests (PRs) should ideally implement a single change. If in doubt, err on the side of making the PR too small, rather than too big.

* Small PRs help reduce the load on the reviewer.
* Avoid adding unrelated changes to an existing PR.
* PRs should be modular: we can iterate on PRs, and any positive delta is a contribution.

Please follow the guidance on
[Github flow](https://docs.github.com/en/get-started/quickstart/github-flow)

Try gauging the value of your contribution for yourself by asking the following questions:

* Will this change bring the team closer to achieving their project goals?
* Will someone else understand my code?
* Will they be able to use my code?
* Will they be able to extend or build on top of my code?

### Pull Request Process

* When creating a Pull Request (PR), do add a summary of your contribution in the PR description.
* The template PR description also contains a checklist for the PR author.
* For collecting early feedback on your work, please use a Draft Pull Request. These PRs are marked with a grey icon in the Github UI, and send a clear signal that the code there is not yet ready for review. When submitting a draft PR, all the checks will be run.
* Once your work is ready for review, click the "Ready for Review" button on the Github PR page, and assign reviewers for your PR.

### Pull Request Contents

* Include a brief description.
* Link to an issue, if relevant.
* Write unit tests for the code - see above for details.
* Add appropriate documentation for any new code that you introduce.

### Pull Request Titles

To enable good auto-generated changelogs, we prefix all PR titles with a category string, like "BUG: Out of bounds error when using small images".
Those category prefixes must be in upper case, followed by a colon (`:`). Valid categories are

* `ENH` for enhancements, new capabilities
* `BUG` for bugfixes
* `STYLE` for stylistic changes (for example, refactoring) that do not impact the functionality
* `DOC` for changes to documentation only
* `DEL` for removing something from the codebase

## Notes on Branching

We should treat the main branch as a collection of code that is

* Of high-quality
* Readable and suitable for re-use
* Well documented

Use a dedicated dev branch for development, debugging and analysis work. Once you are sure that
the work in the dev branch adds enough value to reach our project objectives, then use a pull request
to get your contribution into the main branch.

## Tools for code cleanup

### Vulture

[Vulture](https://pypi.org/project/vulture) is a tool to find orphaned functions.
Use it occasionally to identify orphan functions and make sure that the repository remains simple.

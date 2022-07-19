# Design and Coding Guidelines

## Design Phase

For non-trivial changes, please communicate your plans early on to reach maximum impact and to avoid duplicate work.
Create an [issue](https://github.com/microsoft/hi-ml/issues) or a
[discussion](https://github.com/microsoft/hi-ml/discussions) that describes the problem that you are trying to solve,
and describe why this adds a positive contribution to the project.

Tag one of the core `hi-ml` team (@ant0nsc, @mebristo, @fepegar) to get input on your problem.
In particular, you should find out if this has already been worked on or thought about.
This will help you to get to know about existing code fragments that can simplify your task, and integrate better with existing code.

We recommend to work with a core `hi-ml` team member to create a design for your work.
Depending on the size of the task at hand, a design can be anywhere between a short description of your plan in a GitHub
discussion or issue, or it can be a shared document where you list your design options and invite feedback.
Your design partner will later also be well positioned to review your pull request (PR) because they will be already
familiar with your plan.

When working on a large contribution, we suggest to break it down into a set of small PRs that are more manageable for
you and for the reviewers (see below in the section "Scope of PRs")

## Setting up your development environment

Please see our [detailed instructions](developers.md).

## Coding Guidelines

### Naming

Please follow the general [PEP 8](https://peps.python.org/pep-0008/) Python rules for naming:

* Variables, function and method names should use `snake_case` (lower case with under scores separating words)
* Class names should follow `PascalCase` (first letter of each word capitalized).

To improve readability, functions or methods that return boolean values should follow a `is_...`, `has_...`, `use...` pattern, like `is_status_ok` instead of `ok`.

### Static Analysis and Linting

We use `flake8` as a linter, and `mypy` and pyright for static typechecking. Both tools run as part of the PR workflow, and must run without errors for a contribution to be accepted.
`mypy` requires that all functions and methods carry type annotations.
See the [`mypy` documentation](https://mypy.readthedocs.io/en/latest/getting_started.html#function-signatures-and-dynamic-vs-static-typing) for more information.

We highly recommend to run all those tools _before_ pushing the latest changes to a PR.
If you have `make` installed, you can run both tools in one go via `make check` (from the repository root folder).

### Documentation

For general information around docstrings, please check [PEP 257](https://peps.python.org/pep-0257/).

We use Sphinx for generating documentation.
We recommend using a VSCode extension for auto-generating documentation templates, for example [`njpwerner.autodocstring`](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring).
This extension is already pre-configured in our VSCode workspace settings file.

Reminders about docstrings:

* There must be a separating line between a function description and the documentation for its parameters.
* In multi-line parameter descriptions, continuations on the next line must be indented.
* Sphinx will merge the class description and the arguments of `__init__` method. Hence, there is no need to
  write any text in the docstring for `__init__` other than the arguments.
* Use `>>>` to include code snippets.

To generate the Sphinx documentation on your machine, run

```shell
cd docs
make html
```

Then open the results in `build/html/index.html`.

Example:

```python
class Foo:
    """This is the class description.

    The following block will be pretty-printed by Sphinx. Note the space between >>> and the code!

    Usage example:
        >>> from module import Foo
        >>> foo = Foo(bar=1.23)
    """

    ANY_ATTRIBUTE = "what_ever."
    """Document class attributes after the attribute. Should end with a period."""

    def __init__(self, bar: float = 0.5) -> None:
        """
        :param bar: This is a description for the constructor argument.
            Long descriptions should be indented.
        """
        self.bar = bar

    def method(self, arg: int) -> None:
        """Short method description, followed by an empty line. Sentences should end with a period.

        Longer descriptions can be added as well.
        Argument names like ``arg`` are rendered nicely if enclosed in double backticks.

        :param arg: This is a description for the method argument.
            Long descriptions should be indented.
        :raises ValueError: If something bad happens.
        """
        do_something()  # comments should start with "  # "
```

For more information, check the [Sphinx-RTD tutorial](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html).

## Testing

### Why do we write unit tests?

If you think that there are sensitive parts in your code, where hidden errors can impact the project outcome, please take your time to protect parts of your code.
As the team grows, so does the number of users of the repository, and we may not be able to monitor changes on a frequent basis, but your tests can ensure that the expected behavior is preserved.
We are not making it strictly mandatory, however, contributors should take into account that their code may have negative downstream impact.

We do not need to test every part of the code depending on the project constraints, e.g., quality and test-coverage vs learning and hypothesis-testing.
However, we expect a justification in each PR for this decision and it should be reviewed by team members.
Please regularly monitor the code coverage results that are posted as comments on each pull request.

In particular, please do write unit tests for anything that affects training /results, such as data preprocessing, losses, and metrics.

### Testing Do's / Don'ts

* DO write unit tests for each new function or class that you add.
* DO extend unit tests for existing functions or classes if you change their core behaviour.
* DO try your best to write unit tests that are fast.
  Very often, this can be done by reducing data size to a minimum.
  Also, it is helpful to avoid long-running integration tests, but try to test at the level of the smallest involved
  function.
* DO ensure that your tests are designed in a way that they can pass on the local machine, even if they are relying on
  specific cloud features.
  If required, use `unittest.mock` to simulate the cloud features, and hence enable the tests
  to run successfully on your local machine.
* DO run all unit tests on your dev machine before submitting your changes.
  The test suite is designed to pass completely also on your development machine.
* DO NOT rely only on the test results in the cloud (i.e., run test locally before submitting).
  Apart from the obvious delay in getting your results, CI runs trigger AzureML runs on GPU machines that have a far higher CO2 footprint than your dev machine.
* When fixing a bug, the suggested workflow is to first write a unit test that shows the invalid behaviour, and only
  then start to code up the fix.

### Testing of Scripts

Depending on the project needs, we may write scripts for doing one-off operations (as an example, have a look at
[himl_download.py](https://github.com/microsoft/hi-ml/blob/main/hi-ml-azure/src/health_azure/himl_download.py).
How carefully should we test those?

* Any scripts that perform operations that we anticipate others to need as well should be designed for re-usability and
  be tested.
  "Designed for re-use" here would mean, for example: The script should not contain any hard-coded setup that
  is specific to my machine or user account.
  If that's not achievable, document carefully what others need to prepare so that they can run this script.
* Scripts are inherently difficult to test.
  Testing becomes a lot easier if the script is mainly a front-end to functions that live somewhere else in the codebase.If the script is written as a thin wrapper around library functions, these library functions can be tested in isolation as part of the normal unit test suite.

Writing arguments parsers is particularly error prone.
Consider using automatically generated parsers, like we use in the [`hi-ml`](https://github.com/microsoft/hi-ml/blob/b742223102d6c9092b13b20eafa263cc91f99670/hi-ml-azure/src/health_azure/utils.py#L157-L167):
Starting point is a class that describes inputs to a function.
A parser can be generated automatically from these classes.

Lastly, if you are considering adding a script to your project, also consider the following: If the script should be
called, for example, after an AzureML run to collect results, can this be automated further, and make the script
obsolete? Reasons for this approach:

* People tend to forget that there is a script to do X already, and may re-do the task in question manually.
* Any script that requires input from the user also has a chance to be provided with the wrong input, leading to friction or incorrect results.
  In a programmatic scenario, where the script is called automatically, this chance of errors is greatly minimized.

## What not to commit

* DO NOT check in files taken from or derived from private datasets.
* DO NOT check in any form of credentials, passwords or access tokens.
* Do not check in any code that contains absolute paths (for example, paths that only work on your machine).
* Avoid checking in large files (anything over 100 kB).
  If you need to commit large files, consider adding them via [Git LFS](https://git-lfs.github.com/).

### Jupyter Notebooks

Notebooks can easily become obsolete over time and may not work with code changes.
Also, testing them is generally difficult.
Please follow these guidelines for notebooks:

* If you are planning to use a notebook, avoid committing into the repository unless it is part of a project demo.
* If the notebook is used to document results, then please render the notebook and place the results in a separate
  document outside the repository.
* Bear in mind that notebooks are also an easy way of leaking sensitive data: They can for example contain images
  derived from private datasets.
  You should clear the notebook outputs before committing it.

## Review / Pull Requests

### Scope of Pull Requests

PRs should ideally implement a single change.
If in doubt, err on the side of making the PR too small, rather than too big.

* Small PRs help reduce the load on the reviewer.
* Avoid adding unrelated changes to an existing PR.
* PRs should be modular: we can iterate on PRs, and any positive delta is a contribution.

Please follow the guidance on
[Github flow](https://docs.github.com/en/get-started/quickstart/github-flow).

Try gauging the value of your contribution for yourself by asking the following questions:

* Will this change bring the team closer to achieving their project goals?
* Will someone else understand my code?
* Will they be able to use my code?
* Will they be able to extend or build on top of my code?

### Pull Request Process

* When creating a PR, do add a summary of your contribution in the PR description.
* The template PR description also contains a checklist for the PR author.
* Link your PR to a GitHub issue that describes the problem/feature that you are working on.
* For collecting early feedback on your work, please use a Draft Pull Request.
  These PRs are marked with a grey icon in the Github UI, and send a clear signal that the code there is not yet ready for review.
  When submitting a draft PR, all the checks will be run as for a normal PR.
* Once your work is ready for review, click the "Ready for Review" button on the Github PR page, and, if you want, assign reviewers for your PR.

### Pull Request Titles

To enable good auto-generated changelogs, we prefix all PR titles with a category string, like `BUG: Fix out of bounds error when using small images`.
Those category prefixes must be in upper case, followed by a colon (`:`).
Valid categories are

* `ENH` for enhancements, new capabilities
* `BUG` for bugfixes
* `STY` for stylistic changes (for example, refactoring) that do not impact the functionality
* `DOC` for changes to documentation only
* `DEL` for removing something from the codebase
* `TEST` for adding or modifying tests
* `FIX` to fix something that is not a `BUG` such as a typo
* `MNT` maintenance, to upgrade package version, packaging, linting, CI etc.
* `PERF` performance related

## Notes on Branching

We should treat the main branch as a collection of code that is

* Of high-quality
* Readable and suitable for re-use
* Well documented

Use a dedicated dev branch for development, debugging and analysis work.
Once you are sure that the work in the dev branch adds enough value to reach our project objectives, then use a pull request to get your contribution into the main branch.

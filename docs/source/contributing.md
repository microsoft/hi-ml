# Contributing to this toolbox

We welcome all contributions that help us achieve our aim of speeding up ML/AI research in health and life sciences.
Examples of contributions are

* Data loaders for specific health & life sciences data
* Network architectures and components for deep learning models
* Tools to analyze and/or visualize data

All contributions to the toolbox need to come with unit tests, and will be reviewed when a Pull Request (PR) is started.
If in doubt, reach out to the core `hi-ml` team before starting your work.

Please look through the existing folder structure to find a good home for your contribution.

## Submitting a Pull Request

If you'd like to submit a PR to the codebase, please ensure you:

- Include a brief description
- Link to an issue, if relevant
- Write unit tests for the code - see below for details.
- Add appropriate documentation for any new code that you introduce
- Ensure that you modified [CHANGELOG.md](../CHANGELOG.md) and described your PR there.
- Only publish your PR for review once you have a build that is passing. You can make use of the "Create as Draft"
  feature of GitHub.

## Code style

- We use `flake8` as a linter, and `mypy` and pyright for static typechecking. Both tools run as part of the PR build,
  and must run without errors for a contribution to be accepted. `mypy` requires that all functions and methods carry
  type annotations,
  see [mypy documentation](https://mypy.readthedocs.io/en/latest/getting_started.html#function-signatures-and-dynamic-vs-static-typing)
- We highly recommend to run all those tools _before_ pushing the latest changes to a PR. If you have `make` installed,
  you can run both tools in one go via `make check` (from the repository root folder)

## Unit testing

- DO write unit tests for each new function or class that you add.
- DO extend unit tests for existing functions or classes if you change their core behaviour.
- DO try your best to write unit tests that are fast. Very often, this can be done by reducing data size to a minimum.
  Also, it is helpful to avoid long-running integration tests, but try to test at the level of the smallest involved
  function.
- DO ensure that your tests are designed in a way that they can pass on the local machine, even if they are relying on
  specific cloud features. If required, use `unittest.mock` to simulate the cloud features, and hence enable the tests
  to run successfully on your local machine.
- DO run all unit tests on your dev machine before submitting your changes. The test suite is designed to pass
  completely also outside of cloud builds.
- DO NOT rely only on the test builds in the cloud (i.e., run test locally before submitting). Cloud builds trigger
  AzureML runs on GPU machines that have a far higher CO2 footprint than your dev machine.
- When fixing a bug, the suggested workflow is to first write a unit test that shows the invalid behaviour, and only
  then start to code up the fix.

## Correct Sphinx Documentation

Common mistakes when writing docstrings:

* There must be a separating line between a function description and the documentation for its parameters.
* In multi-line parameter descriptions, continuations on the next line must be indented.
* Sphinx will merge the class description and the arguments of the constructor `__init__`. Hence, there is no need to
  write any text in the constructor, only the classes' parameters.
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
        Method description, followed by an empty line.
        
        :param arg: This is a description for the method argument.
            Long descriptions should be indented.
        """
```

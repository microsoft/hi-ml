# Contributing to this toolbox

We welcome all contributions that help us achieve our aim of speeding up ML/AI research in health and life sciences.
Examples of contributions are
* Data loaders for specific health & life sciences data
* Network architectures and components for deep learning models
* Tools to analyze and/or visualize data

All contributions to the toolbox need to come with unit tests, and will be reviewed when a Pull Request (PR) is started.
If in doubt, reach out to the core `hi-ml` team before starting your work.


## Code style

- We use `flake8` as a linter, and `mypy` for static typechecking. Both tools run as part of the PR build, and must run
without errors for a contribution to be accepted. `mypy` requires that all functions and methods carry type annotations,
see 
- We highly recommend to run both tools _before_ pushing the latest changes to a PR. If you have `make` installed, you
can run both tools in one go via `make check` (from the repository root folder)
- Code should use sphinx-style comments like this:
```python
from typing import List, Optional
def foo(bar: int) -> Optional[List]:
    """
    Creates a list. Or not.
    :param bar: The length of the list. If 0, returns None.
    :return: A list with `bar` elements.
    """
```

## Unit testing
- DO write unit tests for each new function or class that you add.
- DO extend unit tests for existing functions or classes if you change their core behaviour.
- DO try your best to write unit tests that are fast, for example by reducing data size to a minimum.
- DO ensure that your tests are designed in a way that they can pass on the local box, even if they are relying on
specific cloud features. If required, use `unittest.mock` to enable the tests to run without cloud. 
- DO run all unit tests on your dev box before submitting your changes. The test suite is designed to pass completely
also outside of cloud builds.
- DO NOT rely only on the test builds in the cloud. Cloud builds trigger AzureML runs on GPU 
machines that have a far higher CO2 footprint than your dev box.


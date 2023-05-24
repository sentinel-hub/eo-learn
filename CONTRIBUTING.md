# Contributing to **eo-learn**

First of all, thank you for contributing to **eo-learn**. Any effort in contributing to the library is very much appreciated.

Here is how you can contribute:

* [Bug Reports](#bug-reports)
* [Feature Requests](#feature-requests)
* [Pull Requests](#pull-requests)

All contributors agree to follow our [Code of Conduct][code-of-conduct].

**eo-learn** is distributed under the [MIT license][license]. When contributing code to the library, you agree to its terms and conditions. If you would like to keep parts of your contribution private, you can contact us to discuss about the best solution.

For any question, feel free to contact us at [eoresearch@sinergise.com](eoresearch@sinergise.com) or through our [Forum][sh-forum].

[code-of-conduct]: https://github.com/sentinel-hub/eo-learn/blob/master/CODE_OF_CONDUCT.md
[license]: https://github.com/sentinel-hub/eo-learn/blob/master/LICENSE
[sh-forum]: https://forum.sentinel-hub.com/

## Bug Reports

We strive to provide high-quality working code, but bugs happen nevertheless.

When reporting a bug, please check [here][open-bug-list] whether the bug was already reported. If not, open an issue with the **bug** label and report the following information:

* Issue description
* How to reproduce the issue
* Expected outcome
* Actual outcome
* OS and package versions

This information helps us to reproduce, pinpoint, and fix the reported issue. If you are not sure whether the odd behaviour is a bug or a _feature_, best to open an issue and clarify.

[open-bug-list]: https://github.com/sentinel-hub/eo-learn/issues?q=state:open+type:issue+label:"bug"

## Feature Requests

Existing feature requests can be found [here][existing-feature-requests].

A new feature request can be created by opening a new issue with the **enhancement** label, and describing how the feature would benefit the **eo-learn** community. Providing an example use-case would help assessing the scope of the feature request.

[existing-feature-requests]: https://github.com/sentinel-hub/eo-learn/issues?q=state:open+type:issue+label:"enhancement"

## Pull Requests

The GitHub Pull Request (PR) mechanism is the best option to contribute code to the library. Users can fork the repository, make their contribution to their local fork and create a PR to add those changes to the codebase. GitHub provides excellent tutorials on how the [fork and pull][fork-and-pull] mechanism work, and on how to best [create a PR][create-pr].

Existing PRs can be found [here][existing-prs]. Before creating new PRs, you should check whether someone else has contributed a similar feature, and if so, you can add your input to the existing code review.

The following guidelines should be observed when creating a PR.

[fork-and-pull]: https://help.github.com/articles/creating-a-pull-request-from-a-fork
[create-pr]: https://help.github.com/articles/creating-a-pull-request/
[existing-prs]: https://github.com/sentinel-hub/eo-learn/pulls?q=state:open

### General guidelines

* Where applicable, create your contribution in a new branch of your fork based on the `develop` branch, as the `master` branch is aligned to the released package on [PyPI][pypi]. Upon completion of the code review, the branch will be merged into `develop` and, at the next package release, into `master`.

* Document your PR to help maintainers understand and review your contribution. The PR should include:

  * Description of contribution;
  * Required testing;
  * Link to issue/feature request.

* Do not forget to add yourself as a contributor to the `CREDITS.md` file, found in the root of the repository.

* When submitting the PR, a member of the development team has to approve the CI run (notify us if we forget). If it returns any errors, try to fix the issues causing failure. For information on how to run the tests locally, see Section on *Running formatters, linters, and tests*.

* Your contribution should include unit tests, to test correct behaviour of the new feature and to lower the maintenance effort. This applies to both bug fixes and new features.

* Try to keep contributions small, as this speeds up the reviewing process. In the case of large contributions, e.g. a new complex `EOTask`, it's best to contact us first to review the scope of the contribution.

* Keep API compatibility in mind, in particular when contributing a new `EOTask`. Check the Section below for more information on how to contribute an `EOTask`.

* New features or tasks should be appropriately commented using Sphinx style docstrings. The documentation uses the [PEP-8][pep-8] formatting guidelines.

### Development environment

* Get the latest development version by creating a fork and clone the repo:

```bash
git clone git@github.com:<username>/eo-learn.git
```

* Make sure that you have a suitable version of Python installed. Check the [GitHub page](https://github.com/sentinel-hub/eo-learn) badges on which versions are currently supported.

* All **eo-learn** packages can be installed at once using the script `python install_all.py -e`. To install each package separately, run `pip install -e <package_folder>`. We strongly recommend initializing a virtual environment before installing the required packages. Example on how to create an environment an install all subpackages and development requirements:

```bash
cd eo-learn
# The following creates the virtual environment in the ".env" folder.
virtualenv --python python3 .env
source .env/bin/activate

# The following installs all eo-learn subpackages and development packages
# using PyPI in the activated virtualenv environment.
python install_all.py -e
pip install -r requirements-dev.txt -r requirements-docs.txt
```

**Note:** to reduce later merge conflicts, always pull the latest version of the `develop` branch from the upstream eo-learn repository ([located here][dev-branch]) to your fork before starting the work on your PR.

### Contribute an `EOTask`

`EOTask`s are currently grouped into subpackages by scope, e.g. *core*, *IO*, *masks*, and so on. A list of implemented tasks can be found in the [documentation][existing-eo-tasks]. The following code snippet shows how to create your own `EOTask`:

```python
class FooTask(EOTask):
    def __init__(self, foo_params):
        self.foo_params = foo_params

    def execute(self, eopatch, *, execution_kwargs):
        # do what foo does on input eopatch and return it
        return eopatch
```

When creating a new task, bear in mind the following:

* Tasks should be useful for multiple use-cases or solve a common problem, i.e., other users should benefit from it.
* In cases of large monolithic tasks you might want to try and split them into multiple ones, where each does just part of the computation. For instance a task for detecting foliage from S2 imagery could perhaps be split into a task for calculating NDVI and a separate task for thresholding the NDVI values. In many cases you'll find that some of these smaller tasks already exist in `eo-learn`.
* When multiple tasks share a large part of the implementation, consider extracting parts of it into functions or use inheritance to reduce boilerplate.
* Only hard-code parameters that shouldn't be adjusted by the users. Hard-coding things like feature names should be avoided.
* Use feature-parsers in order to support a wider range of possible input (and perhaps validate inputs). You can find them in `eolearn.core.utils.parsing`, or use `EOTask.parse_features` as a shortcut.
* If in doubt on whether a task is general enough to be of interest to the community, or you are not sure to which sub-package to contribute your task to, feel free to open up a [feature request](#feature-requests).

### Running formatters, linters, and tests

This section assumes you have installed all packages in `requirements-dev.txt`.

Most of the automated code-checking is packaged into [pre-commit hooks](https://pre-commit.com/). You can activate them by running `pre-commit install`. If you wish to check all code you can do so by running `pre-commit run --all-files`. This takes care of:
- auto-formatting the code using `black`, `isort`, and `autoflake`
- checking the code with `ruff`
- checking and formatting any Jupyter notebooks with `nbqa`
- various other helpful things (correcting line-endings etc.)

The code is also checked using `pylint` and `mypy`. Because of the project structure invoking these two checkers is a bit trickier and has to be performed on each module separately. If your contribution is in core, you could do the following:
```bash
# to check the 'core' subpackage
mypy core/eolearn/core
pylint core/eolearn/core
```
You can also use the utilities provided in the MAKEFILE to check the entire package with `make pylint` and `make mypy`. Due to the size of the codebase this might take a while.

The last bit to check are the unit tests. We again reiterate that you should include unit-tests of your contributions. If you are not well versed with unit-testing you can ask us for help in the pull-request or by issuing a [ticket](https://github.com/sentinel-hub/eo-learn/issues) instead. To run the tests simply use the command `pytest` from the main folder. Since `eolearn.io` also test integration with SentinelHub you can either skip them with `pylint -m "not sh_integration"` or see the [examples/README.md](examples/README.md) for how to setup you SentinelHub account and local config for testing.


Looking forward to include your contributions into **eo-learn**.

[pypi]: https://pypi.org/project/eo-learn/
[pep-8]: https://www.python.org/dev/peps/pep-0008/
[pylint]: https://www.pylint.org/
[existing-eo-tasks]: https://eo-learn.readthedocs.io/en/latest/eotasks.html
[test-eo-patch]: https://github.com/sentinel-hub/eo-learn/tree/master/example_data/TestEOPatch
[python]: https://www.python.org/downloads/
[conda]: https://www.anaconda.com/distribution/
[dev-branch]: https://github.com/sentinel-hub/eo-learn/tree/develop/

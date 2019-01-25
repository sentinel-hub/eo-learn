# Contributing to **eo-learn**

First of all, thank you for contributing to **eo-learn**. Any effort in contributing
to the library is very much appreciated.

Here is how you can contribute:

 * [Bug Reports](#bug-reports)
 * [Feature Requests](#feature-requests)
 * [Pull Requests](#pull-requests)

All contributors agree to follow our [Code of Conduct][code-of-conduct].

**eo-learn** is distributed under the [MIT license][license]. When contributing
code to the library, you agree to its terms and conditions. If you would like to
keep parts of your contribution private, you can contact us to discuss about
the best solution.

For any question, feel free to contact us at [eoresearch@sinergise.com](eoresearch@sinergise.com) or through our [Forum][sh-forum].

[code-of-conduct]: https://github.com/sentinel-hub/eo-learn/blob/master/CODE_OF_CONDUCT.md
[license]: https://github.com/sentinel-hub/eo-learn/blob/master/LICENSE
[sh-forum]: https://forum.sentinel-hub.com/

## Bug Reports

We strive to provide high-quality working code, but bugs happen nevertheless.

When reporting a bug, please check [here][open-bug-list] whether
the bug was already reported. If not, open an issue with the **bug** label and
report the following information:

 * Issue description
 * How to reproduce the issue
 * Expected outcome
 * Actual outcome
 * OS and package versions

This information helps us to reproduce, pinpoint, and fix the reported issue.

If you are not sure whether the odd behaviour is a bug or a _feature_, best to open
an issue and clarify.

[open-bug-list]: https://github.com/sentinel-hub/eo-learn/issues?q=state:open+type:issue+label:"bug"

## Feature Requests

Existing feature requests can be found [here][existing-feature-requests].

A new feature request can be created by opening a new issue with the **enhancement** label,
and describing how the feature would benefit the **eo-learn** community.
Providing an example use-case would help assessing the scope of the
feature request.

[existing-feature-requests]: https://github.com/sentinel-hub/eo-learn/issues?q=state:open+type:issue+label:"enhancement"

## Pull Requests

The GitHub Pull Request (PR) mechanism is the best option to contribute code
to the library. Users can fork the repository, make their contribution to their
local fork and create a PR to add those changes to the codebase. GitHub provides excellent
tutorials on how the [fork and pull][fork-and-pull] mechanism work, and on
how to best [create a PR][create-pr].

Existing PRs can be found [here][existing-prs]. Before creating new PRs, you should check
whether someone else has contributed a similar feature, and if so, you can add your
input to the existing code review.

The following guidelines should be observed when creating a PR.

[fork-and-pull]: https://help.github.com/articles/creating-a-pull-request-from-a-fork
[create-pr]: https://help.github.com/articles/creating-a-pull-request/
[existing-prs]: https://github.com/sentinel-hub/eo-learn/pulls?q=state:open

### General guidelines

 * Where applicable, create your contribution in a new branch of your fork based on the
   `develop` branch, as the `master` branch is aligned to the released package on [PyPI][pypi]. Upon
   completion of the code review, the branch will be merged into `develop` and, at
   the next package release, into `master`.

 * Document your PR to help maintainers understand and review your contribution. The PR
   should include:

   * Description of contribution;
   * Required testing;
   * Link to issue/feature request.

 * Your contribution should include unit tests, to test correct behaviour of the new feature
   and to lower the maintenance effort. Bug fixes as well as new features should include unit tests.
   When submitting the PR, check whether the Travis CI testing returns any errors, and if it does,
   please try to fix the issues causing failure. A test `EOPatch` is made available [here][test-eo-patch]
   with data for each `FeatureType`. Unit tests evaluating the correctness of new tasks should use data
   available in this `EOPatch`. New fields useful for testing purposes can be added, but should
   be consistent with the `bbox` and `timestamp` of the `EOPatch`.

 * Try to keep contributions small, as this speeds up the reviewing process. In the case of large
   contributions, e.g. a new complex `EOTask`, it's best to contact us first to review the scope
   of the contribution.

 * Keep API compatibility in mind, in particular when contributing a new `EOTask`. In general,
   all new tasks should adhere to the modularity of **eo-learn**.
   Check the Section below for more information on how to contribute an `EOTask`.

 * New features or tasks should be appropriately commented using Sphinx style docstrings. The documentation uses
   the [PEP-8][pep-8] formatting guidelines. [Pylint][pylint] is used to check the coding standard.
   Therefore please run `pylint` from the the main folder, which contains the `pylintrc` file, to make sure your
   contribution is scored `10.0/10.0`.

### Contribute an `EOTask`

`EOTask`s allow to apply **eo-learn** workflows to different use-cases, adapting to imaging sources and
processing chain. If you think a task is general enough to be useful to the community, then we would
be delighted to include it into the library.

`EOTask`s are currently grouped by scope, e.g. core, IO, masks, and so on. A list of implemented
tasks can be found in the [documentation][existing-eo-tasks]. The following code snippet shows how
to create your own `EOTask`

```python
class FooTask(EOTask):
    def __init__(self, foo_params):
        self.foo_params = foo_params

    def execute(self, eopatch, *, runtime_params):
        # do what foo does on input eopatch and return it
        return eopatch
```
When creating a new task, bear in mind the following:

 * Tasks should be as modular as possible, facilitating task re-use and sharing.
 * An `EOTask` should perform a well-defined operation on the input eopatch(es). If the operation
   could be split into atomic sub-operations that could be used separately, then consider splitting
   the task into multiple tasks. Similarly, if tasks share the bulk of the implementation but differ
   in a minority of implementation, consider using Base classes and inheritance. The interpolation
   tasks represent a good example of this.  
 * Tasks should be as generalizable as possible, therefore hard-coding of task parameters or `EOPatch`
   feature types should be avoided. Use the `EOTask._parse_features` method to parse input features in a task,
   and pass task parameters as arguments, either in the constructor, or at run-time.
 * If in doubt on whether a task is general enough to be of interest to the community, or you are not
   sure to which sub-package to contribute your task to, send us an email or open a
   [feature request](#feature-requests).

Looking forward to include your contributions into **eo-learn**.


[pypi]: https://pypi.org/project/eo-learn/
[pep-8]: https://www.python.org/dev/peps/pep-0008/
[pylint]: https://www.pylint.org/
[existing-eo-tasks]: https://eo-learn.readthedocs.io/en/latest/eotasks.html
[test-eo-patch]: https://github.com/sentinel-hub/eo-learn/tree/master/example_data/TestEOPatch

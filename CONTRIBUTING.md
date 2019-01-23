# Contributing to **eo-learn**

First of all, thank you for contributing to **eo-learn**. Any effort in contributing 
to the library is very much appreciated.

Here is how you can contribute:
 
 * [Bug Reports][bug-reports]
 * [Feature Requests][feature-requests]
 * [Pull Requests][pull-requests]
 
All contributors agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

**eo-learn** is distributed under the [MIT license](LICENSE). When contributing
code to the library, you agree to its terms and conditions. If you would like to 
keep parts of your contribution private, you can contact us to discuss about
the best solution.

For any question, feel free to contact us at _eoresearch@sinergise.com_ or through our [Forum][sh-forum].

[sh-forum]: https://forum.sentinel-hub.com/

## Bug Reports
[bug-reports]: #bug-reports

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
[feature-requests]: #feature-requests

Existing feature requests can be found [here][existing-feature-requests]. 

A new feature request can be created by opening a new issue with the **enhancement** label, 
and describing how the feature would benefit the **eo-learn** community. 
Providing an example use-case would help assessing the scope of the 
feature request.

[existing-feature-requests]: https://github.com/sentinel-hub/eo-learn/issues?q=state:open+type:issue+label:"enhancement"

## Pull Requests
[pull-requests]: #pull-requests

The GitHub Pull Request (PR) mechanism is the best option to contribute code 
to the library. Users can fork the repository, make their contribution to their 
local fork and create a PR to add those changes to the codebase. GitHub provides excellent 
tutorials on how the [fork and pull][fork-and-pull] mechanism work, and on
how to best [create a PR][create-pr].

Existing PRs can be found [here][existing-prs]. Before creating new PRs, you should check
whether someone else has contributed a similar feature, and if so, you can add your 
input to the existing code review.

The follwoing guidelines should be observed when creating a PR.

[fork-and-pull]: https://help.github.com/articles/creating-a-pull-request-from-a-fork
[create-pr]: https://help.github.com/articles/about-pull-requests/
[existing-prs]: https://github.com/sentinel-hub/eo-learn/pulls?q=state:open

### General guidelines

 * Where applicable, create your contribution in a new branch of your fork based on the 
   `develop` branch, as the `master` branch is aligned to the released package on [PyPI][pypi]. Upon 
   completion of the code review, the branch will be merged into `develop` and, at
   the next package release, into `master`.
   
 * Document your PR to help mantainers understand and review your contribution. The PR
   should include:
   
   * Description of contribution
   * Required testing 
   * Link to issue/feature request
   
 * Your contribution should include unit tests, to test correct behaviour of the new feature
   and to lower the mantainance effort. Bug fixes as well new features should include unit-tests.
   When submitting the PR, check whether the travis CI testing returns any errors, and if it does,
   please try to fix the issues causing failure. 
   
 * Try to keep contributions small, as this speeds up the reviewing process. In the case of large 
   contributions, e.g. a new complex `EOTask`, it's best to contact us first to review the scope 
   of the contribution.
   
 * Keep API compatibility in mind, in particular when contributing a new `EOTask`. For instance,
   use the `EOTask._parse_features` method to parse input features in a task, so that the task can be as 
   generalisable as possible. Parameters of the tasks should not be hard-coded, but passed as arguments, either in the 
   constructor, or at run-time. In general, all new tasks should adhere to the modularity of **eo-learn**. 
   
 * New features or tasks should be appropriately commented using docstrings. The documentation uses
   the [PEP-8][pep-8] formatting guidelines. [Pylint][pylint] is used to check the coding standard,
   so please run `pylint` from the folder containing the `pylintrc` file to make sure your 
   contribution is scored `10.0/10.0`.


[pypi]: https://pypi.org/
[pep-8]: https://www.python.org/dev/peps/pep-0008/
[pylint]: https://www.pylint.org/

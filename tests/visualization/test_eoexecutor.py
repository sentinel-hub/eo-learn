"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
import tempfile

import pytest

from eolearn.core import EOExecutor, EONode, EOTask, EOWorkflow

logging.basicConfig(level=logging.DEBUG)


class ExampleTask(EOTask):
    def execute(self, *_, **kwargs):
        my_logger = logging.getLogger(__file__)
        my_logger.info("Info statement of Example task with kwargs: %s", kwargs)
        my_logger.warning("Warning statement of Example task with kwargs: %s", kwargs)
        my_logger.debug("Debug statement of Example task with kwargs: %s", kwargs)

        if "arg1" in kwargs and kwargs["arg1"] is None:
            raise RuntimeError(f"Oh no, i spilled my kwargs all over the floor! {kwargs}!")


NODE = EONode(ExampleTask())
WORKFLOW = EOWorkflow([NODE, EONode(task=ExampleTask(), inputs=[NODE, NODE])])
EXECUTION_KWARGS = [
    {NODE: {"arg1": 1}},
    {},
    {NODE: {"arg1": 3, "arg3": 10}},
    {NODE: {"arg1": None}},
    {NODE: {"arg1": None, "arg3": 10}},
]


@pytest.mark.parametrize("save_logs", [True, False])
@pytest.mark.parametrize("include_logs", [True, False])
def test_report_creation(save_logs, include_logs):
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        executor = EOExecutor(
            WORKFLOW,
            EXECUTION_KWARGS,
            logs_folder=tmp_dir_name,
            save_logs=save_logs,
            execution_names=["ex 1", 2, 0.4, None, "beep"],
        )
        executor.run(workers=10)
        executor.make_report(include_logs=include_logs)

        assert os.path.exists(executor.get_report_path()), "Execution report was not created"

"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Nejc Vesel, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import os
import logging
import tempfile

from eolearn.core import EOTask, EOWorkflow, Dependency, EOExecutor


logging.basicConfig(level=logging.DEBUG)


class ExampleTask(EOTask):

    def execute(self, *_, **kwargs):
        my_logger = logging.getLogger(__file__)
        my_logger.info('Info statement of Example task with kwargs: %s', kwargs)
        my_logger.warning('Warning statement of Example task with kwargs: %s', kwargs)
        my_logger.debug('Debug statement of Example task with kwargs: %s', kwargs)

        if 'arg1' in kwargs and kwargs['arg1'] is None:
            raise Exception


TASK = ExampleTask()
WORKFLOW = EOWorkflow([(TASK, []), Dependency(task=ExampleTask(), inputs=[TASK, TASK])])
EXECUTION_ARGS = [
    {TASK: {'arg1': 1}},
    {},
    {TASK: {'arg1': 3, 'arg3': 10}},
    {TASK: {'arg1': None}}
]


def test_report_creation():
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        executor = EOExecutor(
            WORKFLOW, EXECUTION_ARGS, logs_folder=tmp_dir_name, save_logs=True, execution_names=['ex 1', 2, 0.4, None]
        )
        executor.run(workers=10)
        executor.make_report()

        assert os.path.exists(executor.get_report_filename()), 'Execution report was not created'

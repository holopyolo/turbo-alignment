from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.constants import FIXTURES_PATH
from turbo_alignment.cli import app

runner = CliRunner()


@pytest.mark.parametrize(
    'config_path,output_dir',
    [
        (FIXTURES_PATH / 'configs/train/classification/base.json', 'test_train_classification_output'),
        (
            FIXTURES_PATH / 'configs/train/classification/multilabel.json',
            'test_train_classification_multilabel_output',
        ),
    ],
)
def test_classification_train(config_path: Path, output_dir: str):
    result = runner.invoke(
        app,
        [
            'train_classification',
            '--experiment_settings_path',
            config_path,
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert Path(output_dir).is_dir()


if __name__ == '__main__':
    test_classification_train(
        FIXTURES_PATH / 'configs/train/classification/base.json',
        'test_train_classification_output',
    )

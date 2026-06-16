from types import SimpleNamespace

from tests.utils import is_sample_build_from_content
from turbo_alignment.dataset.classification.classification import (
    ClassificationDataset,
    _classification_label_distribution,
)
from turbo_alignment.dataset.classification.models import ClassificationDatasetRecord
from turbo_alignment.dataset.pair_preferences.models import PairPreferenceRecord
from turbo_alignment.dataset.registry import DatasetRegistry
from turbo_alignment.settings.datasets.base import DatasetStrategy, DatasetType
from turbo_alignment.settings.datasets.classification import (
    ClassificationDatasetSettings,
)
from turbo_alignment.settings.datasets.pair_preference import (
    PairPreferenceDatasetSettings,
)


def test_classification_label_distribution_collapses_list_labels():
    assert _classification_label_distribution(
        [
            {'labels': 0},
            {'labels': 1},
            {'labels': [0, 0, 0]},
            {'labels': [0, 1, 0]},
            {'labels': [1, 0, 1]},
            {'labels': None},
        ]
    ) == {0: 2, 1: 3}


def test_classification(tokenizer_llama2, chat_dataset_settings, classification_dataset_source):
    # load dataset and check that samples have required fields

    source, data_dicts = classification_dataset_source

    dataset_cls = DatasetRegistry.by_name(DatasetType.CLASSIFICATION).by_name(DatasetStrategy.TRAIN)

    dataset_settings = ClassificationDatasetSettings(chat_settings=chat_dataset_settings)

    dataset = dataset_cls(tokenizer=tokenizer_llama2, source=source, settings=dataset_settings, seed=42)

    assert len(data_dicts) == len(dataset)

    for data_dict, sample in zip(data_dicts, dataset):
        record = ClassificationDatasetRecord.model_validate(data_dict)

        assert record.label == sample['labels']

        assert is_sample_build_from_content(
            sample['input_ids'], [m.content for m in record.messages], tokenizer_llama2
        )


def test_classification_logs_loaded_samples_and_label_distribution(
    chat_dataset_settings, classification_dataset_source, monkeypatch
):
    messages: list[str] = []
    monkeypatch.setattr(
        'turbo_alignment.dataset.classification.classification.logger',
        SimpleNamespace(info=messages.append),
    )
    monkeypatch.setattr(
        ClassificationDataset,
        '_encode',
        lambda self, records, inference: [{'labels': record.label} for record in records],
    )

    source, data_dicts = classification_dataset_source

    dataset_cls = DatasetRegistry.by_name(DatasetType.CLASSIFICATION).by_name(DatasetStrategy.TRAIN)
    dataset_settings = ClassificationDatasetSettings(chat_settings=chat_dataset_settings)

    dataset_cls(tokenizer=SimpleNamespace(), source=source, settings=dataset_settings, seed=42)

    assert f'Classification dataset {source.name} loaded: {len(data_dicts)} samples' in messages
    assert (
        f'Classification dataset {source.name} label distribution: {{0: 5, 1: 4}} '
        '(list labels are counted as int(sum(label) > 0))'
    ) in messages


def test_multilabel_classification(tokenizer_llama2, chat_dataset_settings, classification_multilabel_dataset_source):
    source, data_dicts = classification_multilabel_dataset_source

    dataset_cls = DatasetRegistry.by_name(DatasetType.CLASSIFICATION).by_name(DatasetStrategy.TRAIN)

    dataset_settings = ClassificationDatasetSettings(chat_settings=chat_dataset_settings)

    dataset = dataset_cls(tokenizer=tokenizer_llama2, source=source, settings=dataset_settings, seed=42)

    assert len(data_dicts) == len(dataset)

    for data_dict, sample in zip(data_dicts, dataset):
        record = ClassificationDatasetRecord.model_validate(data_dict)

        assert record.label == sample['labels']
        assert isinstance(sample['labels'], list)
        assert all(label in (0, 1) for label in sample['labels'])

        assert is_sample_build_from_content(
            sample['input_ids'], [m.content for m in record.messages], tokenizer_llama2
        )


def test_pair_preferences(tokenizer_llama2, chat_dataset_settings, pair_preferences_dataset_source):
    # load dataset and check that samples have required fields

    source, data_dicts = pair_preferences_dataset_source

    dataset_cls = DatasetRegistry.by_name(DatasetType.PAIR_PREFERENCES).by_name(DatasetStrategy.TRAIN)

    dataset_settings = PairPreferenceDatasetSettings(chat_settings=chat_dataset_settings)
    dataset = dataset_cls(tokenizer=tokenizer_llama2, source=source, settings=dataset_settings, seed=42)

    assert len(data_dicts) == len(dataset)

    for data_dict, sample in zip(data_dicts, dataset):
        record = PairPreferenceRecord.model_validate(data_dict)
        context: list[str] = [c.content for c in record.context]
        contents_w = [*context, record.answer_w.content]
        assert is_sample_build_from_content(sample['inputs_w']['input_ids'], contents_w, tokenizer_llama2)

        contents_l = [*context, record.answer_l.content]
        assert is_sample_build_from_content(sample['inputs_l']['input_ids'], contents_l, tokenizer_llama2)

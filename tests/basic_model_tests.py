# Требования:
# - загружать заданные параметры для модели - train, test, unlabeled
# - предобрабатывать данные для обучения
# - контрольные точки на каждом этапе обучения
# - указать этап в конфиге и начать с него
# - сохранять и загружать веса модели
# - сохранять и загружать лучшие гиперпараметры модели
# - предсказывать на загруженной и подготовленной модели
# - предобрабатывать данные для предсказания отдельно

import unittest
from models.basic_model import BasicModel
from unittest.mock import MagicMock, Mock, call
import pandas as pd

class BasicModelTestCase(unittest.TestCase):
    def test_load_parameters(self):
        config = {
            "data_folder": "folder",
            "train_data_path": "train",
            "test_data_path": "test",
            "unlabeled_data_path": "unlabeled",
            "results_folder": "results",
            "stage": "stage"
        }
        model = BasicModel(config)
        assert model.train_data_path == "train"
        assert model.test_data_path == "test"
        assert model.unlabeled_data_path == "unlabeled"
        assert model.data_folder == "folder"
        assert model.results_folder == "results"
        assert model.stage == "stage"

    def test_preprocess_train_data(self):
        train_df = pd.DataFrame(
            [["text", "additional", 1]],
            columns=["text", "additional", "label"]
        )
        train_df.to_csv("train.csv")
        config = {
            "data_folder": "./",
            "train_data_path": "train.csv"
        }
        model = BasicModel(config)

        def remove_additional(df):
            return df.drop(columns=["additional"])

        model._preprocess_data = MagicMock(side_effect=remove_additional)
        model._preprocess_and_save_data(config["train_data_path"])
        preprocessed_train_df = pd.read_csv("./preprocessed_train.csv")

        assert "text" in preprocessed_train_df.columns
        assert "label" in preprocessed_train_df.columns
        assert "additional" not in preprocessed_train_df.columns

    def test_init_model(self):
        model = BasicModel({
            "train_data_path": "train",
            "unlabeled_data_path": "unlabeled",
            "test_data_path": "test"
        })
        model._preprocess_and_save_data = MagicMock()
        model._load = MagicMock()
        model.init()

        model._preprocess_and_save_data.assert_any_call("train")
        model._preprocess_and_save_data.assert_any_call("unlabeled")
        model._preprocess_and_save_data.assert_any_call("test")
        model._load.assert_called_with()

    def test_train_with_stages(self):
        model = BasicModel({
            "stage": "second"
        })
        model._first_stage = MagicMock()
        model._second_stage = MagicMock()
        model._third_stage = MagicMock()
        model._save = MagicMock()
        model.stages = [
            ("first", model._first_stage),
            ("second", model._second_stage),
            ("third", model._third_stage)
        ]
        process = Mock(
            first=model._first_stage,
            second=model._second_stage,
            third=model._third_stage,
            save=model._save
        )
        model.train()

        process.assert_has_calls([
            call.second(),
            call.save(),
            call.third(),
            call.save()
        ])
        model._first_stage.assert_not_called()

    def test_submit(self):
        model = BasicModel({
            "results_folder": "./"
        })
        test_X = pd.DataFrame([[0, 1], [1, 0]], columns=["y", "n"])
        test_y = pd.DataFrame([[0, 1], [1, 1]], columns=["id", "prediction"])
        model._predict = MagicMock(return_value=[1, 1])
        model.submit(test_X)
        results = pd.read_csv("./submission.csv")

        self.assertSequenceEqual(results.to_dict(), test_y.to_dict())

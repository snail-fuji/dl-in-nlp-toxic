import pandas as pd

class BasicModel:
    stages = []

    def __init__(self, config):
        self.train_data_path = config.get("train_data_path")
        self.test_data_path = config.get("test_data_path")
        self.unlabeled_data_path = config.get("unlabeled_data_path")
        self.data_folder = config.get("data_folder")
        self.results_folder = config.get("results_folder")
        self.stage = config.get("stage")

    def _preprocess_and_save_data(self, file_path):
        data = pd.read_csv("{}/{}".format(self.data_folder, file_path))
        preprocessed_data = self._preprocess_data(data)
        preprocessed_data.to_csv("{}/preprocessed_{}".format(self.data_folder, file_path))

    def _preprocess_data(self, data):
        raise Exception("Not implemented")

    def _load(self):
        raise Exception("Not implemented")

    def _save(self):
        raise Exception("Not implemented")

    def init(self):
        self._preprocess_and_save_data(self.unlabeled_data_path)
        self._preprocess_and_save_data(self.train_data_path)
        self._preprocess_and_save_data(self.test_data_path)
        self._load()

    def train(self):
        first_stage = [i for i, (name, _) in enumerate(self.stages) if name == self.stage][0]
        for name, method in self.stages[first_stage:]:
            print("Stage: ", name)
            method()
            self._save()

    def _predict(self, data):
        raise Exception("Not implemented")

    def submit(self, data):
        answers = self._predict(data)
        answers_df = pd.DataFrame()
        answers_df["id"] = data.index
        answers_df["prediction"] = answers
        answers_df.to_csv("{}/{}".format(self.results_folder, "submission.csv"), index=False)
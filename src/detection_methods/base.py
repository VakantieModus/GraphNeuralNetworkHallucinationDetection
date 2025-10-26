from abc import ABC, abstractmethod


class BaseDetectionMethod(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def train(self, train_data: list[dict], validation_data: list[dict] = None):
        pass

    @abstractmethod
    def evaluate(self, test_data: list[dict]):
        pass

    @abstractmethod
    def save_model(self, directory: str, model_name: str):
        pass

    @abstractmethod
    def load_model(self, directory: str, model_prefix: str):
        pass

from abc import ABC, abstractmethod
from config import Config
from core.networks.layers import Layer


class BaseModel(ABC):
    def __init__(self, layers: list, config: Config = Config()):
        super(BaseModel, self).__init__()
        self._model = None
        self._early_stop_range = config.early_stop_range

    @property
    def model(self):
        return self._model

    def train(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.train(), self

    @abstractmethod
    def build_model(self):
        raise NotImplementedError


class HyperParameterSearchable:

    def run(self):
        pass

# Pensar immutables e experimentos
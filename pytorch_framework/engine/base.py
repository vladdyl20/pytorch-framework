from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    @abstractmethod
    def train_step(self, dataloader):
        pass

    @abstractmethod
    def test_step(self, dataloader):
        pass

    @abstractmethod
    def fit(self, train_loader, test_loader, epochs):
        pass
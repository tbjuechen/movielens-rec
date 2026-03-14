from abc import ABC, abstractmethod

class BaseRecallModel(ABC):
    @abstractmethod
    def fit(self, train_data, **kwargs):
        """Train the recall model on historical data."""
        pass

    @abstractmethod
    def retrieve(self, user_ids, k=100):
        """Retrieve top-K items for given users."""
        pass

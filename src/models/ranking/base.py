from abc import ABC, abstractmethod

class BaseRankingModel(ABC):
    @abstractmethod
    def fit(self, train_data, **kwargs):
        """Train the ranking model on historical data."""
        pass

    @abstractmethod
    def predict(self, user_item_pairs):
        """Predict CTR / conversion rate for given user-item pairs."""
        pass

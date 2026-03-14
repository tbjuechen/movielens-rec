class FeatureFetcher:
    def __init__(self, feature_store_dir):
        self.feature_store_dir = feature_store_dir

    def fetch(self, user_ids, item_ids):
        """Fetch static features for users and items from the feature store."""
        pass

class RetrieverScheduler:
    def __init__(self, recall_models):
        # Orchestrate multiple recall base models
        self.recall_models = recall_models

    def fetch_candidates(self, user_ids, k=100):
        pass

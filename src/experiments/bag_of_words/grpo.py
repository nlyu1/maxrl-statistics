from src.experiments.bag_of_words.config import BagOfWordsStudyBaseConfig
from src.experiments.common import GRPOConfig


class BagOfWordsGRPOConfig(BagOfWordsStudyBaseConfig):
    grpo: GRPOConfig

    # Whatever
    def __model_post_init__():
        """
        You can assert here that the grpo num_rollout_samples
        divides the train_batch_size.
        """

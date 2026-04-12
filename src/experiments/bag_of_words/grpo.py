from src.experiments.bag_of_words.config import BagOfWordsStudyBaseConfig
from src.experiments.bag_of_words.sft import BagOfWordsSFTConfig
from src.experiments.common import GRPOConfig


class BagOfWordsGRPOConfig(BagOfWordsStudyBaseConfig):
    grpo: GRPOConfig
    sft: BagOfWordsSFTConfig
    num_sft_steps: int
    """
    You can always assume that num_sft_steps is less than 1 epoch
        (feel free to assert) as well.
    SFT first for some steps, then GRPO
    """

    # Use the correct method!
    def __model_post_init__():
        """
        You can assert here that the grpo num_rollout_samples
        divides the train_batch_size.
        """

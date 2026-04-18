from src.experiments.bag_of_words.config import BagOfWordsStudyBaseConfig
from src.experiments.bag_of_words.sl import BagOfWordsSLConfig
from src.experiments.common import GRPOConfig


class BagOfWordsGRPOConfig(BagOfWordsStudyBaseConfig):
    """
    Zero-step GRPO rollout config
    """

    grpo: GRPOConfig
    sl: BagOfWordsSLConfig
    num_sl_steps: int
    sl_checkpoint_folder: Path
    proj_loss_temperature: float

    # Use the correct method!
    def __model_post_init__():
        """
        You can assert here that the grpo num_rollout_samples
        divides the train_batch_size.
        """

    def get_state_cls(self) -> type["BagOfWordsSLState"]:
        return BagOfWordsSLState


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class BagOfWordsGRPOState(BagOfWordsStudyBaseState):
    pass

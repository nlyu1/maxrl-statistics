from src.experiments.bag_of_words.config import BagOfWordsStudyBaseConfig
from src.experiments.bag_of_words.sft import BagOfWordsSFTConfig
from src.experiments.common import GRPOConfig


class BagOfWordsGRPOConfig(BagOfWordsStudyBaseConfig):
    grpo: GRPOConfig
    sft: BagOfWordsSFTConfig
    num_sft_steps: int
    sft_checkpoint_folder: Path
    proj_loss_temperature: float
    """
    You can always assume that num_sft_steps is less than 1 epoch
        (feel free to assert) as well.
    SFT first for some steps, then GRPO.

    For the proj_loss_logsumexp_beta, we only supervise the best rollout's supervised loss.
    - Temperature -> 0 corresponds to best rollout
    - Tempeature -> high corresponds to mean across rollouts
    Make sure to scale the GRPO policy v. projection losses consistent scale (average proj loss across rollouts).
    """

    # Use the correct method!
    def __model_post_init__():
        """
        You can assert here that the grpo num_rollout_samples
        divides the train_batch_size.
        """

    def get_state_cls(self) -> type["BagOfWordsSFTState"]:
        return BagOfWordsSFTState


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class BagOfWordsGRPOState(BagOfWordsStudyBaseState):
    pass

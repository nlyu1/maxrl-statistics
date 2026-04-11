class TokenizedParquetDatasetConfig(BaseConfig):
    tokenizer_model_name: str
    folder: Path


    filter_samples_above_n_tokens: int
    pad_to_multiple: int

    def to_dataset(self) -> "TokenizedParquetDataset":
        """
        Our working folder is folder/{tokenizer_model_name}
        1. Asserts that "folder" exists
        2. Reads train & val
        3. Instantiates the auto-tokenizer and vector-tokenizes the full dataset.
            - Filter out samples above n tokens
            - Calculates the max train & val num_tokens (uncanonical, I know), prints it,
                then computes a single ceil_padded_seqlen
        4. Instantiates tensors with padding
        """
        ...

class TokenizedParquetDataset:
    config: TokenizedParquetDatasetConfig

    train_tokens: Int[Tensor, "num seq"]
    train_targets: Float[Tensor, "num seq"]
    train_ground_truth: Floath[Tensor, "num seq"]

    val_tokens: Int[Tensor, "num seq"]
    ...

    # When saved, these should be directly horizontally concat-able onto the original parquets
    unfiltered_train_token_lengths: Int[Tensor, "num"]
    unfiltered_val_token_lengths: Int[Tensor, "num"]

    @classmethod
    def from_folder(
        self, *,
        tokenizer_model_name: str,
        folder: Path
    )

    def to_folder(
        self,
        exists_ok: bool = False
    ):
        """
        Serializes to folder. Prefer parquet serialization (ints)
        """

    @property
    def ceil_padded_seqlen
    # Computed from shape[1] of train / val

    @property
    def num_val_samples

    @property
    def num_train_samples
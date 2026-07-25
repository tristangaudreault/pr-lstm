from transformers import PretrainedConfig


class PRLMConfig(PretrainedConfig):
    model_type = "prlm"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=512,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

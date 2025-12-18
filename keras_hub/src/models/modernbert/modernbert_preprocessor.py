import keras
from keras import ops
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.api_export import keras_hub_export

@keras_hub_export("keras_hub.models.ModernBertPreprocessor")
class ModernBertPreprocessor(Preprocessor):
    def __init__(self, tokenizer, sequence_length=512, mlm_probability=0.15, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.mlm_probability = mlm_probability

    def call(self, inputs):
        outputs = self.tokenizer(
            inputs,
            padding="max_length",
            truncation=True,
            max_length=self.sequence_length,
            return_tensors= "tf",
        )
        input_ids = outputs["input_ids"]
        attention_mask = outputs["attention_mask"]
        labels = ops.identity(input_ids)

        probability_matrix = ops.random.uniform(ops.shape(input_ids))
        mask = (probability_matrix < self.mlm_probability) & (input_ids != self.tokenizer.pad_token_id)

        labels = ops.where(mask, labels, -100)
        mask_token_mask = ops.random.uniform(ops.shape(input_ids)) < 0.8
        input_ids = ops.where(mask & mask_token_mask, self.tokenizer.mask_token_id, input_ids)

        return {
            "token_ids": input_ids,
            "padding_mask": attention_mask,
            "labels": labels,
        }

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "mlm_probability": self.mlm_probability,
        })
        return config

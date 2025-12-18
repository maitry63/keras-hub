import keras
from keras import layers
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.task import Task
from keras_hub.src.utils.keras_utils import gelu_approximate

@keras_hub_export("keras_hub.models.ModernBertMaskedLM")
class ModernBertMaskedLM(Task):
    def __init__(self, backbone, preprocessor=None, **kwargs):
        self.backbone = backbone
        self.preprocessor = preprocessor

        sequence_output = backbone.output
        
        # Get the embedding layer for weight tying
        token_embedding = backbone.get_layer("token_embedding")

        # First projection head
        x = layers.Dense(
            backbone.hidden_dim,
            activation=gelu_approximate,
            name="mlm_dense",
        )(sequence_output)

        x = layers.LayerNormalization(
            epsilon=backbone.layer_norm_epsilon,
            name="mlm_layer_norm",
        )(x)

        # Weight tying: project back to vocab size using embedding weights
        logits = token_embedding(x, reverse=True)

        super().__init__(
            inputs=backbone.inputs,
            outputs=logits,
            **kwargs,
        )

    def compute_loss(self, x, y, y_pred, sample_weight=None):
        loss = keras.losses.sparse_categorical_crossentropy(
            y,
            y_pred,
            from_logits=True,
        )
        mask = keras.ops.not_equal(y, -100)
        loss = keras.ops.sum(loss * mask) / keras.ops.sum(mask)
        return loss


    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone": keras.saving.serialize_keras_object(self.backbone),
                "preprocessor": keras.saving.serialize_keras_object(self.preprocessor),
            }
        )
        return config

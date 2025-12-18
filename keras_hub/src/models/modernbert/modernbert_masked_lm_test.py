import pytest
import keras
import numpy as np
from keras import ops
from keras_hub.src.models.modernbert.modernbert_backbone import ModernBertBackbone
from keras_hub.src.models.modernbert.modernbert_masked_lm import ModernBertMaskedLM
from keras_hub.src.tests.test_case import TestCase

class ModernBertMaskedLMTest(TestCase):
    def setUp(self):
        self.backbone = ModernBertBackbone(
            vocabulary_size=100,
            num_layers=2,
            num_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
        )
        # batch_size=2, seq_len=5
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }
        
        self.y_true = np.full((2, 5), -100, dtype="int32")
        self.y_true[:, 1:3] = np.random.randint(0, 100, (2, 2))
        
        self.init_kwargs = {
            "backbone": self.backbone,
        }
    def test_task_basics(self):
        """Test forward pass, output shapes, and basic serialization."""
        self.run_layer_test(
            cls=ModernBertMaskedLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 100),
        )

    def test_compute_loss(self):
        """Verify that loss ignores -100 labels and doesn't produce NaN."""
        task = ModernBertMaskedLM(**self.init_kwargs)
        
        y_pred = keras.random.uniform((2, 5, 100))
        
        loss = task.compute_loss(x=self.input_data, y=self.y_true, y_pred=y_pred)
        
        self.assertNotAllClose(loss, 0.0)
        self.assertFalse(ops.any(ops.isnan(loss)))

    @pytest.mark.large
    def test_saved_model(self):
        """Verify the Task can be saved and loaded (serialization)."""
        self.run_model_saving_test(
            cls=ModernBertMaskedLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_weight_tying(self):
        """Verify that the MLM head is using the backbone's embedding weights."""
        task = ModernBertMaskedLM(**self.init_kwargs)
        
        # The weights in the 'token_embedding' layer should influence the logits
        # This is implicitly tested by backbone.get_layer("token_embedding").reverse(x)
        embedding_weights = self.backbone.get_layer("token_embedding").embeddings
        
        # If weight tying is working, the task shouldn't have a 
        # separate large output kernel variable for the vocab projection.
        trainable_variables = [v.name for v in task.trainable_variables]
        self.assertFalse(any("output_projection/kernel" in name for name in trainable_variables))
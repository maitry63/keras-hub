import pytest
from keras import ops
from keras_hub.src.models.modernbert.modernbert_backbone import ModernBertBackbone
from keras_hub.src.tests.test_case import TestCase

class ModernBertBackboneTest(TestCase):
    """Tests for ModernBERT backbone model."""

    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 8,
            "intermediate_dim": 16,
            "dropout": 0.0,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        """Verify the backbone runs and produces correct output shapes."""
        self.run_backbone_test(
            cls=ModernBertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 8),
        )

    @pytest.mark.large
    def test_saved_model(self):
        """Verify the model can be saved and loaded accurately."""
        self.run_model_saving_test(
            cls=ModernBertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_mixed_precision(self):
        """Verify the backbone works correctly with mixed precision policies."""
        self.run_backbone_test(
            cls=ModernBertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 8),
            run_mixed_precision_check=True,
        )

    def test_variable_sequence_length(self):
        """Ensure the backbone handles different sequence lengths via RoPE."""
        model = ModernBertBackbone(**self.init_kwargs)
        # Test with a different length than setUp
        longer_input = {
            "token_ids": ops.ones((1, 10), dtype="int32"),
            "padding_mask": ops.ones((1, 10), dtype="int32"),
        }
        output = model(longer_input)
        self.assertEqual(ops.shape(output), (1, 10, 8))
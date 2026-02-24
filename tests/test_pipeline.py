import torch
# Replace with your actual import
# from src.model import MyNTPModel


def test_tensor_shapes():
    # Example: Check if your dummy input produces expected output shape
    batch_size = 4
    seq_len = 16
    vocab_size = 1000

    # Simulate some input tokens
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Mock check: logic should ensure output is (batch, seq, vocab)
    assert input_ids.shape == (4, 16)
    print("Shape test passed!")

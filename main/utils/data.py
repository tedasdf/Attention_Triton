from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers import models, trainers, pre_tokenizers, decoders


def get_titles(
    num_titles: int, seed: int, val_frac: float, data_dir: str = "data"
) -> tuple[list[str], list[str]]:
    # Use your persistent storage for the HuggingFace cache
    # This ensures that if you run this again, it's even faster!
    hf_cache_dir = Path(data_dir) / "hf_cache"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Use streaming=True and point to our external storage
    ds = load_dataset(
        "julien040/hacker-news-posts",
        split="train",
        streaming=True,
        cache_dir=str(hf_cache_dir),  # <--- Crucial for your 3090 setup
    )

    # 2. Shuffle with a buffer
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    # 3. Take samples
    sampled_rows = list(ds.take(num_titles))
    titles = [row["title"].strip() for row in sampled_rows if row.get("title")]

    n = int(num_titles * (1 - val_frac))
    return titles[:n], titles[n:]


def train_tokenizer(
    titles: list[str],
    vocab_size: int,
    unk_token: str = "<unk>",
    pad_token: str = "<pad>",
    eos_token: str = "<eos>",
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=[pad_token, eos_token, unk_token]
    )
    tokenizer.train_from_iterator(titles, trainer)
    return tokenizer


class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tk = tokenizer
        self.stoi = {tok: i for tok, i in tokenizer.get_vocab().items()}
        self.itos = {i: tok for tok, i in tokenizer.get_vocab().items()}

    def encode(self, s: str) -> list[int]:
        return self.tk.encode(s).ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self):
        return self.tk.get_vocab_size()

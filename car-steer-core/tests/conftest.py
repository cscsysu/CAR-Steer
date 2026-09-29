import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM,
                          LlamaConfig, LlamaForCausalLM, MistralConfig, MistralForCausalLM)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(42)
    torch.set_num_threads(1)


@pytest.fixture
def tokenizer():
    words = ["[UNK]", "[BOS]", "[EOS]", "[PAD]", "Context", ":", "Mira", "Sen", ".",
             "is", "the", "director", "Question", "Who", "?", "Answer", "with", "shortest",
             "span", "from", "context", "Other", "1998", "-", "Paris", " paris", " Paris",
             "\n", " ", "foo", "bar", "and"]
    backend = Tokenizer(WordLevel(dict(zip(words, range(len(words)))), unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                  bos_token="[BOS]", eos_token="[EOS]", pad_token="[PAD]")


@pytest.fixture(params=["qwen3", "llama", "mistral"])
def tiny_model(request, tokenizer):
    classes = {"qwen3": (Qwen3Config, Qwen3ForCausalLM),
               "llama": (LlamaConfig, LlamaForCausalLM),
               "mistral": (MistralConfig, MistralForCausalLM)}
    config_cls, model_cls = classes[request.param]
    kwargs = dict(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=48,
                  num_hidden_layers=6, num_attention_heads=2, num_key_value_heads=1,
                  max_position_embeddings=256, bos_token_id=1, eos_token_id=2,
                  pad_token_id=3, attention_dropout=0.0, tie_word_embeddings=False)
    if request.param == "qwen3":
        kwargs.update(head_dim=16, use_sliding_window=False)
    if request.param == "mistral":
        kwargs.update(sliding_window=128)
    config = config_cls(**kwargs)
    config._attn_implementation = "eager"
    return model_cls(config).eval()

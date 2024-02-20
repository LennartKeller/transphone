from tokenizers import Regex, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import NFC, Lowercase, Replace, Sequence, Strip
from tokenizers.pre_tokenizers import Sequence as PreTokenizerSequence
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast

# We can't remove '<', '>' during preprocessing because this destroys lang-tokens
# PUNCTUATION_REGEX = (
#     r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\=\?\@\[\\\]\^\_\`\{\|\}\~0123456789]"
# )
# In some languages (e.g. persian) numbers get romanized in latter stages of the pipeline and are transcribed
# so we can't remove numbers in the tokenizer
PUNCTUATION_REGEX = (
    r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\=\?\@\[\\\]\^\_\`\{\|\}\~]"
)


def build_tokenizer(vocab_file):
    tokens = [t.strip() for t in vocab_file.read_text().split("\n") if t.strip()]
    vocab = {t: i for i, t in enumerate(tokens)}

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<pad>"))
    normalizers = [Lowercase(), Strip(), NFC(), Replace(Regex(PUNCTUATION_REGEX), "")]

    tokenizer.normalizer = Sequence(normalizers)

    # tokenizer.pre_tokenizer = Split(r"", behavior="removed")

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>", eos_token="<eos>")
    tokenizer.model_max_length = 5000
    return tokenizer

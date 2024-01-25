import json
from pathlib import Path
from timeit import default_timer as timer

import torch
from more_itertools import batched
from tqdm.auto import tqdm

from transphone.g2p import read_g2p
from transphone.model.jit import JitTransformerG2P
from transphone.model.transformer import TransformerG2P

if __name__ == "__main__":

    
    g2p_model = read_g2p()


    def encode(string: str, lang: str) -> torch.Tensor:
        string = string.lower()
        graphemes = [f"<{lang.lower()}>"]+ list(string)
        encoded = [g2p_model.grapheme_vocab.atoi(g) for g in graphemes]
        return torch.tensor(encoded, dtype=torch.long)

    
    def decode(out):
        out = out.tolist()
        for entry in out:
            entry = [i for i in entry if i > 1]
            phones = [g2p_model.phoneme_vocab.itoa(pid) for pid in entry]
            print("".join(phones))
    

    
    kwargs = {
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "emb_size": 512,
        "nhead": 8,
        "src_vocab_size": 5224,
        "tgt_vocab_size": 427
    }
    
    
    


    grapheme_ids = encode("Hello", "eng")
    grapheme_ids_batched = grapheme_ids.expand(2, -1).contiguous()
    print("grapheme_ids", grapheme_ids)
    print("grapheme_ids.size()", grapheme_ids.size())

    weights = g2p_model.model.state_dict()

    model = TransformerG2P(**kwargs)
    model.load_state_dict(weights)    
    out = model.inference_batch(grapheme_ids_batched)
    print("NORMAL")
    decode(out)

    _model = JitTransformerG2P(**kwargs)
    _model.load_state_dict(weights)
    scripted_model = torch.jit.script(_model)
    scripted_out = scripted_model.transcribe(grapheme_ids_batched)
    print("SCRIPTED")
    decode(scripted_out)

    print("benchmarking scripted model")
    from torch.profiler import ProfilerActivity, profile, record_function
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            model.inference(grapheme_ids_batched)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    

    words = Path("/Users/lennartkeller/Downloads/latest 2/words.txt").read_text().split("\n")
    time_data = []
    batch_size = 64
    device = "cpu"
    scripted_model = scripted_model.to(device)
    batches = list(batched(words, batch_size))
    pbar = tqdm(batches)
    for batch in pbar:
        encoded_words = []
        max_len = 0
        for word in batch:
            word = word.lower()
            encoded = encode(word, lang="eng")
            len_enc = encoded.size(0)
            if len_enc > max_len:
                max_len = len_enc
            encoded_words.append(encoded)
        
        encoded = torch.nn.utils.rnn.pad_sequence(encoded_words, batch_first=True, padding_value=0)
        encoded = encoded.to(device)

        with torch.no_grad():
            start = timer()
            output = model.inference(encoded)
            end = timer()
        elapsed = end - start
        time_data.append({
            "time": elapsed,
            "word": word,
            "input": encoded.tolist(),
            "output": output.tolist()
            })
        pbar.set_description(f"{elapsed:.4f} [{', '.join(map(str, encoded.size()))}]")
    Path("time-data.json").write_text(json.dumps(time_data, indent=4))

    


    




    

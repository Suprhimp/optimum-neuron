import argparse
from typing import List

import torch
from transformers import AutoTokenizer
from transformers_neuronx.llama.model import LlamaForSampling


def get_model(model_path):
    model = LlamaForSampling.from_pretrained(model_path, batch_size=2, amp="f16", tp_degree=2, n_positions=2048)
    model.to_neuron()
    return model


def get_padded_inputs(input_lengths: List[int], mask_inputs: bool = True):
    prompt = (
        "It was a bright cold day in April, and the clocks were striking thirteen."
        " Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind,"
        " slipped quickly through the glass doors of Victory Mansions, though not quickly enough"
        " to prevent a swirl of gritty dust from entering along with him."
    )
    t = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    tokens = t(prompt)["input_ids"]
    input_ids = []
    max_length = max(input_lengths)
    for input_length in input_lengths:
        if input_length > len(tokens):
            raise ValueError(f"Input length should be lower than {len(tokens)}")
        if input_length == max_length:
            ids = tokens[:input_length]
        else:
            ids = [
                t.eos_token_id,
            ] * (
                max_length - input_length
            ) + tokens[:input_length]
        input_ids.append(ids)
    input_ids = torch.tensor(input_ids)
    print(f"Using padded inputs of length {input_lengths}.")
    start_ids = None
    if mask_inputs:
        start_ids = torch.argmax((input_ids != t.eos_token_id).to(torch.int64), dim=1)
        print(f"Using masked inputs, starting at offsets: {start_ids}")
    return {"input_ids": input_ids, "cache_ids": None, "start_ids": start_ids}


def greedy(model, inputs):
    scores = model(**inputs)
    return torch.argmax(scores, dim=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--input-length", type=int, default=64)
    parser.add_argument("--mask-inputs", action="store_true")
    args = parser.parse_args()
    input_lengths = (args.input_length, args.input_length - 1)
    inputs = get_padded_inputs(input_lengths, mask_inputs=args.mask_inputs)
    model = get_model(args.model_path)
    scores = model(**inputs)
    # Greedy
    tokens = greedy(model, inputs)
    print(tokens)
    assert torch.all(tokens != 0)

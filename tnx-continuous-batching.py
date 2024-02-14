import os

import torch
from transformers import AutoTokenizer
from transformers.models.llama import LlamaForCausalLM
from transformers_neuronx.config import ContinuousBatchingConfig, NeuronConfig
from transformers_neuronx.llama.model import LlamaForSampling
from transformers_neuronx.module import save_pretrained_split
from transformers_neuronx.sampling import select_tokens


model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"

if not os.path.exists(f"{model_name_or_path}-split"):

    hf_model = LlamaForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True)
    save_pretrained_split(hf_model, f"{model_name_or_path}-split")


max_model_len = 2048
max_num_seqs = 4
tp_degree = 8

continuous_batching_config = ContinuousBatchingConfig(batch_size_for_shared_caches=max_num_seqs)
neuron_config = NeuronConfig(continuous_batching=continuous_batching_config)
kwargs = {
    "tp_degree": tp_degree,
    "amp": "f16",
    "neuron_config": neuron_config,
    "context_length_estimate": [max_model_len],
    "n_positions": [max_model_len],
    "batch_size": max_num_seqs,
}
neuron_model = LlamaForSampling.from_pretrained(f"{model_name_or_path}-split", **kwargs)
neuron_model.to_neuron()

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
eos_token_id = tokenizer.eos_token_id
test_prompts = [
    "[INST] What is your favourite condiment? [/INST]",
    "[INST] What are the 2 things every sandwich has in common? [/INST]",
    "[INST] What are 6 common condiments for a sandwich? [/INST]",
]
tokenizer.padding_side = "left"
tokenizer.pad_token_id = eos_token_id
input_tokens = tokenizer(test_prompts, padding=True)
all_input_ids, all_attention_masks = input_tokens["input_ids"], input_tokens["attention_mask"]


# %%
# Before the first (multi-batch) prompt encoding, let's write a simple function for padding inputs.
def _pad_to_max(x):
    max_len = len(x[0])
    for item in x:
        max_len = max(max_len, len(item))
    for idx, item in enumerate(x):
        x[idx] = [0] * (max_len - len(x[idx])) + x[idx]
    return x


# %% [markdown]
# ## Decode first batch of prompts

# %%
# Prepare inputs for the first prompt encoding. Here, we will take first two prompts from the list of prompts.
# The essential inputs for prompt encoding would include: 1) (padded) input_ids, 2) cache_ids (aka position ids), 3) seq_ids
input_ids, attention_mask = all_input_ids[:max_num_seqs], all_attention_masks[:max_num_seqs]
n_active_seqs = len(input_ids)
input_ids = torch.tensor(_pad_to_max(input_ids))
attention_mask = torch.tensor(_pad_to_max(attention_mask))
seq_ids = torch.arange(n_active_seqs)
batch_size, context_len = input_ids.shape
cache_ids = torch.zeros_like(input_ids)
_, first_ids = attention_mask.max(axis=1)
for i in range(batch_size):
    first_id = first_ids[i]
    cache_ids[i, first_id:] = torch.arange(context_len - first_id)
breakpoint()

# The first prompt encoding
with torch.inference_mode():
    logits = neuron_model(input_ids, cache_ids=cache_ids, start_ids=seq_ids)
next_tokens = select_tokens(logits)
output_tokens = [[t] for t in next_tokens.flatten().tolist()]
cache_ids = cache_ids.max(dim=1, keepdim=True).values

# %% [markdown]
# Let's perform autoregressive decoding for all sequences, and stop when any of the sequences generates an EOS token.

# %%
min_context_len = cache_ids.min().item()

# autoregressive decoding
for idx in range(max_model_len - min_context_len - 1):
    input_ids = next_tokens
    cache_ids = cache_ids + 1
    batch_ids, _ = torch.where(cache_ids < max_model_len)
    with torch.inference_mode():
        logits = neuron_model(input_ids, cache_ids=cache_ids, start_ids=seq_ids)
    next_tokens = select_tokens(logits)
    for running_batch_id, batch_id in enumerate(batch_ids.tolist()):
        output_tokens[batch_id].append(next_tokens.flatten().tolist()[running_batch_id])
    if torch.any(next_tokens == eos_token_id):
        break

# Since we are going resume the token generation (aka decoding) process after handling sequence eviction and insertion, let's save the current status.
decode_next_tokens, decode_cache_ids, decode_seq_ids = next_tokens, cache_ids, seq_ids

# %%
# Let's take a brief look at the finished sequence.
finished_seq_ids = torch.concat(torch.where(next_tokens.flatten() == eos_token_id))
finished_seq_tokens = all_input_ids[finished_seq_ids] + output_tokens[finished_seq_ids]
print(f"Finished sequence: {tokenizer.batch_decode(torch.tensor([finished_seq_tokens]))}")

# Also, the unfinished sequences.
unfinished_seq_ids = torch.concat(torch.where(next_tokens.flatten() != eos_token_id))
unfinished_seq_tokens = [all_input_ids[id] + output_tokens[id] for id in unfinished_seq_ids]
print("Unfinished sequences:")
for tokens in unfinished_seq_tokens:
    print(tokenizer.batch_decode(torch.tensor([tokens])))

# %% [markdown]
# ## Insert a prompt encoding task
#
# Now, we have a sequence that generated EOS token. Let's evict such sequence from the running batch, and insert a new sequence there.

# %%
# We are able to flatten the next_tokens, since we decode one token per iteration.
new_seq_ids = torch.concat(torch.where(next_tokens.flatten() == eos_token_id))
new_input_ids, new_attention_mask = torch.tensor([all_input_ids[2]]), torch.tensor([all_attention_masks[2]])
n_active_seqs = len(new_input_ids)

# prompt encoding
batch_size, context_len = new_input_ids.shape
new_cache_ids = (
    torch.arange(context_len).reshape(1, context_len).expand(n_active_seqs, context_len).mul(new_attention_mask)
)
with torch.inference_mode():
    logits = neuron_model(new_input_ids, cache_ids=new_cache_ids, start_ids=new_seq_ids)
next_tokens = select_tokens(logits)

# %%
decode_next_tokens[new_seq_ids] = next_tokens
decode_cache_ids[new_seq_ids] = new_cache_ids.max(dim=1, keepdim=True).values
output_tokens[new_seq_ids] = torch.concat([next_tokens], dim=1).flatten().tolist()

# %% [markdown]
# ## Resume autoregressive decoding
#
# Okay, the new sequence has been inserted. Let's resume the token generation process for all sequences.

# %%
next_tokens, cache_ids = decode_next_tokens, decode_cache_ids

# autoregressive decoding
for idx in range(max_model_len - min_context_len - 1):
    input_ids = next_tokens
    cache_ids = cache_ids + 1
    (batch_ids,) = torch.where(
        torch.logical_and(cache_ids.flatten() < max_model_len, input_ids.flatten() != eos_token_id)
    )
    if len(batch_ids) == 0:
        break

    # trim inputs (we don't necessarily decode a sequence that generated EOS token.)
    input_ids = input_ids[batch_ids, :]
    cache_ids = cache_ids[batch_ids, :]
    seq_ids = seq_ids[batch_ids]

    with torch.inference_mode():
        logits = neuron_model(input_ids, cache_ids=cache_ids, start_ids=seq_ids)
    next_tokens = select_tokens(logits)

    for running_batch_id, seq_id in enumerate(seq_ids.tolist()):
        output_tokens[seq_id].append(next_tokens.flatten().tolist()[running_batch_id])

# %%
all_input_ids.pop(finished_seq_ids)
original_input_ids = all_input_ids
all_tokens = [it + ot for it, ot in zip(original_input_ids, output_tokens)]
print("Finished sequences after insertion and resume:")
for tokens in all_tokens:
    print(tokenizer.batch_decode(torch.tensor([tokens])))

# %%

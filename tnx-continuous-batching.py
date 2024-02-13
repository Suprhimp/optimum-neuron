# %% [markdown]
# # Continuous batching development guide on Inf2 & Trn1

# %% [markdown]
# In this example we compile and deploy the Hugging Face [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) model for tensor parallel inference on Neuron using the `transformers-neuronx` package.
#
# The example has the following main sections:
# 1. Set up the Jupyter Notebook
# 1. Install dependencies
# 1. Download the model
# 1. Construct the model|
# 1. Split the model `state_dict` into multiple files
# 1. Perform autoregressive sampling using tensor parallelism
#
# This Jupyter Notebook can be run on an Inf2 instance (`inf2.48xlarge`) or Trn1 instance (`trn1.32xlarge`).

# %% [markdown]
# ## Set up the Jupyter Notebook

# %% [markdown]
# The following steps set up Jupyter Notebook and launch this tutorial:
# 1. Clone the [AWS Neuron Samples](https://github.com/aws-neuron/aws-neuron-samples) repo to your instance using
# ```
# git clone https://github.com/aws-neuron/aws-neuron-samples.git
# ```
# 2. Navigate to the `transformers-neuronx` inference samples folder
# ```
# cd aws-neuron-samples/torch-neuronx/transformers-neuronx/inference
# ```
# 3. Follow the instructions in [Jupyter Notebook QuickStart](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/notebook/setup-jupyter-notebook-steps-troubleshooting.html) to run Jupyter Notebook on your instance.
# 4. Locate this tutorial in your Jupyter Notebook session (`mistral-7b-continuous-batching.ipynb`) and launch it. Follow the rest of the instructions in this tutorial.

# %% [markdown]
# ## Install Dependencies
# This tutorial requires the following pip packages:
#
#  - `torch-neuronx`
#  - `neuronx-cc`
#  - `sentencepiece`
#  - `transformers`
#  - `transformers-neuronx`
#
#
# Most of these packages will be installed when configuring your environment using the [torch-neuronx inference setup guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/torch-neuronx.html#setup-torch-neuronx). The additional dependencies must be installed here:

# %%
!pip install transformers-neuronx sentencepiece -U

# %% [markdown]
# ## Review the model license

# %% [markdown]
# Use of the Mistral 7B model is governed by the Apache License 2.0 and must be downloaded and converted to the standard Hugging Face format prior to running this sample.
#
# Follow the steps described in [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) to get access to the Mistral 7B model from Mistral AI and download the weights and tokenizer.
#
# After gaining access to the model checkpoints, you should be able to use the already converted checkpoints. Otherwise, if you are converting your own model, feel free to use the [conversion script](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py). The script can be called with the following (example) command:
# ```
# python src/transformers/models/llama/convert_llama_weights_to_hf.py \
#     --input_dir /path/to/downloaded/llama/weights --model_size 7Bf --output_dir ./Mistral-7B-Instruct-v0.1
# ```
#
# Note: For the purposes of this sample we assume you have agreed to save the Mistral-7B-Instruct-v0.1 model in a directory called `Mistral-7B-Instruct-v0.1` with the following format:
# ```
# Mistral-7B-Instruct-v0.1/
# ├── config.json
# ├── generation_config.json
# ├── pytorch_model-00001-of-00001.bin
# ├── pytorch_model-00002-of-00002.bin
# ├── pytorch_model.bin.index.json
# ├── special_tokens_map.json
# ├── tokenizer.json
# ├── tokenizer.model
# └── tokenizer_config.json
# ```

# %% [markdown]
# ## Download and split the model state_dict into multiple files

# %% [markdown]
# After downloading the model and converting it to the Hugging Face format we construct the model.
#
# For the sake of reducing host memory usage, it is recommended to save the model `state_dict` as
# multiple files, as opposed to one monolithic file given by `torch.save`. This "split-format"
# `state_dict` can be created using the `save_pretrained_split` function. With this checkpoint format,
# the Neuron model loader can load parameters to the Neuron device high-bandwidth memory (HBM) directly
# by keeping at most one layer of model parameters in the CPU main memory.

# %%
import os

model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.1"

if not os.path.exists(f"{model_name_or_path}-split"):
    from transformers.models.mistral import MistralForCausalLM
    from transformers_neuronx.module import save_pretrained_split

    hf_model = MistralForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True)
    save_pretrained_split(hf_model, f"{model_name_or_path}-split")

# %% [markdown]
# ## Compile the model with tensor parallelism

# %% [markdown]
# Now we have all of the necessary files for running `mistralai/Mistral-7B-Instruct-v0.1` autoregressive sampling.
#
# The memory required to host any model can be computed with:
# ```
# total memory = bytes per parameter * number of parameters
# ```
# When using `float16` casted weights for a 7 billion parameter model, this works out to `2 * 7B` or ~14GB of weights. Each NeuronCore has 16GB of memory which means that a 140GB model cannot fit on a single NeuronCore. In reality, the total space required is often greater than just the number of parameters due to caching attention layer projections (KV caching). This caching mechanism grows memory allocations linearly with sequence length and batch size.
#
# To get very large language models to fit on Inf2 & Trn1, tensor parallelism is used to split weights, data, and compute across multiple NeuronCores. The number of NeuronCores that the weights are split across can be controlled by setting the `tp_degree` parameter. This parallelism degree must be chosen to ensure that the memory usage per NeuronCore will be less than the physical 16GB limit. When configuring tensor parallelism, the memory per NeuronCore can be computed with:
#
# ```
# memory per core = (bytes per parameter * number of parameters) / tp_degree
# ```
#
# This can be used to compute the minimum instance sizing by ensuring that the value selected for `tp_degree` results in less than 16GB allocated per NeuronCore.
#
# Note that increasing the `tp_degree` beyond the minimum requirement almost always results in a faster model. Increasing the tensor parallelism degree improves memory bandwidth which improves model performance. To optimize performance it's recommended to use the highest tensor parallelism degree that is supported by the instance. In this sample we use tensor parallelism degree 8 and 8-bit quantization to optimize performance on `inf2.48xlarge` or `trn1.32xlarge`.
#
# We will use the Neuron `MistralForSampling` class to implement tensor parallelism for the Mistral 7B model. The default model config supports sampling up to sequence length 2048. Tensor parallelism is enabled through the argument `tp_degree=24`. We enable `float16` casting with the `amp='f16'` flag. The model computational graph is compiled by `neuronx-cc` for optimized inference on Neuron. In order to reduce weight memory usage, the 8-bit weight quantization is enabled by adding `QuantizationConfig` into `neuron_config` argument.

# %%
import time
import torch
from transformers import AutoTokenizer
from transformers_neuronx.mistral.model import MistralForSampling
from transformers_neuronx.config import NeuronConfig, ContinuousBatchingConfig

# %%
# load mistralai/Mistral-7B-Instruct-v0.1 to the NeuronCores with 2-way tensor parallelism and run compilation
max_model_len = 128
max_num_seqs = 2
tp_degree = 2

continuous_batching_config = ContinuousBatchingConfig(batch_size_for_shared_caches=max_num_seqs)
neuron_config = NeuronConfig(continuous_batching=continuous_batching_config)
kwargs = dict(tp_degree=tp_degree,
            amp='f32', neuron_config=neuron_config,
            context_length_estimate=[max_model_len],
            n_positions=[max_model_len],
            batch_size=max_num_seqs)
neuron_model = MistralForSampling.from_pretrained(f'{model_name_or_path}-split', **kwargs)
neuron_model.to_neuron()

# %% [markdown]
# ## Tokenize test prompts

# %%
# Construct a tokenizer and encode prompt text
from transformers_neuronx.sampling import select_tokens

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
eos_token_id = tokenizer.eos_token_id
test_prompts = [
    "[INST] What is your favourite condiment? [/INST]",
    "[INST] What are the 2 things every sandwich has in common? [/INST]",
    "[INST] What are 6 common condiments for a sandwich? [/INST]",
]
input_tokens = tokenizer(test_prompts)
all_input_ids, all_attention_masks = input_tokens["input_ids"], input_tokens["attention_mask"]

# %%
# Before the first (multi-batch) prompt encoding, let's write a simple function for padding inputs.
def _pad_to_max(x):
    max_len = len(x[0])
    for item in x:
        max_len = max(max_len, len(item))
    for idx, item in enumerate(x):
        x[idx] = x[idx] + [0] * (max_len - len(x[idx]))
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
cache_ids = torch.arange(context_len).reshape(1, context_len).expand(n_active_seqs, context_len).mul(attention_mask)

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
for idx in range(max_model_len-min_context_len-1):
    input_ids = next_tokens
    cache_ids = cache_ids + 1
    batch_ids, _ = torch.where(cache_ids<max_model_len)
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

# Also, the unfinished sequence.
unfinished_seq_ids = torch.concat(torch.where(next_tokens.flatten() != eos_token_id))
unfinished_seq_tokens = all_input_ids[unfinished_seq_ids] + output_tokens[unfinished_seq_ids]
print(f"Unfinished sequence: {tokenizer.batch_decode(torch.tensor([unfinished_seq_tokens]))}")

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
new_cache_ids = torch.arange(context_len).reshape(1, context_len).expand(n_active_seqs, context_len).mul(new_attention_mask)
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
for idx in range(max_model_len-min_context_len-1):
    input_ids = next_tokens
    cache_ids = cache_ids + 1
    batch_ids, = torch.where(torch.logical_and(cache_ids.flatten() < max_model_len, input_ids.flatten() != eos_token_id))
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
all_tokens = [it+ot for it, ot in zip(original_input_ids, output_tokens)]
print("Finished sequences after insertion and resume:")
for tokens in all_tokens:
    print(tokenizer.batch_decode(torch.tensor([tokens])))

# %%




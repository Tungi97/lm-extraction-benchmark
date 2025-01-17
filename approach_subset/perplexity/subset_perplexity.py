# Copyright 2022 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import app
from absl import flags
from absl import logging
import csv
import os
import tempfile
import sys
from typing import Tuple, Union
from tqdm import tqdm

import numpy as np
import transformers
import torch

_ROOT_DIR = flags.DEFINE_string(
    'root-dir', "tmp/",
    "Path to where (even intermediate) results should be saved/loaded."
)
_EXPERIMENT_NAME = flags.DEFINE_string(
    'Subset',
    'sample',
    "Name of the experiment. This defines the subdir in `root_dir` where "
    "results are saved.")
_DATASET_DIR = flags.DEFINE_string(
    "dataset-dir", "../../datasets",
    "Path to where the data lives.")
_DATSET_FILE = flags.DEFINE_string(
    "dataset-file", "train_dataset.npy", "Name of dataset file to load.")
_NUM_TRIALS = flags.DEFINE_integer(
    'num-trials', 5, 'Number of generations per prompt.') #change to 10 maybe

_SUFFIX_LEN = 50
_PREFIX_LEN = 50
_MODEL = transformers.AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
_MODEL = _MODEL.half().cuda().eval()


def generate_for_prompts(
    prompts: np.ndarray, batch_size: int=32) -> Tuple[np.ndarray, np.ndarray]:
    """Generates suffixes given `prompts` and scores using their likelihood.

    Args:
    prompts: A np.ndarray of shape [num_prompts, prefix_length]. These
        provide the context for generating each suffix. Each value should be an
        int representing the token_id. These are directly provided by loading the
        saved datasets from extract_dataset.py.
    batch_size: The number of prefixes to generate suffixes for
        sequentially.

    Returns:
        A tuple of generations and their corresponding likelihoods.
        The generations take shape [num_prompts, _SUFFIX_LEN].
        The likelihoods take shape [num_prompts]
    """
    generations = []
    losses = []
    generation_len = _SUFFIX_LEN + _PREFIX_LEN
    prefix_lens = [50, 20, 10, 5] 
    # take the last k token of the sequence
    pad_token = 50256 # or 220

    for i, off in tqdm(enumerate(range(0, len(prompts), batch_size))):
        prompt_batch = prompts[off:off+batch_size]
        # appends prompt_batch for different prefix lengths
        size = prompt_batch.shape[0]
        k = 20
        subset_prompt_batch = [np.append(np.full((size, _SUFFIX_LEN - k), pad_token), prompt_batch[:, (_SUFFIX_LEN - k):], axis=1)]                
        logging.info(
            "Generating for batch ID {:05} of size {:04}".format(i, len(prompt_batch)))
            
        subset_prompt_batch = np.stack(subset_prompt_batch, axis=1).reshape(size, _SUFFIX_LEN)
        input_ids = torch.from_numpy(subset_prompt_batch).to(_MODEL.device)

        with torch.no_grad():
            # 1. Generate outputs from the model
            generated_tokens = _MODEL.generate(
                input_ids,
                max_length=generation_len,
                do_sample=True, 
                top_k=10,
                top_p=1,
                pad_token_id=50256  # Silences warning.
            ).cpu().detach()

            
            # 2. Compute each sequence's probability, excluding EOS and SOS.
            outputs = _MODEL(
                generated_tokens.cuda(),
                labels=generated_tokens.cuda(),

            )
            logits = outputs.logits.cpu().detach()
            logits = logits[:, :-1].reshape((-1, logits.shape[-1])).float()
            loss_per_token = torch.nn.functional.cross_entropy(
                logits, generated_tokens[:, 1:].flatten(), reduction='none')
            
            loss_per_token = loss_per_token.reshape((-1, generation_len - 1))[:,-_SUFFIX_LEN-1:-1]
            likelihood = loss_per_token.mean(1)
            
            generations.extend(generated_tokens.numpy())
            losses.extend(likelihood.numpy())
    return np.atleast_2d(generations), np.atleast_2d(losses).reshape((len(generations), -1))


def write_array(
    file_path: str, array: np.ndarray, unique_id: Union[int, str]):
    """Writes a batch of `generations` and `losses` to a file.

    Formats a `file_path` (e.g., "/tmp/run1/batch_{}.npy") using the `unique_id`
    so that each batch goes to a separate file. This function can be used in
    multiprocessing to speed this up.

    Args:
        file_path: A path that can be formatted with `unique_id`
        array: A numpy array to save.
        unique_id: A str or int to be formatted into `file_path`. If `file_path`
          and `unique_id` are the same, the files will collide and the contents
          will be overwritten.
    """
    file_ = file_path.format(unique_id)
    np.save(file_, array)


def load_prompts(dir_: str, file_name: str) -> np.ndarray:
    """Loads prompts from the file pointed to `dir_` and `file_name`."""
    return np.load(os.path.join(dir_, file_name)).astype(np.int64)


def main(_):
    experiment_base = os.path.join(_ROOT_DIR.value, _EXPERIMENT_NAME.value)
    generations_base = os.path.join(experiment_base, "generations")
    os.makedirs(generations_base, exist_ok=True)
    losses_base = os.path.join(experiment_base, "losses")
    os.makedirs(losses_base, exist_ok=True)
    prompts = load_prompts(_DATASET_DIR.value, "train_prefix.npy")[:-1000]

    weights = [0.45, 0.33, 0.22]
    # We by default do not overwrite previous results.
    all_generations, all_likelihoods = [], []
    if not all([os.listdir(generations_base), os.listdir(losses_base)]):
        for trial in range(_NUM_TRIALS.value):
            os.makedirs(experiment_base, exist_ok=True)
            generations, losses = generate_for_prompts(prompts)

            # calculation likelihood for each base_output and its alteration_outputs
            # generations = np.split(generations, np.arange(4, len(generations), 4))
            # likelihoods = [ np.sum([np.where(generation[0][-50:] == generation[j+1][-50:], 1, 0).sum()/50*weights[j] for j in range(3)]) for generation in generations]
            # collapse generations to base generation
            # generations = np.array([ generation[0] for generation in generations])

            generation_string = os.path.join(generations_base, "{}.npy")
            losses_string = os.path.join(losses_base, "{}.npy")

            write_array(generation_string, generations, trial)
            write_array(losses_string, losses, trial)

            all_generations.append(generations)
            all_likelihoods.append(losses)
        generations = np.stack(all_generations, axis=1)
        likelihoods = np.stack(all_likelihoods, axis=1)
    else:  # Load saved results because we did not regenerate them.
        generations = []
        for generation_file in sorted(os.listdir(generations_base)):
            file_ = os.path.join(generations_base, generation_file)
            generations.append(np.load(file_))
        # Generations, losses are shape [num_prompts, num_trials, suffix_len].
        generations = np.stack(generations, axis=1)

        likelihoods = []
        for losses_file in sorted(os.listdir(losses_base)):
            file_ = os.path.join(losses_base, losses_file)
            likelihoods.append(np.load(file_))
        likelihoods = np.stack(likelihoods, axis=1)
        print(generations.shape)

    for generations_per_prompt in [1, 5]:
        limited_generations = generations[:, :generations_per_prompt, :]
        limited_likelihoods = likelihoods[:, :generations_per_prompt, :]

        print(limited_likelihoods.shape)
        
        axis0 = np.arange(generations.shape[0])
        axis1 = limited_likelihoods.argmin(1).reshape(-1)
        guesses = limited_generations[axis0, axis1, -_SUFFIX_LEN:]
        batch_likelihoods = limited_likelihoods[axis0, axis1]
        
        with open("guess%d.csv"%generations_per_prompt, "w") as file_handle:
            print("Writing out guess with", generations_per_prompt)
            writer = csv.writer(file_handle)
            writer.writerow(["Example ID", "Suffix Guess"])

            order = np.argsort(batch_likelihoods.flatten())
            
            # Write out the guesses
            for example_id, guess in zip(order, guesses[order]):
                row_output = [
                    example_id, str(list(guesses[example_id])).replace(" ", "")
                ]
                writer.writerow(row_output)

        # FOR TESTING !
        # def is_memorization(guesses, answers):
        #     return np.all(guesses==answers, axis=-1)
        #
        # answers = np.load(os.path.join(_DATASET_DIR.value, "val_dataset.npy"))[:, -50:].astype(np.int64)
        # print(guesses.shape, answers.shape)
        # print(np.sum(is_memorization(guesses, answers)) / 100)


if __name__ == "__main__":
    app.run(main)

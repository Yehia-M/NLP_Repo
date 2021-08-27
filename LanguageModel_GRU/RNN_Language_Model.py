#!/usr/bin/env python

import os
import trax
import trax.fastmath.numpy as np
import pickle
import numpy
import random as rnd
from trax import fastmath
from trax import layers as tl


def line_to_tensor(line, EOS_int=1):
    """
    Turns a line of text into a tensor

    Args:
        line (str): A single line of text.
        EOS_int (int, optional): End-of-sentence integer. Defaults to 1.

    Returns:
        list: a list of integers (unicode values) for the characters in the `line`.
    """

    tensor = []
    for c in line:
        c_int = ord(c)
        tensor.append(c_int)

    tensor.append(EOS_int)
    return tensor

def data_generator(batch_size, max_length, data_lines, line_to_tensor=line_to_tensor, shuffle=True):
    """Generator function that yields batches of data

    Args:
        batch_size (int): number of examples (in this case, sentences) per batch.
        max_length (int): maximum length of the output tensor.
        NOTE: max_length includes the end-of-sentence character that will be added
                to the tensor.
                Keep in mind that the length of the tensor is always 1 + the length
                of the original line of characters.
        data_lines (list): list of the sentences to group into batches.
        line_to_tensor (function, optional): function that converts line to tensor. Defaults to line_to_tensor.
        shuffle (bool, optional): True if the generator should generate random batches of data. Defaults to True.

    Yields:
        tuple: two copies of the batch (jax.interpreters.xla.DeviceArray) and mask (jax.interpreters.xla.DeviceArray).
        NOTE: jax.interpreters.xla.DeviceArray is trax's version of numpy.ndarray
    """
    index = 0
    cur_batch = []
    num_lines = len(data_lines)

    # create an array with the indexes of data_lines that can be shuffled
    lines_index = [*range(num_lines)]

    if shuffle:
        rnd.shuffle(lines_index)

    while True:
        if index >= num_lines:
            index = 0
            if shuffle:
                rnd.shuffle(lines_index)

        line = data_lines[lines_index[index]]

        if len(line) < max_length:
            cur_batch.append(line)

        index += 1

        if len(cur_batch) == batch_size:
            batch = []
            mask = []

            for li in cur_batch:
                tensor = line_to_tensor(li)

                # Create a list of zeros to represent the padding
                # so that the tensor plus padding will have length `max_length`
                pad = [0] * (max_length - len(li) -1)

                # combine the tensor plus pad
                tensor_pad = tensor + pad
                batch.append(tensor_pad)

                #Mask to detect the padded elements
                example_mask = [1] * len(tensor) + pad
                mask.append(example_mask)

            # convert the batch (data type list) to a trax's numpy array
            batch_np_arr = np.array(batch)
            mask_np_arr = np.array(mask)

            # Yield two copies of the batch and mask.
            yield batch_np_arr, batch_np_arr, mask_np_arr

            # reset the current batch to an empty list
            cur_batch = []

def GRULM(vocab_size=256, d_model=512, n_layers=2, mode='train'):
    """
    Returns a GRU language model.
    Args:
        vocab_size (int, optional): Size of the vocabulary. Defaults to 256.
        d_model (int, optional): Depth of embedding (n_units in the GRU cell). Defaults to 512.
        n_layers (int, optional): Number of GRU layers. Defaults to 2.
        mode (str, optional): 'train', 'eval' or 'predict', predict mode is for fast inference. Defaults to "train".

    Returns:
        trax.layers.combinators.Serial: A GRU language model as a layer that maps from a tensor of tokens to activations over a vocab set.
    """
    model = tl.Serial(
      tl.ShiftRight(mode = 'train'),
      tl.Embedding(vocab_size = vocab_size,d_feature = d_model),
      [tl.GRU(d_model) for _ in range(n_layers)],
      tl.Dense(vocab_size),
      tl.LogSoftmax()
    )
    return model

def n_used_lines(lines, max_length):
    '''
    Args:
    lines: all lines of text an array of lines
    max_length - max_length of a line in order to be considered an int
    Return:
    n_lines -number of efective examples
    '''

    n_lines = 0
    print(len(lines))
    for l in lines:
        if len(l) <= max_length:
            n_lines += 1
    return n_lines


def train_model(model, data_generator, batch_size=32, max_length=64, lines=lines, eval_lines=eval_lines, n_steps=1, output_dir='model/'):
    """Function that trains the model

    Args:
        model (trax.layers.combinators.Serial): GRU model.
        data_generator (function): Data generator function.
        batch_size (int, optional): Number of lines per batch. Defaults to 32.
        max_length (int, optional): Maximum length allowed for a line to be processed. Defaults to 64.
        lines (list, optional): List of lines to use for training. Defaults to lines.
        eval_lines (list, optional): List of lines to use for evaluation. Defaults to eval_lines.
        n_steps (int, optional): Number of steps to train. Defaults to 1.
        output_dir (str, optional): Relative path of directory to save model. Defaults to "model/".

    Returns:
        trax.supervised.training.Loop: Training loop for the model.
    """

    bare_train_generator = data_generator(batch_size, max_length, lines)
    infinite_train_generator = itertools.cycle(data_generator(batch_size, max_length, lines))

    bare_eval_generator = data_generator(batch_size, max_length, eval_lines)
    infinite_eval_generator = itertools.cycle(data_generator(batch_size, max_length, eval_lines))

    train_task = training.TrainTask(
        labeled_data=infinite_train_generator,
        loss_layer=tl.CrossEntropyLoss(),
        optimizer=trax.optimizers.Adam(0.0005)
    )

    eval_task = training.EvalTask(
        labeled_data=infinite_eval_generator,
        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
        n_eval_batches=3     # For better evaluation accuracy in reasonable time
    )

    training_loop = training.Loop(model,
                                  train_task,
                                  eval_task=eval_task,
                                  output_dir=output_dir)

    training_loop.run(n_steps=n_steps)

    return training_loop

def test_model(preds, target):
    """
    Function to test the model, by calculating the preplexity of it.

    Args:
        preds (jax.interpreters.xla.DeviceArray): Predictions of a list of batches of tensors corresponding to lines of text.
        target (jax.interpreters.xla.DeviceArray): Actual list of batches of tensors corresponding to lines of text.

    Returns:
        float: log_perplexity of the model.
    """
    print(preds.shape)
    print(target.shape)
    total_log_ppx = np.sum(tl.one_hot(target,preds.shape[-1]) * preds, axis= -1)
    print(total_log_ppx.shape)
    non_pad = 1.0 - np.equal(target, 0)
    ppx = total_log_ppx * non_pad

    log_ppx = np.sum(ppx) / np.sum(non_pad)

    return -log_ppx



# set random seed
trax.supervised.trainer_lib.init_random_number_generators(32)
rnd.seed(32)

batch_size = 32
max_length = 83

dirname = 'data/'
lines = []
for filename in os.listdir(dirname):
    with open(os.path.join(dirname, filename)) as files:
        for line in files:
            pure_line = line.strip()
            if pure_line:       # if pure_line is not the empty string,
                lines.append(pure_line)

n_lines = len(lines)

for i, line in enumerate(lines):
    lines[i] = line.lower()

eval_lines = lines[-1000:] # Create a holdout validation set
lines = lines[:-1000]      # Leave the rest for training

model = GRULM()
print(model)
num_used_lines = n_used_lines(lines, max_length)
steps_per_epoch = int(num_used_lines/batch_size)
from trax.supervised import training
training_loop = train_model(GRULM(), data_generator, n_steps = 10)

# Testing
model = GRULM()                          #Define the mode
model.init_from_file('model.pkl.gz')     #Load pretrained parameters
batch = next(data_generator(batch_size, max_length, lines, shuffle=False))
preds = model(batch[0])
log_ppx = test_model(preds, batch[1])
print('The log perplexity and perplexity of your model are respectively', log_ppx, np.exp(log_ppx))

def gumbel_sample(log_probs, temperature=1.0):
    """Gumbel sampling from a categorical distribution."""
    u = numpy.random.uniform(low=1e-6, high=1.0 - 1e-6, size=log_probs.shape)
    g = -np.log(-np.log(u))
    return np.argmax(log_probs + g * temperature, axis=-1)

def predict(num_chars, prefix):
    inp = [ord(c) for c in prefix]
    result = [c for c in prefix]
    max_len = len(prefix) + num_chars
    for _ in range(num_chars):
        cur_inp = np.array(inp + [0] * (max_len - len(inp)))
        outp = model(cur_inp[None, :])  # Add batch dim.
        next_char = gumbel_sample(outp[0, len(inp)])
        inp += [int(next_char)]

        if inp[-1] == 1:
            break  # EOS
        result.append(chr(int(next_char)))

    return "".join(result)

print(predict(32, ""))
print(predict(32, ""))
print(predict(32, ""))
print(predict(32, ""))
print(predict(32, ""))

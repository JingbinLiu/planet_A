# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np
import tensorflow as tf

from planet.tools import nested
from planet.tools import shape


def overshooting(
    cell, target, embedded, prev_action, length, amount, ignore_input=False):
  """Perform open loop rollouts from the posteriors at every step.

  First, we apply the encoder to embed raw inputs and apply the model to obtain
  posterior states for every time step. Then, we perform `amount` long open
  loop rollouts from these posteriors.

  Note that the actions should be those leading to the current time step. So
  under common convention, it contains the last actions while observations are
  the current ones.

  Input:

    target, embedded:
      [A B C D E F] [A B C D E  ]

    prev_action:
      [0 A B C D E] [0 A B C D  ]

    length:
      [6 5]

    amount:
      3

  Output:

    prior, posterior, target:
      [A B C D E F] [A B C D E  ]   o---- chunk_length-->
      [B C D E F  ] [B C D E    ]   |
      [C D E F    ] [C D E      ]   |
      [D E F      ] [D E        ]   amount
                                    |
    mask:
      [1 1 1 1 1 1] [1 1 1 1 1 0]
      [1 1 1 1 1 0] [1 1 1 1 0 0]
      [1 1 1 1 0 0] [1 1 1 0 0 0]
      [1 1 1 0 0 0] [1 1 0 0 0 0]

  """
  # Closed loop unroll to get posterior states, which are the starting points
  # for open loop unrolls. We don't need the last time step, since we have no
  # targets for unrolls from it.
  use_obs = tf.ones(tf.shape(
      nested.flatten(embedded)[0][:, :, :1])[:3], tf.bool)   # shape(40,50,1024) --> shape(40,50,1)
  use_obs = tf.cond(
      tf.convert_to_tensor(ignore_input),
      lambda: tf.zeros_like(use_obs, tf.bool),
      lambda: use_obs)
  (prior, posterior), _ = tf.nn.dynamic_rnn(
      cell, (embedded, prev_action, use_obs), length, dtype=tf.float32,    # cell, inputs:shape(batchsize,max_time,?):(40,50,?), sequence_length:shape(batchsize,):(40,)
      swap_memory=True)                                                    # calculate posterior: q(s_t−d |o ≤t−d ,a <t−d )

  # Arrange inputs for every iteration in the open loop unroll. Every loop
  # iteration below corresponds to one row in the docstring illustration.
  max_length = shape.shape(nested.flatten(embedded)[0])[1]                 # max_length = 50
  first_output = {
      'observ': embedded,           # shape(40,50,1024)
      'prev_action': prev_action,   # shape(40,50,2)
      'posterior': posterior,       # {'mean':shape(40,50,30), ...}
      'target': target,             # {'reward': shape(40,50), ...}
      'mask': tf.sequence_mask(length, max_length, tf.int32),   # shape(40,50)
  }
  progress_fn = lambda tensor: tf.concat([tensor[:, 1:], 0 * tensor[:, :1]], 1)   # on the 1st dimension(episode_length): (a[1] ,a[2], ..., 0*a[0])
  other_outputs = tf.scan(                                             # other_outputs: { 'observ': shape(50(amount),40(batchsize),50(episode_length),1024),...}
      lambda past_output, _: nested.map(progress_fn, past_output),     # past_output = progress_fn(past_output), initial past_output is first_output.
      tf.range(amount), first_output)                                  # first_output: { 'observ': shape(40,50,1024),...};
  sequences = nested.map(
      lambda lhs, rhs: tf.concat([lhs[None], rhs], 0),     # first_output[None]
      first_output, other_outputs)                         # sequences: { 'observ': shape(51,40,50,1024),...};

  # Merge batch and time dimensions of steps to compute unrolls from every
  # time step as one batch. The time dimension becomes the number of
  # overshooting distances.
  sequences = nested.map(
      lambda tensor: _merge_dims(tensor, [1, 2]),          # sequences: { 'observ': shape(51,2000,1024),...}
      sequences)
  sequences = nested.map(
      lambda tensor: tf.transpose(
          tensor, [1, 0] + list(range(2, tensor.shape.ndims))),   # [1,0]+[2]
      sequences)                                           # sequences: { 'observ': shape(2000,51,1024),...}
  merged_length = tf.reduce_sum(sequences['mask'], 1)      # shape(2000,51) --> shape(2000,)

  # Mask out padding frames; unnecessary if the input is already masked.
  sequences = nested.map(
      lambda tensor: tensor * tf.cast(
          _pad_dims(sequences['mask'], tensor.shape.ndims),   # sequences['mask']: shape(2000,51) --> shape(2000,51,1); sequences['observ']: shape(2000,51,1024)
          tensor.dtype),                                      # shape(2000,51,1024)*shape(2000,51,1)
      sequences)

  # Compute open loop rollouts.
  use_obs = tf.zeros(tf.shape(sequences['mask']), tf.bool)[..., None]
  prev_state = nested.map(
      lambda tensor: tf.concat([0 * tensor[:, :1], tensor[:, :-1]], 1),   # {'mean': shape(40,50,30), ...}; on the 1st dimension(episode_length): (s1, s2, ..., s50) --> (0, s1, s2, ..., s49)
      posterior)
  prev_state = nested.map(
      lambda tensor: _merge_dims(tensor, [0, 1]), prev_state)             # {'mean': shape(2000,30), ...}
  (priors, _), _ = tf.nn.dynamic_rnn(
      cell, (sequences['observ'], sequences['prev_action'], use_obs),
      merged_length,
      prev_state)    # initial_state = prev_state.    # calculate prior: p(s_t−1 |s_t−d ,a_t−d−1:t−2 )

  # Restore batch dimension.
  target, prior, posterior, mask = nested.map(
      functools.partial(_restore_batch_dim, batch_size=shape.shape(length)[0]),
      (sequences['target'], priors, sequences['posterior'], sequences['mask']))

  mask = tf.cast(mask, tf.bool)
  return target, prior, posterior, mask


def _merge_dims(tensor, dims):                            # tensor: shape(51,40,50,1024)
  """Flatten consecutive axes of a tensor trying to preserve static shapes."""
  if isinstance(tensor, (list, tuple, dict)):
    return nested.map(tensor, lambda x: _merge_dims(x, dims))
  tensor = tf.convert_to_tensor(tensor)
  if (np.array(dims) - min(dims) != np.arange(len(dims))).all():
    raise ValueError('Dimensions to merge must all follow each other.')
  start, end = dims[0], dims[-1]                          # start, end = 1, 2
  output = tf.reshape(tensor, tf.concat([                 # tf.reshape(tensor, shape[51,2000,1024])
      tf.shape(tensor)[:start],                           # [51,]  # tf.shape(tensor):(51,40,50,1024)
      [tf.reduce_prod(tf.shape(tensor)[start: end + 1])], # [40*50,]
      tf.shape(tensor)[end + 1:]], axis=0))               # [1024,]
  merged = tensor.shape[start: end + 1].as_list()         # [40,50]
  output.set_shape(
      tensor.shape[:start].as_list() +                    # [51]+
      [None if None in merged else np.prod(merged)] +     # [2000]+
      tensor.shape[end + 1:].as_list())                   # [1024]
  return output


def _pad_dims(tensor, rank):
  """Append empty dimensions to the tensor until it is of the given rank."""
  for _ in range(rank - tensor.shape.ndims):
    tensor = tensor[..., None]
  return tensor


def _restore_batch_dim(tensor, batch_size):
  """Split batch dimension out of the first dimension of a tensor."""
  initial = shape.shape(tensor)
  desired = [batch_size, initial[0] // batch_size] + initial[1:]
  return tf.reshape(tensor, desired)

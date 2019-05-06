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

from tensorflow_probability import distributions as tfd
import tensorflow as tf

from planet.control import discounted_return
from planet import tools

def delta_degree(x):
  return tf.where(tf.abs(x) < 180 , x, x-tf.sign(x)*360)

# compute angular clue
costn0 = lambda: tf.constant(0.0)    # [throttle, steer(l-,r+)]
costn1 = lambda: tf.constant(1.0)

def cross_entropy_method(
    cell, objective_fn, state, info_cmd, obs_shape, action_shape, horizon,
    amount=1000, topk=100, iterations=10, discount=0.99,
    min_action=-1, max_action=1):       # state,info_cmd: shape(num_envs,4): next_command_id, goal_heading_degree, current_heading_degree,dist_to_intersection
  obs_shape, action_shape = tuple(obs_shape), tuple(action_shape)
  original_batch = tools.shape(tools.nested.flatten(state)[0])[0]    # original_batch: num_envs
  initial_state = tools.nested.map(lambda tensor: tf.tile(
      tensor, [amount] + [1] * (tensor.shape.ndims - 1)), state)
  extended_batch = tools.shape(tools.nested.flatten(initial_state)[0])[0]
  use_obs = tf.zeros([extended_batch, horizon, 1], tf.bool)
  obs = tf.zeros((extended_batch, horizon) + obs_shape)
  length = tf.ones([extended_batch], dtype=tf.int32) * horizon

  # info_cmd components
  info_cmd = tf.squeeze(info_cmd)   # shape(3,)
  cmd_id, goal_heading_degree, current_heading_degree, dist_to_intersection = info_cmd[0],info_cmd[1],info_cmd[2],info_cmd[3]

  def iteration(mean_and_stddev, _):
    mean, stddev = mean_and_stddev
    # Sample action proposals from belief.
    normal = tf.random_normal((original_batch, amount, horizon) + action_shape)
    action = normal * stddev[:, None] + mean[:, None]
    action = tf.clip_by_value(action, min_action, max_action)
    # Evaluate proposal actions.
    action = tf.reshape(
        action, (extended_batch, horizon) + action_shape)
    (_, state), _ = tf.nn.dynamic_rnn(
        cell, (0 * obs, action, use_obs), initial_state=initial_state)

    # objectives
    objectives = objective_fn(state)   # shape: [shape(1000,12), shape(1000,12)]
    reward = objectives['reward']
    angular_speed = objectives['angular_speed_degree']
    forward_speed = objectives['forward_speed']


    ##################    #1. define reward for planning
    return_ = discounted_return.discounted_return(reward, length, discount)[:, 0]           # shape: (1000,)
    return_ = tf.reshape(return_, (original_batch, amount))                                 # shape: (1, 1000)

    # threshold_degree = tf.where(dist_to_intersection<10, 9.0*(10 - dist_to_intersection), 0)
    threshold_degree = tf.where(dist_to_intersection < 9, 9 * (9 - dist_to_intersection), 0)
    angular_turn_ = discounted_return.discounted_return(angular_speed, length, 1.0)[:, 0]   # shape: (1000,)
    # angular_turn_abs = discounted_return.discounted_return(-tf.abs(angular_speed), length, 1.0)[:, 0]
    # angular_turn_relative = tf.reduce_sum(-tf.abs(angular_speed[...,1:]-angular_speed[...,:-1]),axis=-1)
    heading_loss = - tf.abs(delta_degree(goal_heading_degree - (current_heading_degree + angular_turn_)))* \
                   tf.case({ tf.equal(cmd_id,3):costn1, tf.equal(cmd_id,2):costn1, tf.equal(cmd_id,1):costn1}, default=costn0)
    heading_loss_weighted = heading_loss * tf.where(heading_loss>threshold_degree-90, tf.ones((amount,))*0.3, tf.ones((amount,))*1000.0) # + 0.3*angular_turn_relative # + 0.1*angular_turn_abs
    return_heading = tf.reshape(heading_loss_weighted, (original_batch, amount))

    total_return = return_+ return_heading # /90.0*12*4


    # #################    #2. define reward for planning
    # return_ = discounted_return.discounted_return(
    #     reward, length, discount)[:, 0]
    # total_return = tf.reshape(return_, (original_batch, amount))


    # Re-fit belief to the best ones.
    _, indices = tf.nn.top_k(total_return, topk, sorted=False)
    indices += tf.range(original_batch)[:, None] * amount
    best_actions = tf.gather(action, indices)
    mean, variance = tf.nn.moments(best_actions, 1)
    stddev = tf.sqrt(variance + 1e-6)
    return mean, stddev


  '''COMMAND_ORDINAL = {
    "REACH_GOAL": 0,
    "GO_STRAIGHT": 1,
    "TURN_RIGHT": 2,
    "TURN_LEFT": 3,
    "LANE_FOLLOW": 4 }
  '''
  # compute action_bias
  f_0 = lambda: tf.constant([0.0, 0.0])    # [throttle, steer(l-,r+)]
  f_1eft = lambda: tf.constant([0.0, -0.5])
  f_right = lambda: tf.constant([0.0, 0.5])

  pred_func = { tf.equal(cmd_id,3):f_1eft, tf.equal(cmd_id,2):f_right }
  action_bias = tf.case(pred_func, default=f_0)

  # # compute angular clue
  # angular_f_0 = lambda: tf.constant(0.0)    # [throttle, steer(l-,r+)]
  # angular_f_1eft = lambda: tf.constant(-3.0)
  # angular_f_right = lambda: tf.constant(3.0)
  #
  # angular_pred_func = { tf.equal(cmd_id,3):angular_f_1eft, tf.equal(cmd_id,2):angular_f_right, tf.equal(cmd_id,1):angular_f_0 }
  # angular_clue = tf.case(angular_pred_func, default=angular_f_0)


  mean = tf.zeros((original_batch, horizon) + action_shape) # + action_bias
  stddev = tf.ones((original_batch, horizon) + action_shape)

  mean, stddev = tf.scan(
      iteration, tf.range(iterations), (mean, stddev), back_prop=False)
  mean, stddev = mean[-1], stddev[-1]  # Select belief at last iterations.
  return mean

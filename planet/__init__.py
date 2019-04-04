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



# ================ schemes =================

# TASK_NAME = "{tasks: [carla]}"
# IMG_SIZE = (32, 32)
# EPISODE_LEN = 100
# REPEATE = 2
# BATCHSIZE = 50
# REWARD_FUNC = 'custom2'
# USE_DEPTH = False
# LOGDIR = './log_32_8bit_40batch_a'


# TASK_NAME = "{tasks: [carla]}"
# IMG_SIZE = (64, 64)
# EPISODE_LEN = 100
# REPEATE = 2
# BATCHSIZE = 50
# REWARD_FUNC = 'custom2'
# USE_DEPTH = False
# LOGDIR = './log_64_8bits_carla_a'


# TASK_NAME = "{tasks: [carla]}"
# IMG_SIZE = (128, 128)
# EPISODE_LEN = 100
# REPEATE = 2
# BATCHSIZE = 40
# REWARD_FUNC = 'custom2'
# USE_DEPTH = False
# LOGDIR = './log_128_8bit_40batch_a'


# TASK_NAME = "{tasks: [carla]}"
# IMG_SIZE = (64, 64)
# EPISODE_LEN = 130
# REPEATE = 1
# BATCHSIZE = 50
# REWARD_FUNC = 'custom3'
# USE_SENSOR = 'use_rgb'
# NUM_CHANNELS = 3
# LOGDIR = './log_64_8bit_50_1_rgb_I'


# TASK_NAME = "{tasks: [carla]}"
# IMG_SIZE = (64, 64)
# EPISODE_LEN = 100
# REPEATE = 1
# BATCHSIZE = 50
# REWARD_FUNC = 'custom3'
# USE_SENSOR = 'use_rgb'
# NUM_CHANNELS = 3
# LOGDIR = './log_64_8bit_50_1_rgb_II' # standard position for rgb camera.


# TASK_NAME = "{tasks: [carla]}"
# IMG_SIZE = (64, 64)
# EPISODE_LEN = 100
# REPEATE = 1
# BATCHSIZE = 50
# REWARD_FUNC = 'custom_depth'
# USE_SENSOR = 'use_logdepth'
# NUM_CHANNELS = 1
# LOGDIR = './log_64_8bit_50_1_logdepth_I'

# TASK_NAME = "{tasks: [carla]}"
# IMG_SIZE = (64, 64)
# EPISODE_LEN = 100
# REPEATE = 1
# BATCHSIZE = 50
# REWARD_FUNC = 'custom_depth'
# USE_SENSOR = 'use_depth'
# NUM_CHANNELS = 1
# LOGDIR = './log_64_8bit_50_1_depth_II'

# TASK_NAME = "{tasks: [carla]}"
# IMG_SIZE = (64, 64)
# EPISODE_LEN = 140
# REPEATE = 1
# BATCHSIZE = 50
# REWARD_FUNC = 'custom3'
# USE_SENSOR = 'use_semantic'
# NUM_CHANNELS = 1
# LOGDIR = './log_64_8bit_50_1_semantic_I'

# TASK_NAME = "{tasks: [carla]}"
# IMG_SIZE = (64, 64)
# EPISODE_LEN = 100
# REPEATE = 1
# BATCHSIZE = 50
# REWARD_FUNC = 'custom3'
# USE_SENSOR = 'use_2rgb'
# NUM_CHANNELS = 6
# LOGDIR = './log_64_8bit_50_1_2rgb_I'

# TASK_NAME = "{tasks: [carla]}"
# IMG_SIZE = (64, 64)
# EPISODE_LEN = 100
# REPEATE = 1
# BATCHSIZE = 50
# REWARD_FUNC = 'custom3'
# USE_SENSOR = 'use_rgb'
# NUM_CHANNELS = 3
# LOGDIR = './carla_debug_log'


TASK_NAME = "{tasks: [carla]}"
IMG_SIZE = (64, 64)
EPISODE_LEN = 100
REPEATE = 1
BATCHSIZE = 40
REWARD_FUNC = 'custom3'
USE_SENSOR = 'use_rgb'
NUM_CHANNELS = 3
LOGDIR = '~/Data/planet/carla_64'


# TASK_NAME = "{tasks: [breakout]}"
# IMG_SIZE = (128, 128)
# EPISODE_LEN = 100
# REPEATE = 1
# BATCHSIZE = 50
# REWARD_FUNC = 'custom3'
# USE_SENSOR = 'use_rgb'
# NUM_CHANNELS = 3
# LOGDIR = '~/Data/planet/breakout_128_h3'

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


# TASK_NAME = "{tasks: [carla]}"
# IMG_SIZE = (64, 64)
# EPISODE_LEN, COLLECT_EPISODE = 100, 0.0
# RREPEATE, NUM_SEED = 1, 5
# BATCHSIZE, CHUNK_LEN = 50, 50
# REWARD_FUNC = 'custom3'
# USE_SENSOR = 'use_rgb'
# NUM_CHANNELS = 3
# SCENARIO = 'TOWN2_ONE_CURVE_STRAIGHT_NAV' #  'TOWN2_WEATHER_NPC'  #
# LOGDIR = '~/Data/planet/carla_64_reward0.0'



#############################################

# TASK_NAME = "{tasks: [carla]}"; ENABLE_EXPERT = False  # use ENABLE_EXPERT to collect expert data.
# IMG_SIZE = (64, 64)
# H_SIZE, S_SIZE = 400, 60
# EPISODE_LEN, COLLECT_EPISODE = 100, 300000.0   # collect data when accumulative reward < COLLECT_EPISODE.
# REPEATE, NUM_SEED = 1, 300
# BATCHSIZE, CHUNK_LEN = 50, 50
# REWARD_FUNC = 'custom4'
# USE_SENSOR = 'use_rgb'
# NUM_CHANNELS = 3
# SCENARIO = 'TOWN2_ONE_CURVE_STRAIGHT_NAV'
# LOGDIR = '~/Data/planet/auto_carla_h-b_400-60'




##########################################
#
# TASK_NAME = "{tasks: [carla]}"; ENABLE_EXPERT = False
# IMG_SIZE = (64, 64)
# H_SIZE, S_SIZE = 200, 30
# EPISODE_LEN, COLLECT_EPISODE = 600, 300000.0
# REPEATE, NUM_SEED = 1, 5
# BATCHSIZE, CHUNK_LEN = 50, 50
# REWARD_FUNC = 'custom3'
# USE_SENSOR = 'use_rgb'
# NUM_CHANNELS = 3
# SCENARIO = 'TOWN2_ONE_CURVE_STRAIGHT_NAV' # 'TOWN2_NPC'  #  'TOWN2_WEATHER_NPC'  #
# LOGDIR = '~/Data/planet/carla_64_200'

##########################################

# TASK_NAME = "{tasks: [carla]}"; ENABLE_EXPERT = False
# IMG_SIZE = (64, 64)
# H_SIZE, S_SIZE = 200, 30
# EPISODE_LEN, COLLECT_EPISODE = 600, 300000.0
# REPEATE, NUM_SEED = 1, 5
# BATCHSIZE, CHUNK_LEN = 50, 50
# REWARD_FUNC = 'custom3'
# USE_SENSOR = 'use_rgb'
# NUM_CHANNELS = 3
# SCENARIO = 'TOWN2_ONE_CURVE_STRAIGHT_NAV' # 'TOWN2_NPC'  #  'TOWN2_WEATHER_NPC'  #
# LOGDIR = '~/Data/planet/carla_64_400'

#############################################
#
# TASK_NAME = "{tasks: [carla]}"; ENABLE_EXPERT = False
# IMG_SIZE = (64, 64)
# H_SIZE, S_SIZE = 400, 60
# EPISODE_LEN, COLLECT_EPISODE = 600, 300000.0
# REPEATE, NUM_SEED = 1, 5
# BATCHSIZE, CHUNK_LEN = 50, 50
# REWARD_FUNC = 'custom3'
# USE_SENSOR = 'use_rgb'
# NUM_CHANNELS = 3
# SCENARIO = 'TOWN2_ONE_CURVE_STRAIGHT_NAV' # 'TOWN2_NPC'  #  'TOWN2_WEATHER_NPC'  #
# LOGDIR = '~/Data/planet/carla_64_200_400-60'


##########################################

TASK_NAME = "{tasks: [carla]}"; ENABLE_EXPERT = False
IMG_SIZE = (128, 128)
H_SIZE, S_SIZE = 200, 30
EPISODE_LEN, COLLECT_EPISODE = 600, 300000.0
REPEATE, NUM_SEED = 1, 1
BATCHSIZE, CHUNK_LEN = 50, 50
REWARD_FUNC = 'custom3'
USE_SENSOR = 'use_rgb'
NUM_CHANNELS = 3
SCENARIO = 'TOWN2_ONE_CURVE_STRAIGHT_NAV' # 'TOWN2_NPC'  #  'TOWN2_WEATHER_NPC'  #
LOGDIR = '~/Data/planet/carla_test128_angular'


# TASK_NAME = "{tasks: [carla]}"; ENABLE_EXPERT = False
# IMG_SIZE = (64, 64)
# H_SIZE, S_SIZE = 200, 30
# EPISODE_LEN, COLLECT_EPISODE = 300, 300000.0
# REPEATE, NUM_SEED = 1, 1
# BATCHSIZE, CHUNK_LEN = 50, 50
# REWARD_FUNC = 'custom3'
# USE_SENSOR = 'use_rgb'
# NUM_CHANNELS = 3
# SCENARIO = 'TOWN2_ONE_CURVE_STRAIGHT_NAV' # 'TOWN2_NPC'  #  'TOWN2_WEATHER_NPC'  #
# LOGDIR = '~/Data/planet/carla_64_angular_1.5'


#############################################


# TASK_NAME = "{tasks: [carla]}"; ENABLE_EXPERT = False
# IMG_SIZE = (64, 64)
# H_SIZE, S_SIZE = 200, 30
# EPISODE_LEN, COLLECT_EPISODE = 100, 300000.0
# REPEATE, NUM_SEED = 1, 1
# BATCHSIZE, CHUNK_LEN = 5, 5
# REWARD_FUNC = 'custom3'
# USE_SENSOR = 'use_rgb'
# NUM_CHANNELS = 3
# SCENARIO = 'TOWN2_ONE_CURVE_STRAIGHT_NAV' # 'TOWN2_NPC'  #  'TOWN2_WEATHER_NPC'  #
# LOGDIR = '~/Data/planet/carla_debug'
"""OpenAI gym environment for Carla. Run this file for a demo."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import atexit
import cv2
import os
import json
import random
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
try:
    import scipy.misc
except Exception:
    pass

import gym
from gym.spaces import Box, Discrete, Tuple

from planet import REWARD_FUNC, IMG_SIZE,  USE_SENSOR, SCENARIO

exec('from .scenarios import '+ SCENARIO + ' as SCENARIO')

# from .scenarios import TOWN2_NPC, TOWN2_WEATHER, TOWN2_WEATHER_NPC,\
#     LANE_KEEP, TOWN2_ALL, TOWN2_ONE_CURVE, TOWN2_ONE_CURVE_0, TOWN2_ONE_CURVE_STRAIGHT_NAV,TOWN2_STRAIGHT_DYNAMIC_0, TOWN2_STRAIGHT_0

# Set this where you want to save image outputs (or empty string to disable)
CARLA_OUT_PATH = os.environ.get("CARLA_OUT", os.path.expanduser("~/carla_out"))
if CARLA_OUT_PATH and not os.path.exists(CARLA_OUT_PATH):
    os.makedirs(CARLA_OUT_PATH)

# Set this to the path of your Carla binary
SERVER_BINARY = os.environ.get("CARLA_SERVER",
                               os.path.expanduser("/data/carla8/CarlaUE4.sh"))     # the carla8 engine

assert os.path.exists(SERVER_BINARY)
if "CARLA_PY_PATH" in os.environ:
    sys.path.append(os.path.expanduser(os.environ["CARLA_PY_PATH"]))
else:
    # TODO(ekl) switch this to the binary path once the planner is in master
    sys.path.append(os.path.expanduser("/data/carla8/PythonClient/"))             # the carla8 python API

try:
    from carla.client import CarlaClient, VehicleControl
    from carla.sensor import Camera
    from carla.settings import CarlaSettings
    from carla.planner.planner import Planner, REACH_GOAL, GO_STRAIGHT, \
        TURN_RIGHT, TURN_LEFT, LANE_FOLLOW
except Exception as e:
    print("Failed to import Carla python libs, try setting $CARLA_PY_PATH")
    raise e

# Carla planner commands
COMMANDS_ENUM = {
    REACH_GOAL: "REACH_GOAL",
    GO_STRAIGHT: "GO_STRAIGHT",
    TURN_RIGHT: "TURN_RIGHT",
    TURN_LEFT: "TURN_LEFT",
    LANE_FOLLOW: "LANE_FOLLOW",
}

# Mapping from string repr to one-hot encoding index to feed to the model
COMMAND_ORDINAL = {
    "REACH_GOAL": 0,
    "GO_STRAIGHT": 1,
    "TURN_RIGHT": 2,
    "TURN_LEFT": 3,
    "LANE_FOLLOW": 4,
}

# Number of retries if the server doesn't respond
RETRIES_ON_ERROR = 5

# Dummy Z coordinate to use when we only care about (x, y)
GROUND_Z = 0.22

# Default environment configuration
ENV_CONFIG = {
    "log_images": False,  # log images in _read_observation().
    "convert_images_to_video": False,  # convert log_images to videos. when "verbose" is True.
    "verbose": False,    # print measurement information; write out measurement json file.

    "enable_planner": True,
    "framestack": 1,  # note: only [1, 2] currently supported
    "early_terminate_on_collision": True,
    "reward_function": REWARD_FUNC,
    "render_x_res": 400, #800,
    "render_y_res": 175, #600,
    "x_res": 64,  # cv2.resize()
    "y_res": 64,  # cv2.resize()
    "server_map": "/Game/Maps/Town02",
    "scenarios": SCENARIO,
    "use_sensor": USE_SENSOR,
    "discrete_actions": False,
    "squash_action_logits": False,
}

DISCRETE_ACTIONS = {
    # coast
    0: [0.0, 0.0],
    # turn left
    1: [0.0, -0.5],
    # turn right
    2: [0.0, 0.5],
    # forward
    3: [1.0, 0.0],
    # brake
    4: [-0.5, 0.0],
    # forward left
    5: [1.0, -0.5],
    # forward right
    6: [1.0, 0.5],
    # brake left
    7: [-0.5, -0.5],
    # brake right
    8: [-0.5, 0.5],
}


# timeout decorator
def set_timeout(seconds):
    def wrap(func):
        def handle(signum, frame):  # 收到信号 SIGALRM 后的回调函数，第一个参数是信号的数字，第二个参数是the interrupted stack frame.
            raise RuntimeError
        def to_do(*args, **kwargs):
            signal.signal(signal.SIGALRM, handle)  # 设置信号和回调函数
            signal.alarm(seconds)  # 设置 timeout 秒的闹钟
            # print('start alarm signal.')
            r = func(*args, **kwargs)
            # print('close alarm signal.')
            signal.alarm(0)  # 关闭闹钟
            return r
        return to_do
    return wrap



live_carla_processes = set()  # Carla Server


def cleanup():
    print("Killing live carla processes", live_carla_processes)
    for pgid in live_carla_processes:
        os.killpg(pgid, signal.SIGKILL)


atexit.register(cleanup)


class CarlaEnv(gym.Env):
    def __init__(self, config=ENV_CONFIG, enable_autopilot = False):
        self.enable_autopilot = enable_autopilot
        self.config = config
        self.config["x_res"], self.config["y_res"] = IMG_SIZE
        self.city = self.config["server_map"].split("/")[-1]
        if self.config["enable_planner"]:
            self.planner = Planner(self.city)
            self.intersections_node = self.planner._city_track._map.get_intersection_nodes()
            self.intersections_pos = np.array([self.planner._city_track._map.convert_to_world(intersection_node) for intersection_node in self.intersections_node])
            self.pre_intersection = np.array([0.0,0.0])

            # # Cartesian coordinates
            self.headings = np.array([[1,0],[-1,0],[0,1],[0,-1]])
            # self.lrs_matrix = {"GO_STRAIGHT": np.array([[1,0],[0,1]]),\
            #                    "TURN_RIGHT": np.array([[0,-1],[1,0]]),\
            #                    "TURN_LEFT": np.array([[0,1],[-1,0]])}
            # self.goal_heading = np.array([0.0,0.0])
            # self.current_heading = None
            # self.pre_heading = None
            # self.angular_speed = None

            # Angular degree
            self.headings_degree = np.array([0.0, 180.0, 90.0, -90.0])  # one-one mapping to self.headings
            self.lrs_degree = {"GO_STRAIGHT": 0.0,  "TURN_LEFT": -90.0, "TURN_RIGHT": 90.0}
            self.goal_heading_degree = 0.0
            self.current_heading_degree = None
            self.pre_heading_degree = None
            self.angular_speed_degree = np.array(0.0)

        # The Action Space
        if config["discrete_actions"]:
            self.action_space = Discrete(len(DISCRETE_ACTIONS))  # It will be transformed to continuous 2D action.
        else:
            self.action_space = Box(-1.0, 1.0, shape=(2, ), dtype=np.float32)   # 2D action.

        if config["use_sensor"] == 'use_semantic':
            image_space = Box(
                0.0,
                255.0,
                shape=(config["y_res"], config["x_res"],
                       1 * config["framestack"]),
                dtype=np.float32)
        elif config["use_sensor"] in ['use_depth','use_logdepth']:
            image_space = Box(
                0.0,
                255.0,
                shape=(config["y_res"], config["x_res"],
                       1 * config["framestack"]),
                dtype=np.float32)
        elif config["use_sensor"] == 'use_rgb':
            image_space = Box(
                0,
                255,
                shape=(config["y_res"], config["x_res"],
                       3 * config["framestack"]),
                dtype=np.uint8)
        elif config["use_sensor"] == 'use_2rgb':
            image_space = Box(
                0,
                255,
                shape=(config["y_res"], config["x_res"],
                       2 * 3 * config["framestack"]),
                dtype=np.uint8)

        # The Observation Space
        self.observation_space = Tuple(
            [
                image_space,
                Discrete(len(COMMANDS_ENUM)),  # next_command
                Box(-128.0, 128.0, shape=(2, ), dtype=np.float32)  # forward_speed, dist to goal
            ])

        # TODO(ekl) this isn't really a proper gym spec
        self._spec = lambda: None
        self._spec.id = "Carla-v0"

        self.server_port = None
        self.server_process = None
        self.client = None
        self.num_steps = 0
        self.total_reward = 0
        self.prev_measurement = None
        self.prev_image = None
        self.episode_id = None
        self.measurements_file = None
        self.weather = None
        self.scenario = None
        self.start_pos = None
        self.end_pos = None
        self.start_coord = None
        self.end_coord = None
        self.last_obs = None

        self.cnt1 = None
        self.displacement = None

    def init_server(self):
        print("Initializing new Carla server...")
        # Create a new server process and start the client.
        self.server_port = random.randint(10000, 60000)
        self.server_process = subprocess.Popen(
            [
                SERVER_BINARY, self.config["server_map"], "-windowed",
                "-ResX=480", "-ResY=360", "-carla-server", "-benchmark -fps=10",   # "-benchmark -fps=10": to run the simulation at a fixed time-step of 0.1 seconds
                "-carla-world-port={}".format(self.server_port)
            ],
            preexec_fn=os.setsid,
            stdout=open(os.devnull, "w"))         #  ResourceWarning: unclosed file <_io.TextIOWrapper name='/dev/null' mode='w' encoding='UTF-8'>
        live_carla_processes.add(os.getpgid(self.server_process.pid))

        for i in range(RETRIES_ON_ERROR):
            try:
                self.client = CarlaClient("localhost", self.server_port)
                return self.client.connect()
            except Exception as e:
                print("Error connecting: {}, attempt {}".format(e, i))
                time.sleep(2)

    def clear_server_state(self):
        print("Clearing Carla server state")
        try:
            if self.client:
                self.client.disconnect()
                self.client = None
        except Exception as e:
            print("Error disconnecting client: {}".format(e))
            pass
        if self.server_process:
            pgid = os.getpgid(self.server_process.pid)
            os.killpg(pgid, signal.SIGKILL)
            live_carla_processes.remove(pgid)
            self.server_port = None
            self.server_process = None

    def __del__(self):  # the __del__ method will be called when the instance of the class is deleted.(memory is freed.)
        self.clear_server_state()

    def reset(self):
        error = None
        for _ in range(RETRIES_ON_ERROR):
            try:
                if not self.server_process:
                    self.init_server()
                return self._reset()
            except Exception as e:
                print("Error during reset: {}".format(traceback.format_exc()))
                self.clear_server_state()
                error = e
        raise error

    # @set_timeout(15)
    def _reset(self):
        self.num_steps = 0
        self.total_reward = 0
        self.prev_measurement = None
        self.prev_image = None
        self.episode_id = datetime.today().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.measurements_file = None


        # Create a CarlaSettings object. This object is a wrapper around
        # the CarlaSettings.ini file. Here we set the configuration we
        # want for the new episode.
        settings = CarlaSettings()
        self.scenario = random.choice(self.config["scenarios"])
        assert self.scenario["city"] == self.city, (self.scenario, self.city)
        self.weather = random.choice(self.scenario["weather_distribution"])
        settings.set(
            SynchronousMode=True,
            # ServerTimeOut=10000, # CarlaSettings: no key named 'ServerTimeOut'
            SendNonPlayerAgentsInfo=True,
            NumberOfVehicles=self.scenario["num_vehicles"],
            NumberOfPedestrians=self.scenario["num_pedestrians"],
            WeatherId=self.weather)
        settings.randomize_seeds()


        if self.config["use_sensor"] == 'use_semantic':
            camera0 = Camera("CameraSemantic", PostProcessing="SemanticSegmentation")
            camera0.set_image_size(self.config["render_x_res"],
                                   self.config["render_y_res"])
            # camera0.set_position(30, 0, 130)
            camera0.set(FOV=120)
            camera0.set_position(2.0, 0.0, 1.4)
            camera0.set_rotation(0.0, 0.0, 0.0)

            settings.add_sensor(camera0)


        if self.config["use_sensor"] in ['use_depth','use_logdepth']:
            camera1 = Camera("CameraDepth", PostProcessing="Depth")
            camera1.set_image_size(self.config["render_x_res"],
                                   self.config["render_y_res"])
            # camera1.set_position(30, 0, 130)
            camera1.set(FOV=120)
            camera1.set_position(2.0, 0.0, 1.4)
            camera1.set_rotation(0.0, 0.0, 0.0)

            settings.add_sensor(camera1)


        if self.config["use_sensor"] == 'use_rgb':
            camera2 = Camera("CameraRGB")
            camera2.set_image_size(self.config["render_x_res"],
                                   self.config["render_y_res"])
            # camera2.set_position(30, 0, 130)
            # camera2.set_position(0.3, 0.0, 1.3)
            camera2.set(FOV=120)
            camera2.set_position(2.0, 0.0, 1.4)
            camera2.set_rotation(0.0, 0.0, 0.0)

            settings.add_sensor(camera2)

        if self.config["use_sensor"] == 'use_2rgb':
            camera_l = Camera("CameraRGB_L")
            camera_l.set_image_size(self.config["render_x_res"],
                                   self.config["render_y_res"])
            camera_l.set(FOV=120)
            camera_l.set_position(2.0, -0.1, 1.4)
            camera_l.set_rotation(0.0, 0.0, 0.0)
            settings.add_sensor(camera_l)

            camera_r = Camera("CameraRGB_R")
            camera_r.set_image_size(self.config["render_x_res"],
                                   self.config["render_y_res"])
            camera_r.set(FOV=120)
            camera_r.set_position(2.0, 0.1, 1.4)
            camera_r.set_rotation(0.0, 0.0, 0.0)
            settings.add_sensor(camera_r)


        # Setup start and end positions
        scene = self.client.load_settings(settings)
        self.positions = positions = scene.player_start_spots
        self.start_pos = positions[self.scenario["start_pos_id"]]

        self.pre_pos = self.start_pos.location
        self.cnt1 = 0
        self.displacement = 1000.0

        self.end_pos = positions[self.scenario["end_pos_id"]]
        self.start_coord = [
            self.start_pos.location.x // 1, self.start_pos.location.y // 1
        ]
        self.end_coord = [
            self.end_pos.location.x // 1, self.end_pos.location.y // 1
        ]
        print("Start pos {} ({}), end {} ({})".format(
            self.scenario["start_pos_id"], self.start_coord,
            self.scenario["end_pos_id"], self.end_coord))

        # Notify the server that we want to start the episode at the
        # player_start index. This function blocks until the server is ready
        # to start the episode.
        print("Starting new episode...")
        self.client.start_episode(self.scenario["start_pos_id"])

        # remove the vehicle dropping when starting a new episode.
        cnt = 1; z1 = 0
        zero_control = VehicleControl()
        while ( cnt < 3 ):
            self.client.send_control(zero_control)   # VehicleControl().steer = 0, VehicleControl().throttle = 0, VehicleControl().reverse = False
            z0=z1
            z1 = self.client.read_data()[0].player_measurements.transform.location.z
            print(z1)
            if z0 - z1 == 0:
                cnt += 1
        print('Starting new episode done.\n')


        # Process observations: self._read_observation() returns image and py_measurements.
        image, py_measurements = self._read_observation()
        self.prev_measurement = py_measurements

        # self.current_heading = self.pre_heading = np.array([py_measurements["x_orient"], py_measurements["y_orient"]])
        # self.angular_speed = 0.0

        self.pre_heading_degree = self.current_heading_degree = py_measurements["current_heading_degree"]
        self.angular_speed_degree = np.array(0.0)

        return self.encode_obs(self.preprocess_image(image), py_measurements), py_measurements

    def encode_obs(self, image, py_measurements):
        assert self.config["framestack"] in [1, 2]
        prev_image = self.prev_image
        self.prev_image = image
        if prev_image is None:
            prev_image = image
        if self.config["framestack"] == 2:
            image = np.concatenate([prev_image, image], axis=2)
        # obs = (image, COMMAND_ORDINAL[py_measurements["next_command"]], [
        #     py_measurements["forward_speed"],
        #     py_measurements["distance_to_goal"]
        # ])
        obs = image
        self.last_obs = obs
        return obs

    def step(self, action):
        try:
            obs = self._step(action)
            return obs
        except Exception:
            print("Error during step, terminating episode early",
                  traceback.format_exc())
            self.clear_server_state()
            return (self.last_obs, 0.0, True, self.prev_measurement)


    def delta_degree(self, deltaxy):
        return  deltaxy if np.abs(deltaxy) < 180 else deltaxy - np.sign(deltaxy) * 360

    # image, py_measurements = self._read_observation()  --->  self.preprocess_image(image)   --->  step observation output
    # @set_timeout(10)
    def _step(self, action):

        if self.config["discrete_actions"]:
            action = DISCRETE_ACTIONS[int(action)]  # Carla action is 2D.
        assert len(action) == 2, "Invalid action {}".format(action)

        if self.enable_autopilot:
            action[0] = self.autopilot.brake if self.autopilot.brake < 0 else self.autopilot.throttle
            action[1] = self.autopilot.steer
        if self.config["squash_action_logits"]:
            forward = 2 * float(sigmoid(action[0]) - 0.5)
            throttle = float(np.clip(forward, 0, 1))
            brake = float(np.abs(np.clip(forward, -1, 0)))
            steer = 2 * float(sigmoid(action[1]) - 0.5)
        else:
            throttle = float(np.clip(action[0], 0, 1))
            brake = float(np.abs(np.clip(action[0], -1, 0)))
            steer = float(np.clip(action[1], -1, 1))

        # reverse and hand_brake are disabled.
        reverse = False
        hand_brake = False

        if self.config["verbose"]:
            print("steer", steer, "throttle", throttle, "brake", brake,
                  "reverse", reverse)

        # print(self.client)
        self.client.send_control(
            steer=steer,
            throttle=throttle,
            brake=brake,
            hand_brake=hand_brake,
            reverse=reverse)

        # Process observations: self._read_observation() returns image and py_measurements.
        image, py_measurements = self._read_observation()
        if self.config["verbose"]:
            print("Next command", py_measurements["next_command"])
        print("Next command", py_measurements["next_command"])

        if type(action) is np.ndarray:
            py_measurements["action"] = [float(a) for a in action]
        else:
            py_measurements["action"] = action
        py_measurements["control"] = {
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
            "reverse": reverse,
            "hand_brake": hand_brake,
        }


        # compute angular_speed
        self.current_heading_degree = py_measurements["current_heading_degree"]
        self.angular_speed_degree = np.array(self.delta_degree(self.current_heading_degree-self.pre_heading_degree))
        self.pre_heading_degree = py_measurements["current_heading_degree"]

        # compute reward
        reward = compute_reward(self, self.prev_measurement, py_measurements)

        self.total_reward += reward
        py_measurements["reward"] = reward
        py_measurements["total_reward"] = self.total_reward

        # done or not
        # done = False
        # done = self.cnt1 > 50 and (py_measurements["collision_vehicles"] or py_measurements["collision_pedestrians"] or py_measurements["collision_other"] or self.displacement < 0.5)
        done = self.cnt1 > 50 and self.displacement < 0.2

        # done = (self.num_steps > self.scenario["max_steps"]
        #         or py_measurements["next_command"] == "REACH_GOAL" or py_measurements["intersection_offroad"] or py_measurements["intersection_otherlane"]
        #         or (self.config["early_terminate_on_collision"]
        #             and collided_done(py_measurements)))

        py_measurements["done"] = done
        self.prev_measurement = py_measurements

        # Write out measurements to file
        if self.config["verbose"] and CARLA_OUT_PATH:
            if not self.measurements_file:
                self.measurements_file = open(
                    os.path.join(
                        CARLA_OUT_PATH,
                        "measurements_{}.json".format(self.episode_id)), "w")
            self.measurements_file.write(json.dumps(py_measurements))
            self.measurements_file.write("\n")
            if done:
                self.measurements_file.close()
                self.measurements_file = None
                if self.config["convert_images_to_video"]:
                    self.images_to_video()

        self.num_steps += 1
        image = self.preprocess_image(image)
        return (self.encode_obs(image, py_measurements), reward, done,
                py_measurements)

    def images_to_video(self):
        videos_dir = os.path.join(CARLA_OUT_PATH, "Videos")
        if not os.path.exists(videos_dir):
            os.makedirs(videos_dir)
        ffmpeg_cmd = (
            "ffmpeg -loglevel -8 -r 60 -f image2 -s {x_res}x{y_res} "
            "-start_number 0 -i "
            "{img}_%04d.jpg -vcodec libx264 {vid}.mp4 && rm -f {img}_*.jpg "
        ).format(
            x_res=self.config["render_x_res"],
            y_res=self.config["render_y_res"],
            vid=os.path.join(videos_dir, self.episode_id),
            img=os.path.join(CARLA_OUT_PATH, "CameraRGB", self.episode_id))
        print("Executing ffmpeg command", ffmpeg_cmd)
        subprocess.call(ffmpeg_cmd, shell=True)

    def preprocess_image(self, image):

        if self.config["use_sensor"] == 'use_semantic':              # image.data: uint8(0 ~ 12)
            data = image.data * 21                                   # data: uint8(0 ~ 255)
            data = data.reshape(self.config["render_y_res"],
                                self.config["render_x_res"], 1)
            data = cv2.resize(
                data, (self.config["x_res"], self.config["y_res"]),
                interpolation=cv2.INTER_AREA)
            data = np.expand_dims(data, 2)                          # data: uint8(0 ~ 255),  shape(y_res, x_res, 1)

        elif self.config["use_sensor"] == 'use_depth':              # depth: float64(0 ~ 1)
            # data = (image.data - 0.5) * 2
            data = image.data * 255                                 # data: float64(0 ~ 255)
            data = data.reshape(self.config["render_y_res"],
                                self.config["render_x_res"], 1)     # shape(render_y_res,render_x_res,1)
            data = cv2.resize(
                data, (self.config["x_res"], self.config["y_res"]),
                interpolation=cv2.INTER_AREA)                       # shape(y_res, x_res)
            data = np.expand_dims(data, 2)                          # data: float64(0 ~ 255),  shape(y_res, x_res, 1)

        elif self.config["use_sensor"] == 'use_logdepth':           # depth: float64(0 ~ 1)
            data = (np.log(image.data)+7.0)*255.0/7.0               # data: float64(0 ~ 255)
            data = data.reshape(self.config["render_y_res"],
                                self.config["render_x_res"], 1)     # shape(render_y_res,render_x_res,1)
            data = cv2.resize(
                data, (self.config["x_res"], self.config["y_res"]),
                interpolation=cv2.INTER_AREA)                       # shape(y_res, x_res)
            data = np.expand_dims(data, 2)                          # data: float64(0 ~ 255),  shape(y_res, x_res, 1)


        elif self.config["use_sensor"] == 'use_rgb':
            data = image.data.reshape(self.config["render_y_res"],
                                      self.config["render_x_res"], 3)
            data = cv2.resize(
                data, (self.config["x_res"], self.config["y_res"]),        # data: uint8(0 ~ 255),  shape(y_res, x_res, 3)
                interpolation=cv2.INTER_AREA)
            # data = (data.astype(np.float32) - 128) / 128

        elif self.config["use_sensor"] == 'use_2rgb':
            data_l, data_r= image[0].data, image[0].data
            data_l = data_l.reshape(self.config["render_y_res"],
                                      self.config["render_x_res"], 3)
            data_r = data_r.reshape(self.config["render_y_res"],
                                      self.config["render_x_res"], 3)
            data_l = cv2.resize(
                data_l, (self.config["x_res"], self.config["y_res"]),
                interpolation=cv2.INTER_AREA)
            data_r = cv2.resize(
                data_r, (self.config["x_res"], self.config["y_res"]),
                interpolation=cv2.INTER_AREA)

            data = np.concatenate((data_l, data_r), axis=2)    # data: uint8(0 ~ 255),  shape(y_res, x_res, 6)


        return data

    def _read_observation(self):
        # Read the data produced by the server this frame.
        measurements, sensor_data = self.client.read_data()

        if self.enable_autopilot:
            self.autopilot = measurements.player_measurements.autopilot_control

        # Print some of the measurements.
        if self.config["verbose"]:
            print_measurements(measurements)

        observation = None

        if self.config["use_sensor"] == 'use_semantic':
            camera_name = "CameraSemantic"

        elif self.config["use_sensor"] in ['use_depth','use_logdepth']:
            camera_name = "CameraDepth"

        elif self.config["use_sensor"] == 'use_rgb':
            camera_name = "CameraRGB"

        elif self.config["use_sensor"] == 'use_2rgb':
            camera_name = ["CameraRGB_L","CameraRGB_R"]

        # for name, image in sensor_data.items():
        #     if name == camera_name:
        #         observation = image
        if camera_name == ["CameraRGB_L","CameraRGB_R"]:
            observation = [sensor_data["CameraRGB_L"], sensor_data["CameraRGB_R"]]
        else:
            observation = sensor_data[camera_name]



        cur = measurements.player_measurements

        if self.config["enable_planner"]:
            next_command = COMMANDS_ENUM[self.planner.get_next_command(
                [cur.transform.location.x, cur.transform.location.y, GROUND_Z],\
                [cur.transform.orientation.x, cur.transform.orientation.y,GROUND_Z],\
                [self.end_pos.location.x, self.end_pos.location.y, GROUND_Z],\
                [self.end_pos.orientation.x, self.end_pos.orientation.y,GROUND_Z]\
                )]

            # modify next_command
            current_pos = np.array([cur.transform.location.x, cur.transform.location.y])\
                + np.array([cur.transform.orientation.x, cur.transform.orientation.y]) * 5.0
            dist_intersection_current_pos = np.linalg.norm(self.intersections_pos[:,:2] - current_pos, axis=1)
            is_near_intersection = np.min(dist_intersection_current_pos) < 18.0
            if not is_near_intersection:
                next_command = "LANE_FOLLOW"

            # goal_heading
            if is_near_intersection:
                cur_intersection = self.intersections_pos[dist_intersection_current_pos.argmin(),:2]
            else:
                self.goal_heading_degree = 0.0
            if is_near_intersection and np.linalg.norm(self.pre_intersection- cur_intersection) > 0.1:

                cur_heading0 = cur_intersection - current_pos
                cur_heading_1 = cur_heading0/np.linalg.norm(cur_heading0)
                cur_heading_degree = self.headings_degree[np.linalg.norm(cur_heading_1 - self.headings, axis=1).argmin()]
                self.goal_heading_degree = self.delta_degree(cur_heading_degree + self.lrs_degree[next_command])

                self.pre_intersection = cur_intersection


        else:
            next_command = "LANE_FOLLOW"


        if next_command == "REACH_GOAL":
            distance_to_goal = 0.0  # avoids crash in planner
            self.end_pos = self.positions[random.choice(self.config["scenarios"])['end_pos_id']]
        elif self.config["enable_planner"]:
            distance_to_goal = self.planner.get_shortest_path_distance([
                cur.transform.location.x, cur.transform.location.y, GROUND_Z
            ], [
                cur.transform.orientation.x, cur.transform.orientation.y,
                GROUND_Z
            ], [self.end_pos.location.x, self.end_pos.location.y, GROUND_Z], [
                self.end_pos.orientation.x, self.end_pos.orientation.y,
                GROUND_Z
            ])
        else:
            distance_to_goal = -1

        distance_to_goal_euclidean = float(
            np.linalg.norm([  # default norm: L2 norm
                cur.transform.location.x - self.end_pos.location.x,
                cur.transform.location.y - self.end_pos.location.y
            ]) )


        # displacement
        if self.cnt1 > 70 and self.cnt1 % 30 == 0:
            self.displacement = float(
                np.linalg.norm([
                    cur.transform.location.x - self.pre_pos.x,
                    cur.transform.location.y - self.pre_pos.y
                ]) )
            self.pre_pos = cur.transform.location
        self.cnt1 += 1


        py_measurements = {
            "episode_id": self.episode_id,
            "step": self.num_steps,
            "x": cur.transform.location.x,
            "y": cur.transform.location.y,
            "x_orient": cur.transform.orientation.x,
            "y_orient": cur.transform.orientation.y,
            "forward_speed": cur.forward_speed,
            "distance_to_goal": distance_to_goal,
            "distance_to_goal_euclidean": distance_to_goal_euclidean,
            "collision_vehicles": cur.collision_vehicles,
            "collision_pedestrians": cur.collision_pedestrians,
            "collision_other": cur.collision_other,
            "intersection_offroad": cur.intersection_offroad,
            "intersection_otherlane": cur.intersection_otherlane,
            "weather": self.weather,
            "map": self.config["server_map"],
            "start_coord": self.start_coord,
            "end_coord": self.end_coord,
            "current_scenario": self.scenario,
            "x_res": self.config["x_res"],
            "y_res": self.config["y_res"],
            "num_vehicles": self.scenario["num_vehicles"],
            "num_pedestrians": self.scenario["num_pedestrians"],
            "max_steps": self.scenario["max_steps"],
            "next_command": next_command,
            "next_command_id": COMMAND_ORDINAL[next_command],
            "displacement": self.displacement,
            "is_near_intersection": is_near_intersection,

            "goal_heading_degree": self.goal_heading_degree,
            "angular_speed_degree": self.angular_speed_degree,
            "current_heading_degree":cur.transform.rotation.yaw,  # left-, right+, (-180 ~ 180) degrees
        }

        if CARLA_OUT_PATH and self.config["log_images"]:
            for name, image in sensor_data.items():
                out_dir = os.path.join(CARLA_OUT_PATH, name)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                out_file = os.path.join(
                    out_dir, "{}_{:>04}.jpg".format(self.episode_id,
                                                    self.num_steps))
                scipy.misc.imsave(out_file, image.data)            # image.data without preprocess.

        assert observation is not None, sensor_data
        return observation, py_measurements




def compute_reward_corl2017(env, prev, current):
    reward = 0.0

    cur_dist = current["distance_to_goal"]

    prev_dist = prev["distance_to_goal"]

    if env.config["verbose"]:
        print("Cur dist {}, prev dist {}".format(cur_dist, prev_dist))

    # Distance travelled toward the goal in m
    reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)

    # Change in speed (km/h)
    reward += 0.05 * (current["forward_speed"] - prev["forward_speed"])

    # New collision damage
    reward -= .00002 * (
        current["collision_vehicles"] + current["collision_pedestrians"] +
        current["collision_other"] - prev["collision_vehicles"] -
        prev["collision_pedestrians"] - prev["collision_other"])

    # New sidewalk intersection
    reward -= 2 * (
        current["intersection_offroad"] - prev["intersection_offroad"])

    # New opposite lane intersection
    reward -= 2 * (
        current["intersection_otherlane"] - prev["intersection_otherlane"])

    return reward


def compute_reward_custom(env, prev, current):
    reward = 0.0

    cur_dist = current["distance_to_goal"]
    prev_dist = prev["distance_to_goal"]

    if env.config["verbose"]:
        print("Cur dist {}, prev dist {}".format(cur_dist, prev_dist))

    # Distance travelled toward the goal in m
    reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)

    # Speed reward, up 30.0 (km/h)
    reward += np.clip(current["forward_speed"], 0.0, 30.0) / 10

    # New collision damage
    new_damage = (
        current["collision_vehicles"] + current["collision_pedestrians"] +
        current["collision_other"] - prev["collision_vehicles"] -
        prev["collision_pedestrians"] - prev["collision_other"])
    if new_damage:
        reward -= 100.0

    # Sidewalk intersection
    reward -= current["intersection_offroad"]

    # Opposite lane intersection
    reward -= current["intersection_otherlane"]

    # Reached goal
    if current["next_command"] == "REACH_GOAL":
        reward += 100.0

    return reward

def compute_reward_custom1(env, prev, current):
    reward = 0.0

    # cur_dist = current["distance_to_goal"]
    # prev_dist = prev["distance_to_goal"]
    #
    # if env.config["verbose"]:
    #     print("Cur dist {}, prev dist {}".format(cur_dist, prev_dist))
    #
    # # Distance travelled toward the goal in m
    # reward += 0.5 * np.clip(prev_dist - cur_dist, -12.0, 12.0)

    # Speed reward, up 30.0 (km/h)
    reward += np.clip(current["forward_speed"], 0.0, 30.0) / 10
    if current["forward_speed"] > 40:
        reward -= (current["forward_speed"] - 40)/12
    # New collision damage
    new_damage = (
        current["collision_vehicles"] + current["collision_pedestrians"] +
        current["collision_other"] - prev["collision_vehicles"] -
        prev["collision_pedestrians"] - prev["collision_other"])
    # print(current["collision_other"], current["collision_vehicles"], current["collision_pedestrians"])
    # 0.0 41168.109375 0.0
    if new_damage:
        reward -= 100.0

    # Sidewalk intersection [0, 1]
    reward -= np.clip(5 * current["forward_speed"] * int(current["intersection_offroad"] > 0.0001), 0.0, 50)
    # print(current["intersection_offroad"])
    # Opposite lane intersection
    reward -= 4 * current["intersection_otherlane"]  # [0 ~ 1]
    # print(current["intersection_offroad"], current["intersection_otherlane"])
    # Reached goal
    # if current["next_command"] == "REACH_GOAL":
    #     reward += 200.0
    #     print('bro, you reach the goal, well done!!!')

    return reward


def compute_reward_custom4(env, prev, current):
    reward = 0.0

    # cur_dist = current["distance_to_goal"]
    # prev_dist = prev["distance_to_goal"]
    #
    # if env.config["verbose"]:
    #     print("Cur dist {}, prev dist {}".format(cur_dist, prev_dist))
    #
    # # Distance travelled toward the goal in m
    # reward += 0.5 * np.clip(prev_dist - cur_dist, -12.0, 12.0)

    # Speed reward, up 30.0 (km/h)
    # reward += current["forward_speed"]*3.6/ 10.0  # 3.6km/h = 1m/s
    # reward += np.clip(current["forward_speed"]*3.6, 0.0, 30.0) / 10  # 3.6km/h = 1m/s
    reward += np.where(current["forward_speed"]*3.6 < 40.0, current["forward_speed"]*3.6/10, -0.4*current["forward_speed"]*3.6+20.0)
    # New collision damage
    new_damage = (
        current["collision_vehicles"] + current["collision_pedestrians"] +
        current["collision_other"] - prev["collision_vehicles"] -
        prev["collision_pedestrians"] - prev["collision_other"])
    # print(current["collision_other"], current["collision_vehicles"], current["collision_pedestrians"])
    # 0.0 41168.109375 0.0
    if new_damage:
        reward -= 300.0

    # Sidewalk intersection [0, 1]
    reward -= 6 * (current["forward_speed"]+1.0) * current["intersection_offroad"]
    # print(current["intersection_offroad"])
    # Opposite lane intersection
    reward -= 3 * (current["forward_speed"]+1.0) * current["intersection_otherlane"]  # [0 ~ 1]

    return reward




def compute_reward_custom3(env, prev, current):
    reward = 0.0

    # cur_dist = current["distance_to_goal"]
    # prev_dist = prev["distance_to_goal"]
    #
    # if env.config["verbose"]:
    #     print("Cur dist {}, prev dist {}".format(cur_dist, prev_dist))
    #
    # # Distance travelled toward the goal in m
    # reward += 0.5 * np.clip(prev_dist - cur_dist, -12.0, 12.0)

    # Speed reward, up 30.0 (km/h)
    # reward += current["forward_speed"]*3.6/ 10.0  # 3.6km/h = 1m/s
    # reward += np.clip(current["forward_speed"]*3.6, 0.0, 30.0) / 10  # 3.6km/h = 1m/s
    reward += np.where(current["forward_speed"]*3.6 < 30.0, current["forward_speed"]*3.6/10, -0.3*current["forward_speed"]*3.6+12.0)
    # New collision damage
    new_damage = (
        current["collision_vehicles"] + current["collision_pedestrians"] +
        current["collision_other"] - prev["collision_vehicles"] -
        prev["collision_pedestrians"] - prev["collision_other"])
    # print(current["collision_other"], current["collision_vehicles"], current["collision_pedestrians"])
    # 0.0 41168.109375 0.0
    if new_damage:
        reward -= 300.0

    # Sidewalk intersection [0, 1]
    reward -= 5 * (current["forward_speed"]+1.0) * current["intersection_offroad"]
    # print(current["intersection_offroad"])
    # Opposite lane intersection
    reward -= 2 * (current["forward_speed"]+1.0) * current["intersection_otherlane"]  # [0 ~ 1]

    return reward

def compute_reward_custom_depth(env, prev, current):
    reward = 0.0

    # cur_dist = current["distance_to_goal"]
    # prev_dist = prev["distance_to_goal"]
    #
    # if env.config["verbose"]:
    #     print("Cur dist {}, prev dist {}".format(cur_dist, prev_dist))
    #
    # # Distance travelled toward the goal in m
    # reward += 0.5 * np.clip(prev_dist - cur_dist, -12.0, 12.0)

    reward += np.clip(current["forward_speed"] * 3.6, 0.0, 30.0) / 10  # 3.6km/h = 1m/s
    # New collision damage
    new_damage = (
        current["collision_vehicles"] + current["collision_pedestrians"] +
        current["collision_other"] - prev["collision_vehicles"] -
        prev["collision_pedestrians"] - prev["collision_other"])
    # print(current["collision_other"], current["collision_vehicles"], current["collision_pedestrians"])
    # 0.0 41168.109375 0.0
    if new_damage:
        reward -= 300.0

    # Sidewalk intersection [0, 1]
    reward -= 5 * (current["forward_speed"] + 1.0) * current["intersection_offroad"]
    # print(current["intersection_offroad"])
    # Opposite lane intersection
    # reward -= 4 * current["intersection_otherlane"]  # [0 ~ 1]


    return reward

def compute_reward_lane_keep(env, prev, current):
    reward = 0.0

    # Speed reward, up 30.0 (km/h)
    reward += np.clip(current["forward_speed"], 0.0, 30.0) / 10

    # # New collision damage
    # new_damage = (
    #     current["collision_vehicles"] + current["collision_pedestrians"] +
    #     current["collision_other"] - prev["collision_vehicles"] -
    #     prev["collision_pedestrians"] - prev["collision_other"])
    # if new_damage:
    #     reward -= 100.0
    #
    # # Sidewalk intersection
    # reward -= current["intersection_offroad"]
    #
    # # Opposite lane intersection
    # reward -= current["intersection_otherlane"]

    return reward


REWARD_FUNCTIONS = {
    "corl2017": compute_reward_corl2017,
    "custom": compute_reward_custom,
    "custom1": compute_reward_custom1,
    "custom4": compute_reward_custom4,
    "custom3": compute_reward_custom3,
    "custom_depth": compute_reward_custom_depth,
    "lane_keep": compute_reward_lane_keep,
}
def compute_reward(env, prev, current):
    return REWARD_FUNCTIONS[env.config["reward_function"]](env, prev, current)


def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = "Vehicle at ({pos_x:.1f}, {pos_y:.1f}), "
    message += "{speed:.2f} km/h, "
    message += "Collision: {{vehicles={col_cars:.0f}, "
    message += "pedestrians={col_ped:.0f}, other={col_other:.0f}}}, "
    message += "{other_lane:.0f}% other lane, {offroad:.0f}% off-road, "
    message += "({agents_num:d} non-player agents in the scene)"
    message = message.format(
        pos_x=player_measurements.transform.location.x,
        pos_y=player_measurements.transform.location.y,
        speed=player_measurements.forward_speed,
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents)
    print(message)


def sigmoid(x):
    x = float(x)
    return np.exp(x) / (1 + np.exp(x))


def collided_done(py_measurements):
    m = py_measurements
    collided = (m["collision_vehicles"] > 0 or m["collision_pedestrians"] > 0
                or m["collision_other"] > 0)
    return bool(collided or m["total_reward"] < -100)


if __name__ == "__main__":

    env = CarlaEnv(enable_autopilot=False)
    obs = env.reset()
    print("reset")
    start = time.time()
    done = False
    i = 0
    total_reward = 0.0
    while True:
        i += 1
        if ENV_CONFIG["discrete_actions"]:
            obs, reward, done, info = env.step(3)
        else:

            # command from keyboard.
            commd = input('input command:')    # type (str)
            if commd == 'a':
                steer_commd = -1
            elif commd=='d':
                steer_commd = 1
            else:
                steer_commd = 0
            obs, reward, done, info = env.step([0.5, steer_commd])

            # # fixed command.
            # obs, reward, done, info = env.step([0.5, 0.0])

        total_reward += reward
        # print(i, "reward", reward, "total", total_reward, "done", done)
        print(i, 'displacement:', info['displacement'])
        print('curent_heading:',[info['x_orient'],info['y_orient']])
        print('current_heading_degree:', info["current_heading_degree"])
        print("goal_heading_degree:", info["goal_heading_degree"])
        print("angular_speed_degree:", info["angular_speed_degree"])

        if done:
            env.reset()
            i = 0
            total_reward = 0.0

    # print("{} fps".format(i / (time.time() - start)))

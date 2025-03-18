# generic imports
import time
import multiprocessing
import sys
import os
import torch
import cv2
import zarr
import numpy as np
from tqdm import tqdm
from pynput import keyboard

# pyHannessAPI imports
sys.path.append('../../pyHannesAPI/')
from pyHannesAPI.pyHannes import Hannes
from pyHannesAPI import pyHannes_commands
from pyHannesAPI.data_dumper import HannesDataDumper
from pyHannesAPI.video_dumper import HannesAsynchronousVideoDumper, RealSenseVideoDumper
from pyHannesAPI.pyHannes import timestamp
from pyHannesAPI.camera.opencv_camera import OpenCVCamera
from pyHannesAPI.controller.keyboard import KeyboardController

# hannes_imitation imports
sys.path.append('../../hannes-imitation/')
sys.path.append('../../hannes-imitation/hannes_imitation/external/diffusion_policy') # NOTE otherwise importing SequenceSampler fails
from hannes_imitation.control.diffusion_policy import HannesDiffusionPolicyController
from hannes_imitation.external.diffusion_policy.diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy

import argparse
    
def reset_motion(hannes, steps=20, delay=0.20):
    # get current robot position
    #pos_hand = hannes.measurements_hand()['position']
    #pos_wristFE = hannes.measurements_wristFE()['position']
    ref_hand = hannes.hannes_state['references']['hand']
    ref_wristFE = hannes.hannes_state['references']['wrist_FE']

    move_hand_list = np.linspace(ref_hand, 0, steps)
    move_wristFE_list = np.linspace(ref_wristFE, 90, steps)
    
    # gradually reset hand
    print("===== resetting hand =====")
    for pos in move_hand_list:
        print(pos)
        hannes.move_hand(int(pos))
        time.sleep(delay)

    # gradually reset wrist FE
    print("===== resetting wristFE =====")
    for pos in move_wristFE_list:
        print(pos)
        hannes.move_wristFE(int(pos))
        time.sleep(delay)
    
    hannes.move_wristPS(0)
    time.sleep(delay)
    hannes.move_thumb_home()
    time.sleep(delay)
    hannes.move_thumb_power()

def torch_warmup(policy, device):
    # call policy with dummy inputs to initialize cuda
    dummy_obs_dict = {
        'image_in_hand': torch.zeros([1, 2, 3, 96, 128], dtype=torch.float32).to(device),
        'mes_hand': torch.zeros([1, 2, 1], dtype=torch.float32).to(device),
        'mes_wrist_FE': torch.zeros([1, 2, 1], dtype=torch.float32).to(device)}
    
    print("===== testing policy prediction time =====")
    for i in range(10):
        tic = time.time()
        policy.predict_action(dummy_obs_dict)
        toc = time.time()
        print("Trial %d, time: %.3f s" % (i, toc - tic))

def camera_warmup(camera, camera_type):
    # camera : camera handle
    # type : 'cv' for OpenCV or 'rs' for RealSense
    assert(camera_type == 'cv' or camera_type == 'rs')
    camera_read_fun = camera.read if camera_type == 'cv' else camera.read_frame()

    print("===== testing camera read time =====")
    for i in range(10):
        tic = time.time()
        camera_read_fun()
        toc = time.time()
        print("Trial %d, time: %.3f s" % (i, toc - tic))

def do_test(controller, episode_len):
    action_trajectories = []
    action_timestamps = []

    tic = time.time()
    while time.time() - tic <= episode_len: # run for maximum episode_len seconds
        obs_dict = controller.get_observation() # NOTE potrebbe essere real_env.get_observation() simile a interaccia Gym 
        action_trajectory = controller.predict(obs_dict)
        controller.step(action_trajectory)

        action_trajectories.append(action_trajectory)
        action_timestamps.append(time.time())
    toc = time.time()

    test_duration = toc - tic

    return action_trajectories, action_timestamps, test_duration

def start_processes(flag_demo, process_list):
    flag_demo.set() # flag process start
    for process in process_list:
        process.start()

def stop_processes(flag_demo, process_list):
    flag_demo.clear() # flag process termination
    for process in process_list:
        process.join()

def make_in_hand_camera(index):
    in_hand_camera = OpenCVCamera(camera_index=index, width=640, height=480)
    # set camera properties
    in_hand_camera.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # default 3, 1 disables autoexposure
    in_hand_camera.camera.set(cv2.CAP_PROP_EXPOSURE, 500) # default 166. 300=30fps, 400=25fps, 500=20fps
    in_hand_camera.camera.set(cv2.CAP_PROP_GAIN, 100) # 0-128, default 64 (gain=100 is ok when auto exposure is disabled and exposure is 500)
    
    assert(in_hand_camera.camera.get(cv2.CAP_PROP_AUTO_EXPOSURE) == 1)
    assert(in_hand_camera.camera.get(cv2.CAP_PROP_EXPOSURE) == 500)
    assert(in_hand_camera.camera.get(cv2.CAP_PROP_GAIN) == 100)

    return in_hand_camera

def pipeline():
    if store_dir != None:
        # e.g., '/home/calessi-iit.local/Projects/hannes-imitation/data/test/policy_1-8/'
        test_data_base_name = 'collection_%s_object_%s' % (collection, obj) 
        test_data_base_name += timestamp() + '.zarr'
        test_data_path = os.path.join(store_dir, test_data_base_name)
        store = zarr.open(test_data_path, mode='w')

        print("Test will be stored at:\n%s" % test_data_path)
    else:
        print("Data storage disabled.")
        test_data_path = None
        store = None

    # reset motion
    #print("")
    #reset_motion(hannes)

    # load model
    policy_path = '/home/calessi-iit.local/Projects/hannes-imitation/trainings/policy_1_4_7_2025_2_19-21_10_9.pth' # iros2025
    checkpoint = torch.load(policy_path)
    policy = checkpoint['policy']

    # device transfer
    device = torch.device('cuda')
    _ = policy.to(device).eval()

    observation_horizon = policy.n_obs_steps
    action_horizon = policy.n_action_steps

    print("Observation horizon: %d" % observation_horizon)
    print("Action horizon: %d" % action_horizon)
    print("Diffusion iterations: %d" % policy.num_inference_steps)
    print("Episode duration: %.2f s" % episode_len)
    print("Device: %s" %  device)

    in_hand_camera = make_in_hand_camera(index=4)
    torch_warmup(policy, device)
    camera_warmup(in_hand_camera.camera, camera_type='cv')

    # Create share memory objects for synchronized access
    manager = multiprocessing.Manager()
    flag_demo = manager.Event()
    flag_demo.clear()
    frames_list = manager.list()  # (B=1, To, C, H, W)
    frames_timestamps_list = manager.list()
    external_frames_list = manager.list()
    hannes_state_list = manager.list()
    
    hannes_video_capture = HannesAsynchronousVideoDumper(
        camera=in_hand_camera, 
        kwargs=dict(
            flag_demo=flag_demo, 
            frames_list=frames_list, 
            timestamps_list=frames_timestamps_list))

    hannes_hand_capture = HannesDataDumper(
        kwargs={'hannes_state': hannes.hannes_state,
                'flag_demo': flag_demo,
                'hannes_state_cond': hannes.hannes_state_cond,
                'hannes_state_list': hannes_state_list,
                'store': store})

    # external camera for visualization purposes width=640, height=480
    external_video_capture = RealSenseVideoDumper(
        enable_color=True, enable_depth=False, width=1920, height=1080, fps=30,
        name='external', kwargs=dict(flag_demo=flag_demo, frames_list=external_frames_list, store=store))

    controller = HannesDiffusionPolicyController(
        hannes=hannes, policy=policy, control_frequency=20, observation_horizon=observation_horizon, 
        action_horizon=action_horizon, frames_list=frames_list, hannes_states_list=hannes_state_list)

    # TEST LOOP
    process_list = [hannes_video_capture, hannes_hand_capture, external_video_capture]
    start_processes(flag_demo, process_list)
    print("===== start =====")
    action_trajectories, action_timestamps, test_duration = do_test(controller, episode_len)
    print("===== end =====")
    stop_processes(flag_demo, process_list)

    time.sleep(3)
    print("===== please reset motion =====")
    #reset_motion(hannes)

    del policy
    del hannes_video_capture
    del hannes_hand_capture
    del external_video_capture

    # add more stuff to the store
    if store:
        store['policy_actions'] = np.array(action_trajectories)
        store['prediction_timestamps'] = np.array(action_timestamps)
        store['test_duration'] = np.array([test_duration])
        store['camera_frames'] = np.array([frame['rgb'] for frame in frames_list]) 
        store['camera_timestamps'] = np.array(frames_timestamps_list)

    print("Store path: %s" % test_data_path)
    print("Policy called %d times" % len(action_trajectories))
    print("Actual episode duration: %.2f s" % (test_duration))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deploy HannesImitation policy")

    # Adding arguments
    parser.add_argument("--store_dir", type=str, default=None, help="Where to store the trial. Specify None for not saving it.")
    parser.add_argument("--object", type=str, default=None, help="Name of the object to grasp.")
    parser.add_argument("--collection", type=str, default=None, help="Name of the object to grasp.")
    parser.add_argument("--episode_len", type=float, default=10, help="Number of seconds to run the task.")

    # Parsing arguments
    args = parser.parse_args()
    store_dir = args.store_dir
    obj = args.object
    collection = args.collection
    episode_len = args.episode_len

    # create hand handle
    hannes = Hannes(device_name='HANNESFA', timeout=None)

    keyboard_controller = KeyboardController()
    # press key
    keyboard_controller.add_key_press(key='q', function=hannes.move_hand_delta, args=[3]) # close hand
    keyboard_controller.add_key_press(key='w', function=hannes.move_hand_delta, args=[-3]) # open hand
    keyboard_controller.add_key_press(key='a', function=hannes.move_wristFE_delta, args=[-3]) # extend wrist
    keyboard_controller.add_key_press(key='s', function=hannes.move_wristFE_delta, args=[3]) # flex wrist
    keyboard_controller.add_key_press(key='z', function=hannes.move_wristPS, args=[30]) # pronate wrist
    keyboard_controller.add_key_press(key='x', function=hannes.move_wristPS, args=[-30]) # supinate wrist
    
    keyboard_controller.add_key_press(key=keyboard.Key.enter, function=pipeline, args=[]) # start test

    # release key
    keyboard_controller.add_key_release(key='z', function=hannes.move_wristPS, args=[0]) # stop wrist pronation
    keyboard_controller.add_key_release(key='x', function=hannes.move_wristPS, args=[0]) # stop wrist supination

    try:
        hannes.connect()
        hannes.enable_reply_measurements() # header 19
        hannes.set_control_modality(pyHannes_commands.HControlModality.CONTROL_UNITY) # now you can use hand 

        # start and join keyboard listener
        keyboard_controller.start()
        keyboard_controller.join()

    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    finally:
        # disconnect
        hannes.disconnect()
        time.sleep(1)
        del hannes
import time
import numpy as np

import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import RolloutPolicy

import zmq
import json



def run_trained_agent(model_path):

    # load ckpt dict and get algo name for sanity checks
    algo_name, ckpt_dict = FileUtils.algo_name_from_checkpoint(ckpt_path=model_path)

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)
    assert isinstance(policy, RolloutPolicy)

    policy.start_episode()

    # read rollout settings
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:12345")
    while True:
        print('========================       wating for obs data...       ========================')
        data_json = socket.recv_string()
        print("======================== received data, start processing... ========================")
        data = json.loads(data_json)
        name = data["name"]
        pointcloud = np.array(data["pointcloud"])
        # robot = np.array(data["robot"])
        # robot0_eef_hand = robot[:,:1]
        # robot0_eef_pos = robot[:,1:7]
        # robot0_eef_quat = robot[:,7:15]
        # robot0_eef_wrench = robot[:,15:21]
        robot0_eef_hand = np.array(data["robot0_eef_hand"])
        robot0_eef_pos = np.array(data["robot0_eef_pos"])
        robot0_eef_quat = np.array(data["robot0_eef_quat"])
        robot0_eef_wrench = np.array(data["robot0_eef_wrench"])
        obs = dict(pointcloud=pointcloud,
                   robot0_eef_hand=robot0_eef_hand,
                   robot0_eef_pos=robot0_eef_pos,
                   robot0_eef_quat=robot0_eef_quat,
                   robot0_eef_wrench=robot0_eef_wrench)
        # act = policy(ob=obs)
        t0 = time.time()
        act_seq = policy.get_all_action(ob=obs)
        print("========================       inference time: {}           ========================".format(time.time() - t0))
        act_seq = np.array(act_seq)
        print("========================  process done, send data back...   ========================")
        output_data = {
            "name": name,
            "action": act_seq.tolist()
        }
        output_data_json = json.dumps(output_data)
        socket.send_string(output_data_json)
        print("========================        send data back done          ========================")
        print(" ")
        print(" ")


if __name__ == "__main__":
    res_str = None
    # model_path = 'trained_models/diffusion_policy_with_obs_wrench_cucumber_peel_0-11/20240627231828/models/model_epoch_1000.pth'
    # model_path = 'trained_models/diffusion_policy_without_wrench_cucumber_peel_0-11/20240627231514/models/model_epoch_1000.pth'
    # model_path = 'trained_models/diffusion_policy_with_obs_wrench_cucumber_peel_0-9/20240816232810/models/model_epoch_300.pth'
    # model_path = 'trained_models/diffusion_policy_with_wrench_cucumber_peel_0-9/20240816232817/models/model_epoch_400.pth'
    # model_path = 'trained_models/dp_w_action_wrench_wo_robot_cucumber_peel_0-9/20240819230044/models/model_epoch_300.pth'
    model_path = 'trained_models/dp_w_action_wrench_w_robot_cucumber_peel_0-9/20240823235554/models/model_epoch_100.pth'
    model_path = 'trained_models/dp_w_action_wrench_wo_robot_cucumber_peel_0-9/20240823235508/models/model_epoch_200.pth'
    model_path='/home/wenhai/pub_repo/robotlearning/DexCap/STEP3_train_policy/robomimic/trained_models/dp_w_action_wrench_wo_robot_cucumber_peel_0-9/20240823235508/models/model_epoch_200.pth'
    try:
        run_trained_agent(model_path)
    except Exception as e:
        import pdb;pdb.set_trace()

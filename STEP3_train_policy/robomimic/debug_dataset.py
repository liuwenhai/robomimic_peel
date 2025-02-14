from robomimic.utils.dataset import SequenceDataset
import robomimic.utils.obs_utils as ObsUtils
from debug_utils import PeelFTDataRos, KeyboardCtrl
import rospy
from collections import OrderedDict
import numpy as np
import tqdm

dataset_kwargs = dict(
    hdf5_path="/home/wenhai/data/peel_data/peel_data_1/peel_skill/cocozelle_shave_train.hdf5",
    obs_keys=["robot0_eef_pos","robot0_eef_quat","pointcloud","robot0_eef_wrench"],
    action_keys=["action_with_wrench"],
    dataset_keys=["action_with_wrench","rewards","dones"],
    action_config={"action_with_wrench":{"normalization": "min_max"}},
    frame_stack=1,
    seq_length=20,
    pad_frame_stack=True,
    pad_seq_length=True,
    get_pad_mask=False,
    goal_mode=None,
    hdf5_cache_mode=None,
    hdf5_use_swmr=True,
    hdf5_normalize_obs=False,
    filter_by_attribute=None,
    load_next_obs=False,
)
agent = PeelFTDataRos()
kb = KeyboardCtrl()
dataset = SequenceDataset(**dataset_kwargs)
max_action = []
min_action = []
idx = 0
for meta in tqdm.tqdm(dataset):
    # import pdb;pdb.set_trace()
    raw_action = meta["action_with_wrench"]
    robot0_eef_pos = meta['obs']['robot0_eef_pos']
    robot0_eef_quat = meta['obs']['robot0_eef_quat']
    robot0_eef_wrench = meta['obs']['robot0_eef_wrench']
    pointcloud = meta['obs']['pointcloud']
    pointcloud[:,:,3:] *= 255
    action = meta['actions']
    index = meta['index']
    rate = rospy.Rate(30)
    action_normalization_stats = dataset.get_action_normalization_stats()
    ac_dict = OrderedDict()
    if not np.all((action>=-1) & (action<=1)):
        import pdb;pdb.set_trace()
    ac_dict['action_with_wrench'] = action
    ac_dict = ObsUtils.unnormalize_dict(ac_dict, normalization_stats=action_normalization_stats)
    post_action = ac_dict['action_with_wrench']
    vis_action = post_action # post_action  raw_action
    max_action.append(np.max(raw_action,axis=0))
    min_action.append(np.min(raw_action,axis=0))
    idx+=1
    # if idx == 50344:
    #     break
    # continue
    for i in range(raw_action.shape[0]):
        agent.update(pointcloud[0], vis_action[i,:3], vis_action[i,3:7], vis_action[i,7:])
        # agent.update(pointcloud[0], robot0_eef_pos[i], robot0_eef_quat[i], robot0_eef_wrench[i])
        rate.sleep()
        if kb.finish:
            break
    if kb.finish:
        break
import pdb;pdb.set_trace()
np.min(np.vstack(min_action),axis=0)
np.max(np.vstack(max_action),axis=0)
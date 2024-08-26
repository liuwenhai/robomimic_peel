#CUDA_VISIBLE_DEVICES=1 taskset -c 16,17,18,19,20,21,22,23 python scripts/train.py --config training_config/dp_w_action_wrench_wo_robot_cucumber_peel_0-9.json
CUDA_VISIBLE_DEVICES=1 taskset -c 24,25,26,27,28,29,30,31 python scripts/train.py --config training_config/dp_w_action_wrench_w_robot_cucumber_peel_0-9.json

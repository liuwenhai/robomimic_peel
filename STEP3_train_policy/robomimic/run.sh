#CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config training_config/diffusion_policy_without_wrench_cucumber_peel_0-11.json --debug --resume trained_models/diffusion_policy_without_wrench_cucumber_peel_0-11/20240627231514/models/model_epoch_1000.pth
#CUDA_VISIBLE_DEVICES=1 python scripts/test_real.py --agent trained_models/diffusion_policy_without_wrench_cucumber_peel_0-11/20240627231514/models/model_epoch_1000.pth
CUDA_VISIBLE_DEVICES=1 python scripts/test_real.py --agent trained_models/diffusion_policy_with_obs_wrench_cucumber_peel_0-11/20240627231828/models/model_epoch_1000.pth

# CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config training_config/[NAME_OF_CONFIG].json
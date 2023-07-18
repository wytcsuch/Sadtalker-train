from src.utils.safetensor_helper import load_x_from_safetensor
import safetensors  
import safetensors.torch


org = '/home/yckj3822/img2video/SadTalker-main/checkpoints/SadTalker_V0.0.2_256.safetensors'
new = '/home/yckj3822/img2video/SadTalker-main/result_pose/64pretrain/ep400_iter4000.safetensors'

org = safetensors.torch.load_file(org)
new = safetensors.torch.load_file(new)
add = {}
for key in org:
    print(key)
    if key not in new:
        add[key] = org[key]
new.update(add)
safetensors.torch.save_file(new, 'latest.safetensors')  #保存
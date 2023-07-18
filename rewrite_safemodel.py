from src.utils.safetensor_helper import load_x_from_safetensor
import safetensors  
import safetensors.torch


org = './checkpoints/SadTalker_V0.0.2_256.safetensors'
new = 'your train model'

org = safetensors.torch.load_file(org)
new = safetensors.torch.load_file(new)
add = {}
for key in org:
    print(key)
    if key not in new:
        add[key] = org[key]
new.update(add)
safetensors.torch.save_file(new, 'latest.safetensors') 
import torch
from collections import OrderedDict
from models import *

def load_model(model_name, ckpt_path, device="cuda"):
    network = eval(model_name.replace('-', '_'))()
    network.to(device)
    network.eval()

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['state_dict']

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v  # remove 'module.'

    network.load_state_dict(new_state_dict)
    return network

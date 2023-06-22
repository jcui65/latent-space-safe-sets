import torch
import numpy as np

#import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '/home/cuijin/Project6remote/latent-space-safe-sets')

#from latentsafesets.utils.arg_parser_reacher import parse_args
#import latentsafesets.utils as utils
#from latentsafesets.utils.arg_parser_push import parse_args
#params = parse_args()#
gpuno=0#2#3#1#params['gpunumber']#
if gpuno==0:
    TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
elif gpuno==1:
    TORCH_DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
elif gpuno==2:
    TORCH_DEVICE = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
elif gpuno==3:
    TORCH_DEVICE = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')

def torchify(x):
    if type(x) is not torch.Tensor and type(x) is not np.ndarray:
        x = np.array(x)
    if type(x) is not torch.Tensor:
        x = torch.FloatTensor(x)
    return x.to(TORCH_DEVICE)


def to_numpy(x):
    if x is None:
        return x
    return x.detach().cpu().numpy()

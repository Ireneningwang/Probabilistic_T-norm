import torch
from tnorms.tnorms_binary import t_norm_and_conditional_ratio

def t_norm_multi_and_conditional_ratio(pxs, gammas ):
    if len(pxs) == 2:
        t_norm_and_conditional_ratio(pxs[0], pxs[1], gammas[0])
        
    q = t_norm_and_conditional_ratio(pxs[-2], pxs[-1], gammas[-1])
    new_pxs = torch.cat((pxs[:-2],q)) # this needs to be worked out
    new_gammas = gammas[:-1] # this may need some modification for tensors
    return t_norm_multi_and_conditional_ratio(new_pxs, new_gammas )
import torch
from importlib import reload
# Load existing functions from other files
import tyche.tnorms.tnorms_binary
reload(tyche.tnorms.tnorms_binary)
from tyche.tnorms.tnorms_binary import get_rho_pearsons_r
from tyche.tnorms.tnorms_binary import get_py1_pearsons_r
from tyche.tnorms.tnorms_binary import corr_bounds_old
from tyche.tnorms.tnorms_binary import corr_bounds_pearsonsr
from tyche.tnorms.tnorms_binary import py1_lowerbound
from tyche.tnorms.tnorms_binary import py1_upperbound
from tnorm_activation import latent_tnorm_module



# tests (to be absorbed into unit test module)
def unit_test():
    in_features = (2,3)
    a = torch.rand(in_features) # create a matrix with shape 2*3 (valuses all in (0,1)), to represent the probailities of xs and ys
    pX = a[0]
    pY = a[1]
    pinvX = 1-pX
    pinvY = 1-pY
    print(a, pX, pinvX)
    for method in ['interpolation_ratio', 'conditional_ratio', 'pearsons_r']:
        print(f"Testing:\n\tmethod = {method}")
        for expression in ['X & Y', '~X & Y', 'X & ~Y', '~X & ~Y', 'X | Y', '~X | Y', 'X | ~Y', '~X | ~Y', 'X', 'Y', '~X', '~Y']: 
            for gamma in [0.0,0.5,1.0]: # if add 0.25/ 0.75, unknown target for gamma
                beta = gamma
                rho = 2*gamma-1
                # print(f"Testing:\n\tmethod = {method}, expression = {expression}, gamma={gamma}, beta={beta}, rho={rho}")
                tnorm = latent_tnorm_module(gamma=gamma, beta=beta, rho=rho, method=method, expression=expression)
                out = tnorm()
            
                if expression == 'X & Y':
                    if gamma == 0.0:
                        test = torch.maximum(pX+pY-1, torch.tensor(0.))
                    elif gamma == 0.5:
                        test = pX*pY
                    elif gamma == 1.0:
                        test = torch.minimum(pX,pY)
                    else:
                        raise ValueError(f"Unknown target value for gamma  = {gamma}")
                elif expression == '~X & Y':
                    if gamma == 0.0:
                        test = torch.minimum(pinvX,pY)
                    elif gamma == 0.5:
                        test = pinvX * pY
                    elif gamma == 1.0:
                        test = torch.maximum(pinvX+pY-1, torch.tensor(0.))
                    else:
                        raise ValueError(f"Unknown target value for gamma  = {gamma}")
                elif expression == 'X & ~Y':
                    if gamma == 0.0:
                        test = torch.minimum(pX,pinvY)
                    elif gamma == 0.5:
                        test = pX * pinvY
                    elif gamma == 1.0:
                        test = torch.maximum(pX+pinvY-1, torch.tensor(0.))
                    else:
                        raise ValueError(f"Unknown target value for gamma  = {gamma}")
                elif expression == '~X & ~Y':
                    if gamma == 0.0:
                        test = torch.maximum(pinvX+pinvY-1, torch.tensor(0.))
                    elif gamma == 0.5:
                        test = pinvX*pinvY
                    elif gamma == 1.0:
                        test = torch.minimum(pinvX,pinvY)
                    else:
                        raise ValueError(f"Unknown target value for gamma  = {gamma}")
                elif expression == 'X | Y':
                    if gamma == 0.0:
                        test = 1 - torch.maximum(pinvX+pinvY-1, torch.tensor(0.))
                    elif gamma == 0.5:
                        test = 1 - pinvX * pinvY
                    elif gamma == 1.0:
                        test = 1 - torch.minimum(pinvX,pinvY)
                    else:
                        raise ValueError(f"Unknown target value for gamma  = {gamma}")
                elif expression == '~X | Y':
                    if gamma == 0.0:
                        test = 1 - torch.minimum(pX,pinvY)
                    elif gamma == 0.5:
                        test = 1 - pX * pinvY
                    elif gamma == 1.0:
                        test = 1 - torch.maximum(pX+pinvY-1, torch.tensor(0.))
                    else:
                        raise ValueError(f"Unknown target value for gamma  = {gamma}")
                elif expression == 'X | ~Y':
                    if gamma == 0.0:
                        test = 1 - torch.minimum(pinvX,pY)
                    elif gamma == 0.5:
                        test = 1 - pinvX * pY
                    elif gamma == 1.0:
                        test = 1 - torch.maximum(pinvX+pY-1, torch.tensor(0.))
                    else:
                        raise ValueError(f"Unknown target value for gamma  = {gamma}")
                elif expression == '~X | ~Y':
                    if gamma == 0.0:
                        test = 1 - torch.maximum(pX+pY-1, torch.tensor(0.))
                    elif gamma == 0.5:
                        test = 1 - pX * pY
                    elif gamma == 1.0:
                        test = 1 - torch.minimum(pX,pY)
                    else:
                        raise ValueError(f"Unknown target value for gamma  = {gamma}")
                elif expression == 'X':
                    test = pX
                elif expression == 'Y':
                    test = pY
                elif expression == '~X':
                    test = pinvX
                elif expression == '~Y':
                    test = pinvY
                else:
                    raise ValueError(f"Unknown expression  = {expression}")
                print(f"out = {out}")
                print(f"test = {test}")
                # print(f"torch.allclose(out, test) = {torch.allclose(out, test)}\n\n")
                try:
                    assert(torch.allclose(out, test))
                except:
                    print(f"Fails for:\n\tmethod = {method}, expression = {expression}, gamma={gamma}, beta={beta}, rho={rho}")
                
    # the above code with throw a ValueError or AssertionError



def upper_lower_bound():
    py = torch.tensor(0.7)
    px = torch.tensor(0.6)
    py1 = torch.tensor(0.8)
    print(f"px = {px:.3f}")
    print(f"py = {py:.3f}")
    print(f"py1 = {py1:.3f}")
    print(f"get_rho(px, py, py1) = {get_rho_pearsons_r(px, py, py1):.3f}")
    print(f"get_py1(px, py, get_rho(px, py, py1)) = {get_py1_pearsons_r(px, py, get_rho_pearsons_r(px, py, py1))}")
    print(f"corr_bounds_old(px, py) = {corr_bounds_old(px, py)}")
    print(f"corr_bounds_old(py, px) = {corr_bounds_old(py, px)}")
    print(f"corr_bounds(px, py) = {corr_bounds_pearsonsr(px, py)}")
    print(f"corr_bounds(py, px) = {corr_bounds_pearsonsr(py, px)}")

    print("\nGrids")
    px_vec = torch.tensor([0.25, 0.5, 0.75])
    py_vec = torch.tensor([0.25, 0.5, 0.75])
    px, py = torch.meshgrid(px_vec, py_vec, indexing='ij')
    print(f"px = {px}")
    print(f"py = {py}")
    rho=-0.7
    print(f"rho = {rho}")

    py1_lb = py1_lowerbound(px, py)
    print(f"py1_lb = {py1_lb}")
    py1_ub = py1_upperbound(px, py)
    print(f"py1_ub = {py1_ub}")
    py1 = get_py1_pearsons_r(px, py, rho)
    print(f"py1 = {py1}")
    rhos = get_rho_pearsons_r(px, py, py1)
    print(f"rhos = {rhos}")



if __name__ == "__main__":
    unit_test()
    upper_lower_bound()
    # if there are problems with the implementation
    print("\nAll tests passed!")
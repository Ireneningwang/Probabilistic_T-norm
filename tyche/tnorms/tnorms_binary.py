import torch

def py1_lowerbound(px, py):
    """
    Get the lowerbound(s) for p(Y=1|X=1) given p(X=1) and p(Y=1)
    
    parameters
    ----------
    px - probability or array of probabilities for p(X=1)
    py - probability or array of probabilities for p(Y=1)
    
    returns
    -------
    py1 - minimum value(s) for p(Y=1|X=1)
    """
    return torch.maximum(torch.tensor([0]),(px + py - 1)/px)
    
def py1_upperbound(px, py):
    """
    Get the upper bound(s) for p(Y=1|X=1) given p(X=1) and p(Y=1)
    
    parameters
    ----------
    px - probability or array of probabilities for p(X=1)
    py - probability or array of probabilities for p(Y=1)
    
    returns
    -------
    py1 - maximum value(s) for p(Y=1|X=1)
    """
    return torch.minimum(torch.tensor([1]),py/px)
    
## interpolation ratio
def t_norm_and_interpolation_ratio(px, py, gamma):
    """
    parameters
    ----------
    px - the array of x probabilities
    py - the array of y probabilities
    gamma - a fraction which tells us how far between the extreme values
        the parametrised T-norm sits. gamma == 0.5 gives the product T-norm
        gamma > 0.5 mixes product T-norm with Godel T-norm
            (2*gamma-1)*T_prod + (2-2*gamma)*T_God
        gamma < 0.5 mixes Lukawickz T-norm with product T-norm
            2*gamma*T_luk + (1-2*gamma)*T_prod

    returns
    -------
    pxy - array of t-norm values that recreate the probability p(x=1, y=1|d)
        for each element in px and py, assuming the distributional form above
    """
    # the middle point of the and p(x) and p(y) independent
    pandsmid = px*py    # Product T-norm
    if gamma >= 0.5:
        # alphaplus = 2*(gamma - 0.5)
        # alphaminus = 0
        # the upper bound of the and
        pandsub = torch.minimum(px, py) # Godel T-norm
        # a weighted average of the upper bound and midpoint
        # -----------Irene's alternative-------------
        # pands = (2*gamma-1)*px*py  + (2-2*gamma)*torch.minimum(px, py)
        # pands = alphaplus*pandsub + (1-alphaplus)*pandsmid
        # -----------Corrected -------------
        pands = (2*gamma-1)*pandsub + (2-2*gamma)*pandsmid
    else:
        # alphaplus = 0
        # alphaminus = 2*(0.5 - gamma)
        # the lower bound of the and
        pandslb = torch.maximum(px + py - 1, torch.tensor(0))   # Lukasiewicz T-norm
        # a weighted average of the midpoint and the lower bound
        # -----------Irene's alternative -------------
        pands = 2*gamma*torch.maximum(px + py - 1, torch.tensor(0)) + (1-2*gamma)*px*py
        # -----------WRONG-------------
        # pands = alphaminus*pandslb + (1 - alphaminus)*pandsmid
        # -----------Corrected -------------
        pands = (1-2*gamma)*pandslb + 2*gamma*pandsmid 
    return pands
 
def t_norm_or_interpolation_ratio(px, py, gamma):
    return 1- t_norm_and_interpolation_ratio(1-px, 1-py, gamma)
    
## Pearson's r interpolation
    
def get_py1_pearsons_r(px, py, rho):
    """
    Get the corresponding p(Y=1|X=1) given p(X=1), p(Y=1) and 
    desired Pearson's r correlation coefficient rho. If this is 
    not feasible then return the p(Y=1|X=1) corresponding to the
    nearest valid correlation value to the supplied rho.
    
    parameters
    ----------
    px - probability or array of probabilities for p(X=1)
    py - probability or array of probabilities for p(Y=1)
    rho - desired Pearson's r correlation coefficient 
    
    returns
    -------
    py1 - probability or array of probabilities for p(Y=1|X=1)
    """
    ox = px/(1-px)
    oy = py/(1-py)
    py1_raw = (rho/torch.sqrt(ox*oy)+1)*py
    py1_lb = py1_lowerbound(px, py)
    py1_ub = py1_upperbound(px, py)
    return torch.maximum(py1_lb, torch.minimum(py1_raw, py1_ub))


def get_rho_pearsons_r(px, py, py1):
    """
    Get the Pearson's r correlation coefficient for specified
    marginals p(X=1), p(Y=1) and conditional p(Y=1|X=1).
    Currently does not check whether py1 is valid (can add this later).
    
    parameters
    ----------
    px - probability or array of probabilities for p(X=1)
    py - probability or array of probabilities for p(Y=1)
    py1 - probability or array of probabilities for p(Y=1|X=1)
    
    returns
    -------
    rho - Pearson's r correlation coefficient if the specified distribution is valid
    """
    ox = px/(1-px)
    oy = py/(1-py)
    return torch.sqrt(ox*oy)*(py1/py - 1)


def corr_bounds_old(px, py):
    if px >= 0.5:
        if px >= py:
            return get_rho_pearsons_r(1-px, 1- py, 0), get_rho_pearsons_r(py, px, 1)
        return get_rho_pearsons_r(1-py, 1-px, 0), get_rho_pearsons_r(px, py, 1)
    else:
        if px >= py:
            return get_rho_pearsons_r(px, py, 0), get_rho_pearsons_r(1-px, 1-py, 1)
        return get_rho_pearsons_r(py, px, 0), get_rho_pearsons_r(1-py, 1-px, 1)


def corr_bounds_pearsonsr(px, py):
    """
    Get the bounds on valid values for the Pearson's r 
    correlation coefficient for specified
    marginals p(X=1), p(Y=1).
    
    parameters
    ----------
    px - probability or array of probabilities for p(X=1)
    py - probability or array of probabilities for p(Y=1)
    
    returns
    -------
    rho_lb - lower bound(s) on Pearson's r correlation coefficient
    rho_ub - upper  bound(s) on Pearson's r correlation coefficient
    """
    py1_lb = py1_lowerbound(px, py)
    py1_ub = py1_upperbound(px, py)
    return get_rho_pearsons_r(px, py, py1_lb), get_rho_pearsons_r(px, py, py1_ub)

def t_norm_and_pearsons_r(px, py, rho):
    """
    parameters
    ----------
    px - the numpy array of x probabilities
    py - the numpy array of y probabilities
    rho - a fraction which tells us how far between the extreme values
        p(x,y) sits given that we know p(x) and p(y)

        assume d is rho 
        if d>(1-d)
        if d < (1-d)

    returns
    -------
    pxy - array of t-norm values that recreate the probability p(x=1, y=1|d)
        for each element in px and py, assuming the distributional form above
    """
    py1s = get_py1_pearsons_r(px, py, rho)
    return px * py1s
 
def t_norm_or_pearsons_r(px, py, rho):
    return 1- t_norm_and_pearsons_r(1-px, 1-py, rho)
## Conditional ratio 

def get_py1_conditional_ratio(px, py, gamma):
    py1_lb = py1_lowerbound(px, py)
    py1_ub = py1_upperbound(px, py)
    try:
        py1_raw = (gamma/(1-gamma)) * py
    except:
        # fails if gamma = 1 , so set to max
        py1_raw = py1_ub
    return torch.maximum(py1_lb, torch.minimum(py1_raw, py1_ub))

def t_norm_and_conditional_ratio(px, py, gamma):
    """
    parameters
    ----------
    px - the numpy array of x probabilities, p(X=1)
    py - the numpy array of y probabilities, p(Y=1)
    gamma - a fraction from which the odds-ratio is created
        py1 ( p(Y=1|X=1) is calculated as
            max(py1_lb, min(py1_ub, ((1-gamma)/gamma)*py))
        this is the conditional ratio value constrained within the 
        range [py1_lb, py1_ub] where py1_lb and py1_ub are the
        lower and upper bound values for py1 given px and py
    
    returns
    -------
    pands - array of t-norm values that recreate the probability 
        p(x=1, y=1|gamma) for each element in px and py, assuming the
        distributional form above
    """
    py1 = get_py1_conditional_ratio(px, py, gamma)
    pands = px*py1
    return pands
 
def t_norm_or_conditional_ratio(px, py, gamma):
    """
    px - the numpy array of x probabilities, p(X=1)
    py - the numpy array of y probabilities, p(Y=1)
    gamma - a fraction from which the odds-ratio is created
        py1 ( p(Y=1|X=1) is calculated as
            max(py1_lb, min(py1_ub, ((1-gamma)/gamma)*py))
        this is the conditional ratio value constrained within the 
        range [py1_lb, py1_ub] where py1_lb and py1_ub are the
        lower and upper bound values for py1 given px and py
    
    returns
    -------
    pors - array of t-norm values that recreate the probability
        1-p(x=0, y=0|gamma) for each element in px and py, assuming the
        distributional form above
    """
    return 1- t_norm_and_conditional_ratio(1-px, 1-py, gamma)
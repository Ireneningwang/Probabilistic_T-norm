# Load external packages
import torch
import sympy
from sympy.parsing.sympy_parser import parse_expr
import torch.nn as nn
from torch.nn.parameter import Parameter

# Load existing functions from other files
from importlib import reload
import tyche.tnorms.tnorms_binary
reload(tyche.tnorms.tnorms_binary)
from tyche.tnorms.tnorms_binary import t_norm_and_interpolation_ratio
from tyche.tnorms.tnorms_binary import t_norm_or_interpolation_ratio
from tyche.tnorms.tnorms_binary import t_norm_and_conditional_ratio
from tyche.tnorms.tnorms_binary import t_norm_or_conditional_ratio
from tyche.tnorms.tnorms_binary import t_norm_and_pearsons_r
from tyche.tnorms.tnorms_binary import t_norm_or_pearsons_r


# this is a prototype pytorch module to implement a single probabilistic logic operation.
class latent_tnorm_module(nn.Module):
    '''
    Implementation of tnorm activation.
    Shape:
        - Input: (2,N, *) where * means, any number of additional dimensions
        - Output: (N, *), same shape as the input in the trailing array dimensions
    Parameters:
        - gamma - trainable parameter for interpolation tnorm
        - rho - trainable parameter for pearsons r tnorm
        - beta - trainable parameter for conditional tnorm
    References:
        - See probabilistic_tnorms_foundations.ipynb
    Examples:
        >>> 
    '''
    
    @staticmethod
    def left_path_pos(x):
        # return x[0]
        return x
    @staticmethod
    def left_path_neg(x):
        # return 1-x[0]
        return 1-x
    @staticmethod
    def right_path_pos(x):
        # return x[1]
        return x
    @staticmethod
    def right_path_neg(x):
        # return 1- x[1]
        return 1-x
    @staticmethod
    def left_only(lhs, rhs, dependency):
        return lhs
    @staticmethod
    def right_only(lhs, rhs, dependency):
        return rhs


    def invert_dependency(self):
        self.dep_invert = not self.dep_invert

    def apparent_dependency(self):
        if not self.dep_invert:
            return self.dependency
        if self.method == 'pearsons_r':
            return -self.dependency
        else:
            return 1 - self.dependency 

            
    def __init__(self, gamma=None, rho=None, beta=None, method='interpolation_ratio', expression='X | Y', learn_dependency=False):
        '''
        Initialization.
        INPUT:
            - core_features: shape of the input barring the duplex nature. 
        '''
        super(latent_tnorm_module, self).__init__()
        self.px = Parameter(data=torch.tensor(0.5))
        self.py = Parameter(data=torch.tensor(0.5))
        # self.in_features = in_features
        # assert(in_features[0] == 2)
        # self.out_features = in_features[:1]
        self.method = method
        # now deal with method
        if method == 'interpolation_ratio':
            self.and_func = t_norm_and_interpolation_ratio
            self.or_func = t_norm_or_interpolation_ratio
            # initialize dependency from gamma
            if gamma == None:
                self.dependency = Parameter(torch.tensor(0.0))
            else:
                # create a tensor out of the input
                self.dependency = Parameter(torch.tensor(gamma)) 
        elif method == 'pearsons_r':
            self.and_func = t_norm_and_pearsons_r
            self.or_func = t_norm_or_pearsons_r
            # initialize dependency from rho
            if rho == None:
                self.dependency = Parameter(torch.tensor(0.0))
            else:
                # create a tensor out of the input
                self.dependency = Parameter(torch.tensor(rho)) 
        elif method == 'conditional_ratio':
            self.and_func = t_norm_and_conditional_ratio
            self.or_func = t_norm_or_conditional_ratio
            # initialize bata
            if beta == None:
                self.dependency = Parameter(torch.tensor(0.0))
            else:
                # create a tensor out of the input
                self.dependency = Parameter(torch.tensor(beta)) 
        else:
            raise ValueError(f"Unrecognised tnorm method = {method}")

        self.set_expression(expression)
        if learn_dependency:
            # set requiresGrad to true
            self.dependency.requiresGrad = True 


    def set_expression(self, expression):
        # this deals with the direction in which correlation affects the probability
        self.dep_invert = False
        # define logic symbols to use later
        X, Y = sympy.symbols('X,Y')
         # convert expression string to sympy Expr object for ease of comparison (if passedas string)
        if type(expression) == type(''):
            self.expression = parse_expr(expression)
        # if expression contains a negative instance of X
        # then invert X probabilities
        if (self.expression.subs({Y:True}) == ~X) or (self.expression.subs({Y:False}) == ~X):
            self.left_path = self.left_path_neg
            self.invert_dependency()
        else:
            self.left_path = self.left_path_pos
        # if expression contains a negative instance of Y,  then invert Y probabilities
        if (self.expression.subs({X:True}) == ~Y) \
                or (self.expression.subs({X:False}) == ~Y):
            self.right_path = self.right_path_neg
            self.invert_dependency()
        else:
            self.right_path = self.right_path_pos
        # now form the 
        if type(self.expression) == sympy.logic.boolalg.And:
            # out probability is of conjoint binary RVs
            self.tnorm_func = self.and_func
        elif type(self.expression) == sympy.logic.boolalg.Or:
            # out probability is of disjoint binary RVs
            self.tnorm_func = self.or_func
        elif self.expression.equals(X) or (~X).equals(self.expression):
            # out probability simply preserves lhs probabily 
            self.tnorm_func = self.left_only
        elif self.expression.equals(Y) or (~Y).equals(self.expression):
            # out probability simply preserves rhs probabily 
            self.tnorm_func = self.right_only
            
            
    def forward(self):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        return self.tnorm_func(self.left_path(self.px), self.right_path(self.py), self.apparent_dependency())

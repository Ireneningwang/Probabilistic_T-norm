import random
import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
import sympy
from sympy.parsing.sympy_parser import parse_expr



# Construct a Bayesian network model for X and Y
def generate_model_input():
    nodes = ['X', 'Y']
    edges = [ (0, 1) ]
    adjacency = np.zeros((2,2),dtype=int)
    for i,j in edges: 
        adjacency[i,j] = 1
    edges_str = [(nodes[parent], nodes[child]) for (parent, child) in edges]
    model = BayesianNetwork(edges_str)
    for nodeid, parents in enumerate(adjacency.T):
        # set randomly generated conditional probability distribution to X and Y
        # from single P(X) to P(Y|X)
        cpd = create_random_binary_cpd(nodes, nodeid, parents)
        print(cpd)
        print(f"cpd.variable = {cpd.variable}")
        print(f"cpd.get_evidence() = {cpd.get_evidence()}")
        # Associating the CPDs with the network
        model.add_cpds(cpd)
    # check_model checks for the network structure and CPDs and verifies that the CPDs are correctly 
    # defined and sum to 1.     
    print("Model check result: ", model.check_model())
    return model


# Add CPD(Conditional Probability Distribution) to the Bayesian Model
def create_random_binary_cpd(nodes, nodeid, parents):   
    num_parents = np.sum(parents)
    num_cols = 2**num_parents
    values = np.empty((2,num_cols))
    values[0,:] = np.random.random((num_cols,))
    values[1,:] = 1 - values[0,:]
    evidence = [nodes[k] for k, a_parent in enumerate(parents) if a_parent]
    if len(evidence) == 0:
        cpd = TabularCPD(
            variable=nodes[nodeid], variable_card=2, values=values)
    else:
        evidence_card = [2 for _ in evidence]
        cpd = TabularCPD(
            variable=nodes[nodeid], variable_card=2, values=values,
            evidence=evidence,  evidence_card=evidence_card)
    return cpd


# Generate N random logical correct experssions 
# In fact, for two variables X and Y, there can be four combinations in total. 
# So the generate_model_data is to randomly select N combination from the four based on the probability distribution of X and Y 
# For example, p(X=0) = 0.0266134 , p(X=1) = 0.973387, then in the 10 sets of generated data, no X=1, nearly all equals to 
def generate_model_data(model, N, expr_strs):
    X, Y = sympy.symbols('X,Y')
    raw_data = model.simulate(N)
    expr_data = raw_data.copy()[[]]
    for expr_str in expr_strs:
        expr = parse_expr(expr_str)
        for Xval in [False, True]:
            for Yval in [False, True]:
                if expr.subs({X:Xval, Y:Yval}):
                    expr_data.loc[(raw_data['X']==Xval) & (raw_data['Y']== Yval), expr] = int(1)
                else:
                    expr_data.loc[(raw_data['X']==Xval) & (raw_data['Y']== Yval), expr] = int(0)
    expr_data = expr_data.astype(int)
    for k in expr_data.keys():      # there exists two situations that X is after Y
        if str(k) == 'Y & ~X':
            expr_data.rename(columns={k:'~X & Y'}, inplace = True)
        elif str(k) == 'Y | ~X':
            expr_data.rename(columns={k:'~X | Y'}, inplace = True)
    return expr_data



# For reference we can use sympy to construct the logic table
# It is the general logical truth table for X and Y, not related to the probability of X and Y
def refernce_table():
    X, Y = sympy.symbols('X,Y')
    # 10 here, 'X' and 'Y' will be add into the final logical table
    expr_strs = ['~X', '~Y', 'X & Y', '~X & Y', 'X & ~Y', '~X & ~Y', 'X | Y', '~X | Y', 'X | ~Y', '~X | ~Y']    
    logic_table = pd.DataFrame([[0,0],[0,1],[1,0],[1,1]], columns=['X','Y'])
    for expr_str in expr_strs:
        expr = parse_expr(expr_str)
        for Xval in [False, True]:
            for Yval in [False, True]:
                if expr.subs({X:Xval, Y:Yval}):
                    logic_table.loc[(logic_table['X']==Xval) & (logic_table['Y']== Yval), expr_str] = 1
                else:
                    logic_table.loc[(logic_table['X']==Xval) & (logic_table['Y']== Yval), expr_str] = 0
    logic_table = logic_table.astype(int)
    return logic_table


def generate(N=20, num_exprs=5):
    """
    """
    model = generate_model_input()
    expressions = ['~X', '~Y', 'X & Y', '~X & Y', 'X & ~Y', '~X & ~Y', 'X | Y', '~X | Y', 'X | ~Y', '~X | ~Y']
    # generate subset of size num_exprs from 10 logical expressions 
    expre_sample = random.sample(expressions, num_exprs)    
    data = generate_model_data(model, N, expre_sample)
    logic_table = refernce_table()
    cpds = model.get_cpds()
    px = cpds[0].values[0]
    py1, py0 = cpds[1].values[0,:]
    print(f"cpds[0].values = {cpds[0].values}") # for single probability of p(X)
    print(f"cpds[1].values = {cpds[1].values}") # for the probability of p(Y|X)
    print("\n Generated  data: \n",data)
    print("\n The referencng logic table: \n", logic_table )
    return (px, py1, py0), data

if __name__ == "__main__":
    generate()

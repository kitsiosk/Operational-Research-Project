#!/usr/bin/env python
# coding: utf-8
from __future__ import division
from pyomo.environ import *
import numpy as np


# ## Model
# Define concrete model
model = ConcreteModel()


# ## Variables
# I ranges from 1 to 18
model.N = 18
model.I = RangeSet(1, model.N)

# II ranges from 1 to 9
model.n = 9
model.II = RangeSet(1, model.n)

model.J = RangeSet(1, model.n - 1)
# x is the weekly supply
model.x = Var(model.I, domain=NonNegativeIntegers)
# m is the max between the two types each week
model.m = Var(model.II, domain=NonNegativeIntegers)
# b is boolean: 0=> typeA | 1=> typeB
model.b = Var(model.II, domain=Binary)
# x is boolean: The XOR of b_i and b_(i-1) implemented with max
model.c = Var(model.II, domain=Binary)


# ## Parameters

model.y = np.array([-1, 55, 55, 44, 0, 45, 45, 36, 35, 35, 38, 38, 30, 0, 48, 48, 58, 57, 58])

# k is the cost
coeff = np.zeros((19))
for i in range(1, 10):
    coeff[i] = 225
    coeff[9+i] = 310
model.k = coeff


# ## Objective
def get_objective(model):
    obj = sum(
        model.k[i]*model.x[i] + model.k[i+9]*model.x[i+9] + 
        model.c[i]*500 + 
        (sum(model.x[k] - model.y[k] for k in range(1, i+1)) + 125)*225*0.195/52 +  
        (sum(model.x[k+9] - model.y[k+9] for k in range(1, i+1)) + 143)* 310*0.195/52
        for i in range(1, model.n)
    )
    return obj

model.OBJ = Objective(rule=get_objective)


# ## Constraints

# Big-M value
M = 10e5
# Below constraings are called with I=1,..,9
def maxConstraint1(model, i):
    return model.m[i] >= model.x[i]
def maxConstraint2(model, i):
    return model.m[i] >= model.x[i+9]
def maxConstraint3(model, i):
    return model.m[i] <= model.x[i] + M*model.b[i]
def maxConstraint4(model, i):
    return model.m[i] <= model.x[i+9] + M*(1 - model.b[i])
def maxConstraint5(model, i):
    return model.m[i] == model.x[i] + model.x[i+9]

# Upper bounds
# Below constraings are called with I=1,..,9
def ubConstraint1(model, i):
    return model.x[i] <= 100
def ubConstraint2(model, i):
    return model.x[i+9] <= 80


# Stock rules
def stockConstraint1(model, i):
    return sum(model.x[k] - model.y[k] for k in range(1, i+1)) + 125 >= model.y[i+1]
def stockConstraint2(model, i):
    return sum(model.x[k] - model.y[k] for k in range(10, i + 10)) + 143 >= model.y[10+i]

# The below constraints set  c[i] = max{b[i], b[i-1]}
# to be used only inside minimization objective
def xorConstraint1(model, i):
    if i==1:
        # Take care of b[0] = 0
        return model.c[i] >= model.b[i] - 0
    return model.c[i] >= model.b[i] - model.b[i-1]
def xorConstraint2(model, i):
    if i==1:
    # Take care of b[0] = 0
        return model.c[i] >= -model.b[i]
    return model.c[i] >= model.b[i-1] - model.b[i]


model.maxConstraint1 = Constraint(model.II, rule=maxConstraint1)
model.maxConstraint2 = Constraint(model.II, rule=maxConstraint2)
model.maxConstraint3 = Constraint(model.II, rule=maxConstraint3)
model.maxConstraint4 = Constraint(model.II, rule=maxConstraint4)
model.maxConstraint5 = Constraint(model.II, rule=maxConstraint5)

model.ubConstraint1 = Constraint(model.II, rule=ubConstraint1)
model.ubConstraint2 = Constraint(model.II, rule=ubConstraint2)

model.stockConstraint1 = Constraint(model.J, rule=stockConstraint1)
model.stockConstraint2 = Constraint(model.J, rule=stockConstraint2)

model.xorConstraint1 = Constraint(model.II, rule=xorConstraint1)
model.corConstraint2 = Constraint(model.II, rule=xorConstraint2)


opt = SolverFactory('glpk')
results = opt.solve(model)

print('Parameter values:')
for v in model.component_data_objects(Var):
  print(str(v), v.value)
  
print('Objective value:')
print(model.OBJ.expr())






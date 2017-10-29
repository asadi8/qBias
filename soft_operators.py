import numpy
import sys

def mellowmax(X,beta):
	type_check(X,beta)
	beta=float(beta)
	return numpy.log(numpy.mean(numpy.exp(beta * X)))/beta

def mellowmax_grad(X,beta):
	type_check(X,beta)
	return boltzmann_policy(X,beta)

def mellowmax_hessian_diag(X,beta):
	type_check(X,beta)
	bp=boltzmann_policy(X,beta)
	return beta * bp * (1-bp)

def boltzmann_policy(X,beta):
	type_check(X,beta)
	exps=numpy.exp(beta*X)
	return exps/numpy.sum(exps)

def type_check(X,beta):
	if type(X) is numpy.ndarray:
		pass
	else:
		print('X is not a numpy array')
		sys.exit(1)
	beta=float(beta)

ar=numpy.array([1,2,2,5,4.8])
ans=mellowmax(ar,0.75)
print(ans)
import numpy
import sys
import math

def mellowmax(X,beta):
	type_check(X,beta)
	beta=float(beta)
	return numpy.log(numpy.mean(numpy.exp(beta * X)))/beta

def mellowmax_grad(X,beta):
	type_check(X,beta)
	return boltzmann_policy(X,beta)

def mellowmax_hessian_diag(X,beta):
	type_check(X,beta)
	while True:
		done=1
		bp=boltzmann_policy(X,beta)	
		for b in bp:
			if math.isnan(b):
				done=0
		beta=0.97*beta
		#print('backtracking beta ....')
		#print('new beta: ',beta)
		if done==1:
			break
	#sys.exit(1)
	out= beta * bp * (1-bp)
	return out

def boltzmann_policy(X,beta):
	type_check(X,beta)
	C=numpy.max(X)
	exps=numpy.exp(beta*(X))
	return exps/numpy.sum(exps)

def type_check(X,beta):
	if type(X) is numpy.ndarray:
		pass
	else:
		print('X is not a numpy array')
		sys.exit(1)
	beta=float(beta)

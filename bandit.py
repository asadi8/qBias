import tensorflow as tf
import numpy, sys
import matplotlib.pyplot as plt
import random
import soft_operators
import time
import neural_net
import bias_estimator


def get_next_batch(params):
	X = numpy.random.choice(xrange(params['num_arms']), (params['batch_size'],))
	Y_mean = []
	Y_std = []
	for x in X:
		Y_mean.append(params['true_values'][x])
		Y_std.append(params['values_std'][x])
	Y = numpy.array([numpy.random.normal(loc=x, scale=y) for x, y in zip(Y_mean, Y_std)])
	return X, Y

def plot(params,nn):
	X=[j for j in range(params['num_arms'])]
	means, sigmas = sess.run([nn.mu, nn.sigma],feed_dict={nn.action:X})
	Y_mean = params['true_values']
	Y_std = [params['const_std'] for x in X]

	plt.clf()
	plt.plot(means,label='means')
	plt.plot(sigmas,label='vars')
	plt.plot(Y_mean, label='True means')
	plt.plot(Y_std, label='True vars')
	plt.xlim([0,params['num_arms']])
	plt.ylim([-2,2])
	plt.legend()
	plt.hold(True)
	plt.pause(0.01)

params={}
params['width']=20
params['alpha']=0.05
params['batch_size']=64
params['num_batches']=20
params['num_hidden_layers']=1
params['num_arms'] = 50
params['embed_size'] = 5
params['const_std'] = .75
params['max_experiments']=1000
params['actions_list']=[j for j in range(params['num_arms'])]
#params['true_values']=[numpy.cos(a) for a in params['actions_list']]
params['true_values']=[numpy.tanh(0.1*a) for a in params['actions_list']]
params['values_std']=[params['const_std'] for a in params['actions_list']]
params['temperature']=75
max_expectation=numpy.max(params['true_values'])

li_default_bias=[]
li_taylor_bias=[]
plt.ion()

for experiment in range(1,params['max_experiments']):
	tf.reset_default_graph()
	numpy.random.seed(100+experiment)
	random.seed(100+experiment)
	tf.set_random_seed(100+experiment)
	with tf.Session() as sess:
		params['session']=sess
		nn=neural_net.q_network(params)
		bs=bias_estimator.bias_estimator(params)

		for j in range(params['num_batches']):
			X, Y=get_next_batch(params)
			bs.increment_counts(X)
			nn.fit(X,Y)
		plot(params,nn)
		estimated_bias=bs.compute_bias(nn)
		expectation_max=nn.get_max()
		
		default_bias=expectation_max-max_expectation
		taylor_bias=expectation_max-max_expectation-estimated_bias

		li_default_bias.append(default_bias)
		li_taylor_bias.append(taylor_bias)

	if experiment %20==0:
		print(experiment)
		print('average default bias on {} experiments is {}'.format(experiment,numpy.mean(li_default_bias)))
		print('average taylor bias on {} experiments is {}'.format(experiment,numpy.mean(li_taylor_bias)))



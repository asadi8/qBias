import tensorflow as tf
import numpy, sys
import matplotlib.pyplot as plt
import random
import soft_operators
import time

class q_network:
	def __init__(self,params):
		self.num_actions=params['num_arms']
		self.embed_size=params['embed_size']
		self.width=params['width']
		self.alpha=params['alpha']
		self.network()
		return

	def network(self):
	    self.action = tf.placeholder(tf.int32, [None])
	    self.output = tf.placeholder(tf.float32, [None])

	    self.E = tf.get_variable('E', [self.num_actions, self.embed_size])
	    emb = tf.nn.embedding_lookup(self.E, self.action)
	    h1 = emb
	    for _ in range(params['num_hidden_layers']):
	    	h1 = tf.contrib.layers.fully_connected(inputs=h1, num_outputs=self.width, activation_fn=tf.nn.relu)

	    out = tf.contrib.layers.fully_connected(inputs=h1, num_outputs=2, activation_fn=None,biases_initializer=tf.zeros_initializer())
	    self.mu, self.sigma = out[:, 0], out[:, 1]
	    self.sigma = tf.nn.softplus(self.sigma) + 1e-5

	    self.distro = tf.contrib.distributions.Normal(mu=self.mu, sigma=self.sigma)
	    self.loss = tf.reduce_mean(-self.distro.log_prob(self.output))
	    
	    
	    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha)
	    self.train_op = self.optimizer.minimize(self.loss)


def get_next_batch(params):
	X = numpy.random.choice(xrange(params['num_arms']), (params['batch_size'],))
	Y_mean = [numpy.sin(x) for x in X]
	Y_std = [params['const_std'] for x in X]
	Y = numpy.array([numpy.random.normal(loc=x, scale=y) for x, y in zip(Y_mean, Y_std)])
	return X, Y

def increment_counts(counts,X):
	for x in X:
		counts[x]=counts[x]+1
	print(counts)
	return counts

params={}
params['width']=20
params['alpha']=0.05
params['batch_size']=64
params['num_batches']=20
params['num_hidden_layers']=1
params['num_arms'] = 50
params['embed_size'] = 5
params['const_std'] = .5
params['max_experiments']=1000

li_bias=[]
li_modified_bias=[]
plt.ion()
print('exp_max is equal to 1 .... for now')
for experiment in range(params['max_experiments']):
	tf.reset_default_graph()
	numpy.random.seed(experiment)
	random.seed(experiment)
	tf.set_random_seed(experiment)
	with tf.Session() as sess:
		counts=params['num_arms']*[0]
		nn=q_network(params)
		sess.run(tf.global_variables_initializer())
		for j in range(params['num_batches']):
			X, Y=get_next_batch(params)
			counts=increment_counts(counts,X)
			_,l,sigma,mu=sess.run([nn.train_op,nn.loss,nn.sigma,nn.mu],feed_dict={nn.output:Y,nn.action:X})

		X=[j for j in range(params['num_arms'])]
		means, sigmas = sess.run([nn.mu, nn.sigma],feed_dict={nn.action:X})
		Y_mean = [numpy.sin(x) for x in X]
		Y_std = [params['const_std'] for x in X]

		plt.clf()
		plt.plot(means,label='means')
		plt.plot(sigmas,label='vars')
		plt.plot(Y_mean, label='True means')
		plt.plot(Y_std, label='True vars')
                plt.legend()
                plt.hold(True)
                plt.pause(0.01)
		
		#plt.show()
		#plt.close()
		
		expectation_max=numpy.max(means)
		
		#print(sigmas*sigmas)
		#print(soft_operators.mellowmax_hessian_diag(means,5))
		#sys.exit(1)
		variance=sigmas*sigmas
		mean_estimator_variance=variance/(numpy.array(counts))
		estimated_bias=numpy.sum(mean_estimator_variance*soft_operators.mellowmax_hessian_diag(means,40))
		max_expectation=1
		bias=expectation_max-max_expectation
		li_bias.append(bias)
		li_modified_bias.append(expectation_max-max_expectation-estimated_bias)
		print(bias,estimated_bias)
	print('average bias on {} experiments is {}'.format(experiment,numpy.mean(li_bias)))
	print('average modified bias on {} experiments is {}'.format(experiment,numpy.mean(li_modified_bias)))
	#sys.exit(1)




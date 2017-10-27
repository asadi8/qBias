import tensorflow as tf
import numpy, sys
import matplotlib.pyplot as plt
import random

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

params={}
params['width']=20
params['alpha']=0.05
params['batch_size']=64
params['num_batches']=500
params['num_hidden_layers']=1
params['num_arms'] = 20
params['embed_size'] = 5
params['const_std'] = 5.
params['max_experiments']=1000

li_bias=[]
for experiment in range(params['max_experiments']):
	tf.reset_default_graph()
	numpy.random.seed(experiment)
	random.seed(experiment)
	tf.set_random_seed(experiment)
	with tf.Session() as sess:
		nn=q_network(params)
		sess.run(tf.global_variables_initializer())
		for j in range(params['num_batches']):
			X, Y=get_next_batch(params)
			_,l,sigma,mu=sess.run([nn.train_op,nn.loss,nn.sigma,nn.mu],feed_dict={nn.output:Y,nn.action:X})
			if j%100==0:
				pass
				#print('loss:',l)
				#print('')

		X=[j for j in range(params['num_arms'])]
		means, sigmas = sess.run([nn.mu, nn.sigma],feed_dict={nn.action:X})
		Y_mean = [numpy.sin(x) for x in X]
		Y_std = [params['const_std'] for x in X]

		'''
		plt.plot(means,label='means')
		plt.plot(sigmas,label='vars')
		plt.plot(Y_mean, label='True means')
		plt.plot(Y_std, label='True vars')
		plt.legend()
		plt.show()
		plt.close()
		'''
		exp_max=numpy.max(means)
		#print(exp_max)
		li_bias.append(exp_max-1)
	print('average bias on {} experiments is {}'.format(experiment,numpy.mean(li_bias)))




import tensorflow as tf
import numpy

class q_network:
	def __init__(self,params):
		self.num_actions=params['num_arms']
		self.embed_size=params['embed_size']
		self.width=params['width']
		self.alpha=params['alpha']
		self.num_hidden=params['num_hidden_layers']
		self.actions_list=params['actions_list']
		self.network()
		self.sess=params['session']
		self.sess.run(tf.global_variables_initializer())
		return

	def network(self):
	    self.action = tf.placeholder(tf.int32, [None])
	    self.output = tf.placeholder(tf.float32, [None])

	    self.E = tf.get_variable('E', [self.num_actions, self.embed_size])
	    emb = tf.nn.embedding_lookup(self.E, self.action)
	    h1 = emb
	    for _ in range(self.num_hidden):
	    	h1 = tf.contrib.layers.fully_connected(inputs=h1, num_outputs=self.width, activation_fn=tf.nn.relu)

	    out = tf.contrib.layers.fully_connected(inputs=h1, num_outputs=2, activation_fn=None,biases_initializer=tf.zeros_initializer())
	    self.mu, self.sigma = out[:, 0], out[:, 1]
	    self.sigma = tf.nn.softplus(self.sigma) + 1e-5

	    self.distro = tf.contrib.distributions.Normal(mu=self.mu, sigma=self.sigma)
	    self.loss = tf.reduce_mean(-self.distro.log_prob(self.output))
	    
	    
	    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha)
	    self.train_op = self.optimizer.minimize(self.loss)

	def fit(self,X,Y):
		self.sess.run([self.train_op],feed_dict={self.output:Y,self.action:X})

	def get_max(self):
		mu=self.sess.run(self.mu,feed_dict={self.action:self.actions_list})
		return numpy.max(mu)
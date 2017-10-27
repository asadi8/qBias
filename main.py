import tensorflow as tf
import numpy, sys

class q_network:
	def __init__(self,params):
		self.state_size=params['state_size']
		self.output_size=params['output_size']
		self.width=params['width']
		self.alpha=params['alpha']
		self.network()
		return

	def network(self):
	    self.state = tf.placeholder(tf.float32, [None,self.state_size], "state")
	    self.output = tf.placeholder(tf.float32,[None,self.output_size],"output")
	    
	    self.mu=self.state
	    for _ in range(params['num_hidden_layers']):
			print('here')
			self.mu = tf.contrib.layers.fully_connected(
			    inputs=self.mu,
			    num_outputs=self.width,
			    activation_fn=tf.nn.relu,
			    )
	    self.mu = tf.contrib.layers.fully_connected(
	        inputs=self.mu,
	        num_outputs=1,
	        activation_fn=None,
	        )

	    self.sigma=self.state
	    for _ in range(params['num_hidden_layers']):
			print('here as well')
			self.sigma = tf.contrib.layers.fully_connected(	
			    inputs=self.sigma,
			    num_outputs=self.width,
			    activation_fn=tf.nn.relu,
			    )
	    self.sigma = tf.contrib.layers.fully_connected(
	        inputs=self.sigma,
	        num_outputs=1,
	        activation_fn=None,
	        )
	    
	    #self.sigma = tf.squeeze(self.sigma)
	    self.sigma = tf.nn.softplus(self.sigma) + 1e-5

	    self.distro = tf.contrib.distributions.Normal(mu=self.mu, sigma=self.sigma)
	    self.loss = tf.reduce_mean(-self.distro.log_prob(self.output))
	    
	    
	    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha)
	    self.train_op = self.optimizer.minimize(self.loss)


def get_next_batch(params):
	X=[numpy.random.rand(params['state_size']) for b in range(params['batch_size'])]
	Gold=[numpy.linalg.norm(x)+numpy.cos(x[0])+numpy.sin(x[1])+x[0]*x[1]+5*x[3]*x[4]*x[2] for x in X]
	Y=[]
	for x,g in zip(X,Gold):
		Y.append(g+x[0]*(numpy.random.rand()-0.5))
	bs=params['batch_size']
	return numpy.array(X).reshape(bs,params['state_size']),numpy.array(Y).reshape(bs,1),numpy.array(Gold).reshape(bs,1)

params={}
params['state_size']=5
params['output_size']=1
params['width']=50

params['alpha']=0.0025
params['batch_size']=64
params['num_batches']=20000
params['num_hidden_layers']=2

tf.reset_default_graph()
with tf.Session() as sess:
    nn=q_network(params)
    sess.run(tf.global_variables_initializer())
    for j in range(params['num_batches']):
    	X,Y,Gold=get_next_batch(params)
    	_,l,sigma,mu=sess.run([nn.train_op,nn.loss,nn.sigma,nn.mu],feed_dict={nn.output:Y,nn.state:X})
    	if j%100==0:
    		print('loss:',l)
    for z in range(params['batch_size']):
    	print(X[z],'mean error',mu[z][0]-Gold[z][0],'variance estimate',sigma[z][0])





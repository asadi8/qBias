import numpy, soft_operators

class bias_estimator:
	def __init__(self,params):
		self.counts=params['num_arms']*[0]
		self.num_arms=params['num_arms']
		self.sess=params['session']
		self.actions_list=params['actions_list']
		self.true_means=params['true_values']
		self.temperature_parameter=params['temperature']
		return

	def increment_counts(self,X):
		for x in X:
			self.counts[x]=self.counts[x]+1
	def compute_bias(self,nn):
		means, sigmas = self.sess.run([nn.mu, nn.sigma],feed_dict={nn.action:self.actions_list})
		variance=sigmas*sigmas
		mean_estimator_variance=variance/(numpy.array(self.counts))
		estimated_bias=numpy.sum(mean_estimator_variance*soft_operators.mellowmax_hessian_diag(means,self.temperature_parameter
																							   )
								)
		#estimated_bias=numpy.sum(mean_estimator_variance*soft_operators.mellowmax_hessian_diag(numpy.array(self.true_means),))/2
		
		return estimated_bias
'''
@brief:
Implementation of wasserstein distance used for adversarial training
@author:liuxin,2018.1.16
'''
import torch
import torch.nn as nn
import torch.autograd import Variable


class wasserstein(nn.Module)
	def __init__(self)
	    super(wasserstein,self).__init__()
	    self.transform = nn.Sequential(
		nn.Linear(20,10),
		nn.ReLU(),
		nn.Linear(10,1)	
	    ) 		

	def forward(self,h_all)
	    h_whole = self.transform(h_all)
	    h_table = torch.split(h_whole,3,1)#fix split size
	    wd_loss =torch.mean(h_table[0]) - torch.mean(h_table[1])
	    h_whole.backward()#?? 
	    grad_loss = torch.pow(torch.norm(h_whole.grad.data)-1,2) 
 	 
	    return wd_loss,grad_loss	




class domain_critic(nn.Module)
	def __init__(self,critic_name='wasserstein')
	    super(domain_critic,self).__init__()
	    if critic_name == 'wasserstein'
		self.critic = wasserstein()
	    else
		pass
	
	def forward(self,src,target)
	    loss=self.critic(src,target)
	    return loss	
		

'''
@brief:train critic network and maximise critic loss
@param[input]:critic model
@param[input]:src domain data and target domain data
@param[input]:training parameters,a dict including:training step,lr,gamma .etc
'''
def critic_model_train(critic_model,src,target,optimizer,**kwargs)
	    '''
		construct interpolates from src and target 
	    '''
	    alpha = torch.rand(cn)
	    differences = src-target
	    interpolates = target +alpha*differences
	    h_all = torch.cat((src,target,interpolates),1)
	    
	    for i in range(kwargs[0]):
            	wd_loss,grad_loss = critic_model(h_all)
	    	total_loss = -wd_loss+kwargs['gamma']*grad_loss
	        optimizer.zero_grad()
		total_loss.backward()
		optimizer.step()
				    
       

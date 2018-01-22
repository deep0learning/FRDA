'''
@brief:
implement domain adversarial networks to learn representations 
from different domains
@author liuxin
'''

import torch
import torch.nn as nn
from light_cnn  import network_29layers as LightCNN29
from light_cnn  import network_9layers as LightCNN9
import critic

'''
@brief:source classidier
It may include softmax,A-softmax or other like normalized softmax .etc
'''
class classifier_model(nn.Module)

	def __init__(self,classifier='softmax',num_classes)
		super(classifier_model,self).__init__()
		if classifier == 'softmax'
		self.classifier = nn.Sequential(
			nn.Linear(256,num_classes)
		)
		else
		   pass
	def forward(self,feature)
	    cls_pred = self.classifieri(feature)
	return cls_pred



'''
@brief:framework for adversarial training network
'''
class network_DA(nn.Module) 
	def __init__(self, critic='wassertein',classifier='softmax',src_classes,feature_extractor='LightCNN29'):
        super(network_DA, self).__init__()

        self.FExtractor = LightCNN(feature_extractor)
	
        self.classifier = classifier_model(classifier,src_classes)
        self.critic = domain_critic(critic)

    def forward(self, src,target):
        x1 = self.FExtractor(src)
        x2 = self.FExtractor(target)
        cls = self.classifier(x1)
        wd_loss= self.critic(x1,x2)
        return cls,wd_loss

     def load_network(self,network,network_name='All',epoch_label,save_dir):
	 		save_filename = '%s_net_%s.pth' % (epoch_label,network_name)
	 		save_path = os.join.join(save_dir,save_filename)
	 		network.load_state_dict(torch.load(save_path))
 
def Network_DA(**kwargs):
    model = network_DA(**kwargs)
    return model




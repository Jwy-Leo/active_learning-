import torch 
import numpy as np
RESET_WARNING = False
def reset_model(model,pretrain_model=None):

    if pretrain_model is not None:
        print("Use Pretrain model")
        model.load_state_dict(torch.load(pretrain_model))
    else:
        def weights_init(m):
            No_weight_club = ["ReLU","MaxPool"]
            Have_weight_club = ["Conv","Linear"]
            Have_Hyperparmater_weight_club = ["BatchNorm"]
            Module_strcture = ["Sequential"]
            Self_Define_Module = ["classification_model_feature_out_with_BN","LeNet_5_with_BN_feature_out"]
            
            classname = m.__class__.__name__
            # print(classname)
            
            if (np.array([classname.find(Have_weight_club[i]) for i in range(len(Have_weight_club))])!=-1).any():
                # m.weight.data = torch.nn.init.kaiming_normal(m.weight.data)
                m.weight.data.normal_(0.0,0.02)
            elif (np.array([classname.find(Have_Hyperparmater_weight_club[i]) for i in range(len(Have_Hyperparmater_weight_club))])!=-1).any():
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            elif (np.array([classname.find(No_weight_club[i]) for i in range(len(No_weight_club))])!=-1).any():
                if RESET_WARNING:
                    print("warning : {} doesn't initial".format(classname))
            elif (np.array([classname.find(Module_strcture[i]) for i in range(len(Module_strcture))])!=-1).any():
                for i in range(len(m)):
                    weights_init(m[i])
            elif (np.array([classname.find(Self_Define_Module[i]) for i in range(len(Self_Define_Module))])!=-1).any():
                temp_name = []
                for name,parameters in m.state_dict().items():
                    module_name = str.split(name,".")[0]
                    if module_name in temp_name:
                        continue
                    else:
                        temp_name.append(module_name)
                        mycode="weights_init(m.{})".format(module_name)
                        eval(mycode)
            else:
                import pdb;pdb.set_trace()
                print(m.weight.data)
                raise TypeError("This module :  {} doesn't define initial behavier".format(classname))        
        model.apply(weights_init)
    return model
import logging 
import torch
import torch.nn as nn
def main():
    test_early_stop()
class EarlyStop(object):
    def __init__(self,weak_period_stop = 10):
        assert type(weak_period_stop) is int
        self.num_patience = weak_period_stop
        self.best_model_state_dict=None
        self.count_early_stop = 0
        self.best_scores = -1.0
        self.cpu = torch.device("cpu")
        pass
    def __call__(self,performance,model):
        assert type(performance) is float
        assert isinstance(model,nn.Module)
        
        output_decision = False
               
        if performance <= self.best_scores:
            if self.count_early_stop > self.num_patience:
                output_decision = True
                self.count_early_stop=0
            self.count_early_stop+=1
        else:
            self.best_scores = performance
            self.best_model_state_dict = self._state_dict_to_cpu(model.state_dict())
            self.count_early_stop = 0
            for name,variable in model.state_dict().items():
                self.__setattr__(name,variable)
            output_decision = False  

        assert type(output_decision) is bool
        
        return output_decision
    def best_static_dict(self):
        if self.best_model_state_dict is None:
            print("EarlyStop Criteria yet to collect")
        return self.best_model_state_dict
    def _state_dict_to_cpu(self,state_dict):

        assert isinstance(state_dict,dict)

        for name,variable in state_dict.items():
            state_dict[name] =  variable.to(self.cpu)

        assert isinstance(state_dict,dict)

        return state_dict
    def _reset(self):
        self.best_model_state_dict=None
        self.count_early_stop = 0
        self.best_scores = -1.0
    def _New_Update_Round(self):
        self.count_early_stop = 0
def test_early_stop():
    model = nn.Sequential([nn.Linear(2,30),nn.BatchNorm(30),nn.Relu(),
                   nn.Linear(30,1)])
    
if "__main__" == __name__:
    main()
    

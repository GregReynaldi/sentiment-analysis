import pickle
import os

class SavePickle() : 
    def __init__(self, **models) : 
        self.models = models
        self.dir = os.path.dirname(os.path.abspath(__file__))
    def save(self) :
        targ_fold = os.path.join(self.dir,"models")
        os.makedirs(targ_fold, exist_ok=True)
        for model_name in self.models : 
            fileName = os.path.join(targ_fold, f"{model_name}.pickle")
            with open(fileName, "wb") as f : 
                pickle.dump(self.models[model_name], f)
        print("Saved Already")

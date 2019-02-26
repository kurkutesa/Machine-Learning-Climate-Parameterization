import numpy as np
seed = 42

###################################
## Data Container


class SampleSubset:
    def __init__(self, binary, y, Xscalar, Xvec, Xflatten=None, Xboundary=None, bin_hat=None, y_hat=None):
        self.bin = binary
        self.y = y
        self.Xscalar = Xscalar
        self.Xvec = Xvec

        self.Xflatten = Xflatten
        self.Xboundary = Xboundary

        self.bin_hat = bin_hat
        self.y_hat = y_hat

        self.size = binary.size
        self.class_ratio = binary.mean()


class AugmentedSubset:
    def __init__(self, ss, seed=seed):
        from utils import unflattening
        from imblearn.over_sampling import ADASYN
        adasyn = ADASYN(random_state=seed, n_jobs=-1)

        self.Xflatten, self.bin = adasyn.fit_sample(ss.Xflatten, ss.bin)
        self.Xboundary = ss.Xboundary
        self.Xscalar, self.Xvec = unflattening(self.Xflatten, ss.Xboundary)
        self.class_ratio = self.bin.mean()

        self.bin_hat = None

class Sample:
    def __init__(self, train, validation, test, scaler=None):
        self.train = train
        self.validation = validation
        self.test = test
        self.scaler = scaler

###################################
## Data scaler


class Scaler:
    def __init__(self, y, Xscalar, Xvec):
        self.y = y
        self.Xscalar = Xscalar
        self.Xvec = Xvec

###################################
## Predictor


class Classifier:
    def __init__(self, name, trained_classifier):
        self.name = name
        self.classifier = trained_classifier[0]
        self.f1_macro = trained_classifier[1]


class Regressor:
    def __init__(self, name, trained_regressor):
        self.name = name
        self.regressor = trained_regressor[0]
        self.mse = trained_regressor[1]

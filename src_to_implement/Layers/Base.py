class BaseLayer(object):
    def __int__(self):
        self.trainable = False
        self.weights = []
        self.testing_phase = False


class Neuron:
    def __init__(self, X= 0.0,  V = 0.0, Links = []):
        self.input = X
        self.Value = V
        self.Weights = Links


class Layer:
    def __init__(self, Neurons = []):
        self.Neurons = Neurons

    def AppendNode(self, Neuron, Link):
        self.Neurons.append(Neuron)

class Network:
    def __init__(self, Network = []):
        self.Layers = Network

    def ConstructNetwork(self, NewNetwork):
        self.Layers = NewNetwork

    def AddLayer(self, NewLayer):
        self.Layers.append(NewLayer)

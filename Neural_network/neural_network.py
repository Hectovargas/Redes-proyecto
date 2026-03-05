import json
import numpy as np
from Neural_network.activation import Activation_Relu, softmax
from Neural_network.loss import LossCategoricalCrossentropy
from Neural_network.metrics import accuracy
from Neural_network.dense_layer import DenseLayer


class Neural_network:
    def __init__(self, input_shape, neurons_list: list, act_func_list: list, targets=None):
        self.targets = targets
        self.input_shape = input_shape 
        self.layers = []
        self.functions = [] 

        inputs = np.prod(input_shape)
        
        for i in range(len(neurons_list)):
            self.layers.append(DenseLayer(inputs, neurons_list[i]))
            if act_func_list[i] == "relu": 
                self.functions.append(Activation_Relu())
            else:
                self.functions.append(softmax())
            inputs = neurons_list[i]
        
    def use(self, batch):
        output = batch
        for l, f in zip(self.layers, self.functions):
            output = f.forward(l.forward(output))
        return output

    def to_dict(self):
        layers_data = []
        for ly, fn in zip(self.layers, self.functions):
            fn_act = "relu" if isinstance(fn, Activation_Relu) else "softmax"
            layers_data.append({
                "type": "dense",
                "units": ly.biases.shape[1], 
                "activation": fn_act,
                "W": ly.weight.tolist(),
                "b": ly.biases.tolist()
            })
        return {
            "input_shape": self.input_shape,
            "preprocess": {"scale": 255.0},
            "layers": layers_data
        }

    @classmethod 
    def from_dict(cls, data:dict):
        input_shape = data["input_shape"]
        layers_info = data["layers"]
        
        neuron_list = [ly["units"] for ly in layers_info]
        activation_list = [ly["activation"] for ly in layers_info]
        
        nn = cls(input_shape, neuron_list, activation_list)

        for i, layer_data in enumerate(layers_info):
            nn.layers[i].weight = np.array(layer_data["W"], dtype=np.float32)
            nn.layers[i].biases = np.array(layer_data["b"], dtype=np.float32)
        return nn
class InputPerceptron:
    def __init__(self, input_value):
        self.input = input_value

class HiddenPerceptron:
    def __init__(self, initial_weight, num_perceptron_previous_layer):
        self.output = 0
        self.delta = 0
        self.weight = []
        self.delta_weight = []
        for i in range (num_perceptron_previous_layer):
            self.weight.append(initial_weight)
            self.delta_weight.append(0)

class OutputPerceptron:
    def __init__(self, target_value, initial_weight, num_perceptron_previous_layer):
        self.output = 0
        self.target = target_value
        self.error = 0
        self.delta = 0
        self.weight = []
        self.delta_weight = []
        for i in range (num_perceptron_previous_layer):
            self.weight.append(initial_weight)
            self.delta_weight.append(0)

class Bias:
    def __init__(self):
        self.value = 1

class Layer:
    def __init__(self):
        self.perceptron_list = []
        self.num_perceptron = 0

    def add_perceptron(self, perceptron_or_bias):
        self.perceptron_list.append(perceptron_or_bias)
        self.num_perceptron += 1

class Model:
    def __init__(self):
        self.layer_list = []
        self.num_layer = 0

    def add_layer(self, layer):
        self.layer_list.append(layer)
        self.num_layer += 1
import math
class InputPerceptron:
    def __init__(self, input_value):
        self.output = input_value

class HiddenPerceptron:
    def __init__(self, initial_weight, num_perceptron_previous_layer):
        self.output = 0
        self.delta = 0
        self.weight = []
        self.delta_weight = []
        for i in range (num_perceptron_previous_layer):
            self.weight.append(initial_weight)
            self.delta_weight.append(0)

    def set_output(self, output_value):
        self.output = output_value

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

    def set_output(self, output_value):
        self.output = output_value

    def set_error(self):
        self.error = math.pow((self.target-self.output),2) * 0.5

    def set_delta(self):
        self.delta = (self.output - self.target) * self.output * (1 - self.output)

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

    def get_net(self, layer_idx, perceptron_idx):
        net_value = 0
        i = 0
        for x in self.layer_list[layer_idx-1]:
            net_value += (x.output * self.layer_list[layer_idx][perceptron_idx].weight[i])
            i += 1
        net_value += 1 * self.layer_list[layer_idx][perceptron_idx].weight[i]
        return net_value

    def set_sigmoid(self, net_value, layer_idx, perceptron_idx):
        output_value = 1 / (1+math.exp(-net_value))
        self.layer_list[layer_idx][perceptron_idx].set_output(output_value)

    def set_delta_weight(self, layer_idx, perceptron_idx):
        for i in range (len(self.layer_list[layer_idx][perceptron_idx].delta_weight)):
            self.layer_list[layer_idx][perceptron_idx].delta_weight[i] += self.layer_list[layer_idx][perceptron_idx].delta * self.layer_list[layer_idx-1][i].output

    def set_hidden_delta(self, layer_idx, perceptron_idx):
        e_per_output = 0
        for i in range (len(self.layer_list[layer_idx+1].num_perceptron)):
            e_per_output += (self.layer_list[layer_idx+1][i].delta * self.layer_list[layer_idx+1][i].weight[perceptron_idx])
        self.layer_list[layer_idx][perceptron_idx].delta = e_per_output * self.layer_list[layer_idx][perceptron_idx].output * (1 - self.layer_list[layer_idx][perceptron_idx].output)

    def set_weight(self):
        for i in range (1,self.num_layer):
            for j in range (self.layer_list[i].num_perceptron):
                for k in range (self.layer_list[i][j].weight):
                    self.layer_list[i][j].weight += self.layer_list[i][j].delta_weight
                    self.layer_list[i][j].delta_weight = 0 # jadi 0 lagi ga ya

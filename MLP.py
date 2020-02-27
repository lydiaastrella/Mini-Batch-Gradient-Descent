import math
import copy
class InputPerceptron:
    def __init__(self, input_value):
        self.output = input_value           # input data

class HiddenPerceptron:
    def __init__(self, num_perceptron_previous_layer):
        self.output = 0                     # sigmoid function result
        self.delta = 0                      # used for backward phase
        self.weight = []                    # weight of edges to previous layer + bias weight at last index
        self.delta_weight = []              # weight difference of edges to previous layer + bias delta weight at last index
        for i in range (num_perceptron_previous_layer + 1):
            self.weight.append(0)
            self.delta_weight.append(0)

    def set_output(self, output_value):
        self.output = output_value

    def set_weight (self, weight_idx, weight_value):
        self.weight[weight_idx] = weight_value

class OutputPerceptron:
    #similar structure with hidden perceptron
    def __init__(self, target_value, num_perceptron_previous_layer):
        self.output = 0
        self.target = target_value
        self.error = 0
        self.delta = 0
        self.weight = []
        self.delta_weight = []
        for i in range (num_perceptron_previous_layer + 1):
            self.weight.append(0)
            self.delta_weight.append(0)

    def set_output(self, output_value):
        self.output = output_value

    def set_error(self):
        self.error = math.pow((self.target-self.output),2) * 0.5

    def set_delta(self):
        self.delta = (self.output - self.target) * self.output * (1 - self.output)

    def set_weight (self, weight_idx, weight_value):
        self.weight[weight_idx] = weight_value

class Layer:
    def __init__(self):
        self.perceptron_list = []           #list of perceptron in layer
        self.num_perceptron = 0

    def add_perceptron(self, perceptron):
        self.perceptron_list.append(perceptron)
        self.num_perceptron += 1

class Model:
    def __init__(self):
        self.layer_list = []               #list of layer in model (input layer, hidden layer, and output layer)
        self.num_layer = 0
        self.cummulative_error = 0

    def get_perceptron(self, layer_idx, perceptron_idx):
        return self.layer_list[layer_idx].perceptron_list[perceptron_idx]

    def add_layer(self, layer):
        self.layer_list.append(layer)
        self.num_layer += 1

    # return net value of perceptron in [layer_idx, perceptron_idx]
    def get_net(self, layer_idx, perceptron_idx):
        net_value = 0
        i = 0
        for x in self.layer_list[layer_idx-1].perceptron_list:
            net_value += (x.output * self.get_perceptron(layer_idx, perceptron_idx).weight[i])
            i += 1
        net_value += 1 * self.get_perceptron(layer_idx, perceptron_idx).weight[i]
        return net_value

    # set output value of perceptron in [layer_idx, perceptron_idx]
    def set_sigmoid(self, net_value, layer_idx, perceptron_idx):
        output_value = 1 / (1+math.exp(-net_value))
        self.get_perceptron(layer_idx, perceptron_idx).set_output(output_value)

    # set all delta weight of perceptron in [layer_idx, perceptron_idx]
    def set_delta_weight(self, layer_idx, perceptron_idx):
        for i in range (len(self.get_perceptron(layer_idx, perceptron_idx).delta_weight) - 1):
            self.get_perceptron(layer_idx, perceptron_idx).delta_weight[i] += self.get_perceptron(layer_idx, perceptron_idx).delta * self.get_perceptron(layer_idx-1, i).output
        self.get_perceptron(layer_idx, perceptron_idx).delta_weight[i+1] += self.get_perceptron(layer_idx, perceptron_idx).delta * 1

    # set delta (backward phase) of hidden perceptron.
    # For output perceptron, use set_delta() method instead
    def set_hidden_delta(self, layer_idx, perceptron_idx):
        e_per_output = 0
        for i in range (self.layer_list[layer_idx+1].num_perceptron):
            e_per_output += (self.get_perceptron(layer_idx+1, i).delta * self.get_perceptron(layer_idx+1, i).weight[perceptron_idx])
        self.get_perceptron(layer_idx, perceptron_idx).delta = e_per_output * self.get_perceptron(layer_idx, perceptron_idx).output * (1 - self.get_perceptron(layer_idx, perceptron_idx).output)

    # add all delta weight value to weight (last step of one batch)
    def set_all_weight(self, learning_rate):
        for i in range (1,self.num_layer):
            for j in range (self.layer_list[i].num_perceptron):
                for k in range (len(self.get_perceptron(i,j).weight)):
                    self.get_perceptron(i,j).weight[k] -= learning_rate * self.get_perceptron(i,j).delta_weight[k]
                    self.get_perceptron(i,j).delta_weight[k] = 0 # jadi 0 lagi ga ya? saya bingung gais wkwk
    
    #update cummulative_error at the end of feed forward
    def update_cumulative_error(self):
        for x in (self.layer_list[self.num_layer-1].perceptron_list):
            self.cummulative_error += x.error

    #reset cummulative error before epoch begin
    def reset_cumulative_error(self):
        for x in (self.layer_list[self.num_layer-1].perceptron_list):
            self.cummulative_error += x.error
        self.cummulative_error = 0

    def feedForward(self):
        #set all net value and sigmoid value
        for i in range(1, self.num_layer):
            for j in range(self.layer_list[i].num_perceptron):
                net = self.get_net(i,j)
                self.set_sigmoid(net,i,j)

        #set error value
        output = self.num_layer-1
        for k in range(self.layer_list[output].num_perceptron):
            self.layer_list[output].perceptron_list[k].set_error()
        
        self.update_cumulative_error()

    #BACKWARD PHASE
    def backward_phase(self, learning_rate):
        for output_perceptron in (self.layer_list[self.num_layer-1].perceptron_list):
            output_perceptron.set_delta()
            print('delta o ' +str(output_perceptron.delta))

        for idx_output_perceptron in range (self.layer_list[self.num_layer-1].num_perceptron):
            # print('idx_output_perceptron : ' + str(idx_output_perceptron))
            self.set_delta_weight(self.num_layer-1, idx_output_perceptron)
            print('delta weight o ' +str(self.get_perceptron(self.num_layer-1, idx_output_perceptron).delta_weight[0]))

        for idx_hidden_layer in range (1, self.num_layer-1):
            for idx_hidden_perceptron in range (self.layer_list[idx_hidden_layer].num_perceptron):
                self.set_hidden_delta(idx_hidden_layer, idx_hidden_perceptron)
                print('delta h ' +str(self.get_perceptron(idx_hidden_layer, idx_hidden_perceptron).delta))
        
        for idx_hidden_layer in range (1, self.num_layer-1):
            for idx_hidden_perceptron in range (self.layer_list[idx_hidden_layer].num_perceptron):
                self.set_delta_weight(idx_hidden_layer, idx_hidden_perceptron)
                print('delta weight h ' +str(self.get_perceptron(idx_hidden_layer, idx_hidden_perceptron).delta_weight[0]))

        self.set_all_weight(learning_rate)

# CARA PAKAI SESUAI CONTOH DI PPT ANN HALAMAN 49 dst.
# STRUKTUR
i1 = InputPerceptron(0.05)
i2 = InputPerceptron(0.1)
h1 = HiddenPerceptron(2)
h2 = HiddenPerceptron(2)
o1 = OutputPerceptron(0.01, 2)
o2 = OutputPerceptron(0.99, 2)

h1.set_weight(0, 0.15)
h1.set_weight(1, 0.2)
h1.set_weight(2, 0.35)
h2.set_weight(0, 0.25)
h2.set_weight(1, 0.3)
h2.set_weight(2, 0.35)
o1.set_weight(0, 0.4)
o1.set_weight(1, 0.45)
o1.set_weight(2, 0.6)
o2.set_weight(0, 0.5)
o2.set_weight(1, 0.55)
o2.set_weight(2, 0.6)

l1 = Layer()
l1.add_perceptron(i1)
l1.add_perceptron(i2)
l2 = Layer()
l2.add_perceptron(h1)
l2.add_perceptron(h2)
l3 = Layer()
l3.add_perceptron(o1)
l3.add_perceptron(o2)

model = Model()
model.add_layer(l1)
model.add_layer(l2)
model.add_layer(l3)

# print('jumlah layer : ' + model.num_layer)

# FEED FORWARD

model.feedForward()
print('output h1 ' +str(h1.output))
print('output h2 ' +str(h2.output))

print('output o1 ' +str(o1.output))
print('output o2 ' +str(o2.output))

print('error o1 ' +str(o1.error))
print('error o2 ' +str(o2.error))

print('cummulative error ' +str(model.cummulative_error))

# BACKWARD PHASE

print('----------------------------------------')
model.backward_phase(0.5)
# o1.set_delta()
# print('delta o1 ' +str(o1.delta))
# #o2.set_delta()

# # model.set_delta_weight(2, 0)
# # model.set_delta_weight(2, 1)
# print('delta weight o1 ' +str(o1.delta_weight[0]))

# # model.set_hidden_delta(1, 0)
# # model.set_hidden_delta(1, 1)
# print('delta h1 ' +str(h1.delta))

# # model.set_delta_weight(1, 0)
# # model.set_delta_weight(1, 1)
# print('delta weight h1 ' +str(h1.delta_weight[0]))

# PENGUBAHAN BOBOT
print('----------------------------------------')

print('w1 ' + str(h1.weight[0]))
print('w2 ' + str(h1.weight[1]))
print('w3 ' + str(h2.weight[0]))
print('w4 ' + str(h2.weight[1]))
print('w5 ' + str(o1.weight[0]))
print('w6 ' + str(o2.weight[1]))
print('w7 ' + str(o2.weight[0]))
print('w8 ' + str(o2.weight[1]))

print('w bias h1 '+ str(h1.weight[2]))
print('w bias h2 '+ str(h2.weight[2]))
print('w bias o1 '+ str(o1.weight[2]))
print('w bias o2 '+ str(o2.weight[2]))

print('delta weight h1 ' +str(h1.delta_weight[0]))


import math
class InputPerceptron:
    def __init__(self, input_value):
        self.output = input_value

class HiddenPerceptron:
    def __init__(self, num_perceptron_previous_layer):
        self.output = 0
        self.delta = 0
        self.weight = []
        self.delta_weight = []
        for i in range (num_perceptron_previous_layer + 1):
            self.weight.append(0)
            self.delta_weight.append(0)

    def set_output(self, output_value):
        self.output = output_value

    def set_weight (self, weight_idx, weight_value):
        self.weight[weight_idx] = weight_value

class OutputPerceptron:
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
        self.perceptron_list = []
        self.num_perceptron = 0

    def add_perceptron(self, perceptron):
        self.perceptron_list.append(perceptron)
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
        for x in self.layer_list[layer_idx-1].perceptron_list:
            net_value += (x.output * self.layer_list[layer_idx].perceptron_list[perceptron_idx].weight[i])
            i += 1
        net_value += 1 * self.layer_list[layer_idx].perceptron_list[perceptron_idx].weight[i]
        return net_value

    def set_sigmoid(self, net_value, layer_idx, perceptron_idx):
        output_value = 1 / (1+math.exp(-net_value))
        self.layer_list[layer_idx].perceptron_list[perceptron_idx].set_output(output_value)

    def set_delta_weight(self, layer_idx, perceptron_idx):
        #print(len(self.layer_list[layer_idx].perceptron_list[perceptron_idx].delta_weight))
        for i in range (len(self.layer_list[layer_idx].perceptron_list[perceptron_idx].delta_weight) - 1):
            print(i)
            self.layer_list[layer_idx].perceptron_list[perceptron_idx].delta_weight[i] += self.layer_list[layer_idx].perceptron_list[perceptron_idx].delta * self.layer_list[layer_idx-1].perceptron_list[i].output
        print(i+1)
        self.layer_list[layer_idx].perceptron_list[perceptron_idx].delta_weight[i+1] += self.layer_list[layer_idx].perceptron_list[perceptron_idx].delta * 1

    def set_hidden_delta(self, layer_idx, perceptron_idx):
        e_per_output = 0
        for i in range (self.layer_list[layer_idx+1].num_perceptron):
            e_per_output += (self.layer_list[layer_idx+1].perceptron_list[i].delta * self.layer_list[layer_idx+1].perceptron_list[i].weight[perceptron_idx])
        self.layer_list[layer_idx].perceptron_list[perceptron_idx].delta = e_per_output * self.layer_list[layer_idx].perceptron_list[perceptron_idx].output * (1 - self.layer_list[layer_idx].perceptron_list[perceptron_idx].output)

    def set_all_weight(self):
        for i in range (1,self.num_layer):
            for j in range (self.layer_list[i].num_perceptron):
                for k in range (len(self.layer_list[i].perceptron_list[j].weight)):
                    self.layer_list[i].perceptron_list[j].weight[k] += self.layer_list[i].perceptron_list[j].delta_weight[k]
                    self.layer_list[i].perceptron_list[j].delta_weight[k] = 0 # jadi 0 lagi ga ya? saya bingung gais wkwk

# Cara pakai

# Struktur
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

# Feed Forward
neth1 = model.get_net(1,0)
neth2 = model.get_net(1,1)
model.set_sigmoid(neth1, 1, 0)
model.set_sigmoid(neth2, 1, 1)
print(h1.output)
print(h2.output)

neto1 = model.get_net(2,0)
neto2 = model.get_net(2,1)
model.set_sigmoid(neto1, 2, 0)
model.set_sigmoid(neto2, 2, 1)
print(o1.output)
print(o2.output)

o1.set_error()
o2.set_error()
print(o1.error)
print(o2.error)

#Backward Phase
print('----------------------------------------')
o1.set_delta()
print(o1.delta)
o2.set_delta()

model.set_delta_weight(2, 0)
model.set_delta_weight(2, 1)
print(o1.delta_weight[0])

model.set_hidden_delta(1, 0)
model.set_hidden_delta(1, 1)
print(h1.delta)

model.set_delta_weight(1, 0)
model.set_delta_weight(1, 1)
print(h1.delta_weight[0])

model.set_all_weight()
print('w1 ' + str(h1.weight[0]))
print('w2 ' + str(h1.weight[1]))
print('w3 ' + str(h2.weight[0]))
print('w4 ' + str(h2.weight[1]))
print('w5 ' + str(o1.weight[0]))
print('w6 ' + str(o2.weight[1]))
print('w7 ' + str(o2.weight[0]))
print('w8 ' + str(o2.weight[1]))

print('bias h1 '+ str(h1.weight[2]))
print('bias h2 '+ str(h2.weight[2]))
print('bias o1 '+ str(o1.weight[2]))
print('bias o2 '+ str(o2.weight[2]))

print(h1.delta_weight[0])
import random
import math
import copy
import pandas
import os,sys
class InputPerceptron:
    def __init__(self):
        self.output = 0           # input data
    
    def set_input_value(self, input_value):
        self.output = input_value

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
    def __init__(self, label, num_perceptron_previous_layer):
        self.label = label
        self.output = 0
        self.target = 0
        self.error = 0
        self.delta = 0
        self.weight = []
        self.delta_weight = []
        for i in range (num_perceptron_previous_layer + 1):
            self.weight.append(0)
            self.delta_weight.append(0)

    def set_target(self, target_value):
        self.target = target_value

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
        self.layer_list[layer_idx].perceptron_list[perceptron_idx].set_output(output_value)
        #self.get_perceptron(layer_idx, perceptron_idx).set_output(output_value)

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
        
    #BACKWARD PHASE
    def backward_phase(self):
        # for output_perceptron in (self.layer_list[self.num_layer-1].perceptron_list):
        #     output_perceptron.set_delta()
        #     #print('delta o ' +str(output_perceptron.delta))

        for idx_output_perceptron in range (self.layer_list[self.num_layer-1].num_perceptron):
            #print('idx_output_perceptron : ' + str(idx_output_perceptron))
            self.layer_list[self.num_layer-1].perceptron_list[idx_output_perceptron].set_delta()
            self.set_delta_weight(self.num_layer-1, idx_output_perceptron)
            #print('delta weight o ' +str(self.get_perceptron(self.num_layer-1, idx_output_perceptron).delta_weight[0]))

        for idx_hidden_layer in range (1, self.num_layer-1):
            for idx_hidden_perceptron in range (self.layer_list[idx_hidden_layer].num_perceptron):
                self.set_hidden_delta(idx_hidden_layer, idx_hidden_perceptron)
                self.set_delta_weight(idx_hidden_layer, idx_hidden_perceptron)
                #print('delta h ' +str(self.get_perceptron(idx_hidden_layer, idx_hidden_perceptron).delta))
        
        # for idx_hidden_layer in range (1, self.num_layer-1):
        #     for idx_hidden_perceptron in range (self.layer_list[idx_hidden_layer].num_perceptron):
        #         self.set_delta_weight(idx_hidden_layer, idx_hidden_perceptron)
        #         #print('delta weight h ' +str(self.get_perceptron(idx_hidden_layer, idx_hidden_perceptron).delta_weight[0]))

    #PRINT MODEL
    def print_model(self):
        #print input layer
        print("INPUT LAYER")
        for i in range(self.layer_list[0].num_perceptron):
            print("|I-" + str(i+1) + "| <= value(" + str(self.layer_list[0].perceptron_list[i].output)+ ")  ")
        print("\n")

        #print hidden layer
        print("HIDDEN LAYER")
        for j in range(1, self.num_layer-1):
            for k in range(self.layer_list[j].num_perceptron):
                print("|H-(" + str(j) + "," + str(k+1) + ")| <= weight ", end="")
                for w in range(len(self.layer_list[j].perceptron_list[k].weight)):
                    print("(" + str(self.layer_list[j].perceptron_list[k].weight[w]), end="),  ")
                print("\n")
        print("\n")

        #print output layer
        print("OUTPUT LAYER")
        last = self.num_layer -1
        for o in range(self.layer_list[last].num_perceptron):
            print("|O-" + str(o+1) + "| <= weight ", end="")
            for x in range(len(self.layer_list[last].perceptron_list[o].weight)):
                print("(" + str(self.layer_list[last].perceptron_list[o].weight[x]), end="),  ")
            print("( target => " + str(self.layer_list[last].perceptron_list[o].label), end=" )  ")
            print("\n")
        print("\n")

# Initiate empty model
model = Model()

# Ask for data source
data_source = input('Masukkan data yang diinginkan (iris/ppt): ')

# Fetch data
if (data_source == 'ppt'):
    data = {
        'attr1': [0.05],
        'attr2': [0.01],
        'result': ['Output2']
    }
    df = pandas.DataFrame(data)
else:
    df = pandas.read_csv('iris.csv')
print('Data loaded.')

# Determine info about input and outputs from data
attributes = df.columns.values.tolist()
result_column_name = attributes.pop()
results = set(df.iloc[:,-1].tolist())
result_labels = []
for result in results:
    result_labels.append(result)

# Ask for number of hidden layers and number of perceptrons per layer
if (data_source == 'ppt'):
    result_labels.append('Output1')
    num_hidden_layer = 1
    num_perceptrons_in_layer = [2]
else:
    num_hidden_layer = int(input('Masukkan jumlah hidden layer: '))
    num_perceptrons_in_layer = []
    print('Masukkan jumlah perceptron untuk tiap layer.')
    for x in range(num_hidden_layer):
        num_perceptrons_in_layer.append(int(input()))

# Build model according to given variables
layer = Layer()
for x in range(len(attributes)):
    layer.add_perceptron(InputPerceptron())
model.add_layer(layer)

for layer_idx in range(len(num_perceptrons_in_layer)):
    layer = Layer()
    for x in range(num_perceptrons_in_layer[layer_idx]):
        if (layer_idx == 0):
            hp = HiddenPerceptron(len(attributes))
        else:
            hp = HiddenPerceptron(num_perceptrons_in_layer[layer_idx-1])
        for input_idx in range(len(hp.weight)):
            hp.set_weight(input_idx, 0)
            # hp.set_weight(input_idx, float(random.randrange(0, 100)) / 100)
        layer.add_perceptron(hp)
    model.add_layer(layer)

layer = Layer()
for x in range(len(result_labels)):
    op = OutputPerceptron(result_labels[x], num_perceptrons_in_layer[len(num_perceptrons_in_layer)-1])
    for input_idx in range(num_perceptrons_in_layer[len(num_perceptrons_in_layer)-1] + 1):
        op.set_weight(input_idx, 0)
        # op.set_weight(input_idx, float(random.randrange(0, 100)) / 100)
    layer.add_perceptron(op)
model.add_layer(layer)

print('Model initialized.')

# INPUT VARIABLES
max_iteration = int(input('Jumlah maksimal iterasi  : '))
error_threshold = float(input('Error threshold          : '))
learning_rate = float(input('Learning rate            : '))
batch_size = int(input('Jumlah data per batch    : '))

print('Backpropagation in progress...')

# MAIN LOOP
itr = 0
error = float("inf")

num_batches = round(len(df.index) / batch_size)
if (num_batches == 0):
    num_batches = 1

f = open("output.txt", "w")
sys.stdout = f
while (itr < max_iteration) and (error > error_threshold):
    cummulative_error = 0
    itr += 1
    print('----------------------- ITERATION', itr, '-----------------------')
    for x in range(num_batches):
        for y in range(batch_size):
            if (x * batch_size + y < len(df.index)):
                data_row = df.iloc[x * batch_size + y]
                # set input values
                for i in range(len(attributes)):
                    model.get_perceptron(0, i).set_input_value(data_row.get(attributes[i]))
                # set target values
                for i in range(len(result_labels)):
                    if data_row.get(result_column_name) != model.get_perceptron(model.num_layer-1, i).label:
                        model.get_perceptron(model.num_layer-1, i).set_target(0)
                    else:
                        model.get_perceptron(model.num_layer-1, i).set_target(1)
                        #print('Index', str(x * batch_size + y), '; target:', model.get_perceptron(model.num_layer-1, i).label)
                
                model.feedForward()

                # Choose output label with largest output value
                idx_best = 0
                for i in range(len(result_labels)):
                    print(model.get_perceptron(model.num_layer-1, i).output, '?', model.get_perceptron(model.num_layer-1, idx_best).output)
                    if model.get_perceptron(model.num_layer-1, i).output > model.get_perceptron(model.num_layer-1, idx_best).output:
                        idx_best = i
                
                print('terpilih :', idx_best)

                # Set error if output is not desired label
                if model.get_perceptron(model.num_layer-1, idx_best).label != data_row.get(result_column_name):
                    print('Index', str(x * batch_size + y), model.get_perceptron(model.num_layer-1, idx_best).label, data_row.get(result_column_name))
                    cummulative_error += 1
                
                model.backward_phase()
        
        model.set_all_weight(learning_rate)
    
    error = float(cummulative_error) / len(df.index)

print('Backpropagation finished, calculating accuracy...')

num_error = 0
# ACCURACY CHECK
for x in range(len(df.index)):
    data_row = df.iloc[x]
    # set input values
    for i in range(len(attributes)):
        model.get_perceptron(0, i).set_input_value(data_row.get(attributes[i]))
    # set target values
    for i in range(len(result_labels)):
        if data_row.get(result_column_name) != model.get_perceptron(model.num_layer-1, i).label:
            model.get_perceptron(model.num_layer-1, i).set_target(0)
        else:
            model.get_perceptron(model.num_layer-1, i).set_target(1)
    
    model.feedForward()

    # Choose output label with largest output value
    idx_best = 0
    for i in range(len(result_labels)):
        if model.get_perceptron(model.num_layer-1, i).output > model.get_perceptron(model.num_layer-1, idx_best).output:
            idx_best = i
    
    print(model.get_perceptron(model.num_layer-1, idx_best).label)
    # Set error if output is not desired label
    if model.get_perceptron(model.num_layer-1, idx_best).label != data_row.get(result_column_name):
        num_error += 1

accuracy = 1 - float(num_error) / len(df.index)
print('Model has an accuracy of', accuracy)

model.print_model()
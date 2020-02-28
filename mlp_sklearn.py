from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import pandas as pd
import os,sys

# Loading iris dataset
df = pd.read_csv('iris.csv')
print('Data loaded.')

# Determine info about input and outputs from data
attributes = df.columns.values.tolist()
attributes.pop()
results = set(df.iloc[:,-1].tolist())
result_labels = []
for result in results:
    result_labels.append(result)
result_numeric = df['variety'].replace(result_labels,[0,1,2])

# Splitting data to inputs and targets
input_data = []
for i in range (len(df.values)):
    input_data_values = []
    for j in range (len((df.values)[i])-1):
        input_data_values.append(((df.values)[i])[j])
    input_data.append(input_data_values)

target = []
for i in range (len(df.values)):
    target.append(result_numeric[i])

status = input("Mau pakai parameter manual atau auto aja? (manual/auto) ")

if (status == 'manual'):
    # INPUT VARIABLES
    num_hidden_layer = int(input('Masukkan jumlah hidden layer: '))
    num_perceptrons_in_layer = []
    print('Masukkan jumlah perceptron untuk tiap layer.')
    for x in range(num_hidden_layer):
        num_perceptrons_in_layer.append(int(input()))    
    max_iteration = int(input('Jumlah maksimal iterasi  : '))
    learning_rate = float(input('Learning rate            : '))
    batch_size = int(input('Jumlah data per batch    : '))

    # MLP from sklearn dengan param macem2
    mlp = MLPClassifier(hidden_layer_sizes=tuple(num_perceptrons_in_layer), max_iter=max_iteration, solver='sgd', 
                        batch_size=batch_size, learning_rate_init=learning_rate)
else:
    # MLP from sklearn dengan param auto semua
    mlp = MLPClassifier()

mlp.fit(input_data, target)

f = open("output_sklearn.txt", "w")
sys.stdout = f

prediction = mlp.predict(input_data)
print('Akurasi MLP dari sklearn :', metrics.accuracy_score(prediction,target))
print('')
print('HIDDEN LAYERS\' WEIGHT :')
for input_layer_id in range (len(mlp.coefs_[0])):
    for hidden_layer_id in range (len(mlp.coefs_[0][input_layer_id])):
        print("from input layer " + str(input_layer_id+1) + " to hidden layer " + str(hidden_layer_id+1) + " : " + str(mlp.coefs_[0][input_layer_id][hidden_layer_id]))
print('')
print('OUTPUT LAYERS\' WEIGHT :')
for hidden_layer_id in range (len(mlp.coefs_[1])):
    for output_layer_id in range (len(mlp.coefs_[1][hidden_layer_id])):
        print("from hidden layer " + str(hidden_layer_id+1) + " to output layer " + str(output_layer_id+1) + " : " + str(mlp.coefs_[1][hidden_layer_id][output_layer_id]))
print('')
print('HIDDEN LAYERS\' BIAS WEIGHT :')
for hidden_layer_id in range (len(mlp.intercepts_[0])):
    print("bias hidden layer " + str(hidden_layer_id+1) + " : " + str(mlp.intercepts_[0][hidden_layer_id]))
print('')
print('OUTPUT LAYERS\' BIAS WEIGHT :')
for output_layer_id in range (len(mlp.intercepts_[1])):
    print("bias output layer " + str(output_layer_id+1) + " : " + str(mlp.intercepts_[1][output_layer_id]))
print('')
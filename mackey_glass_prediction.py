import numpy as np
import matplotlib.pyplot as plt
import random


#Parameters for the Training

dataset = './data/data.npy'
num_taps = 4

learning_rate = 0.0001
epochs = 200


#Function to load dataset with required number of taps
def load_dataset():
    data = np.load(dataset)

    x_train = []
    y_train = []

    for i in range(len(data)-num_taps):
        x_train.append(np.array([data[j] for j in range(i, i+num_taps)], dtype=np.float64))
        y_train.append(data[i+num_taps])

    x_train = np.array(x_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    x_values = [i for i in range(num_taps,len(data))]

    training_data = {}
    
    for x,y,z in zip(x_values, x_train, y_train):
        training_data[x] = [y, z]
    
    return training_data

#Function to split the dataset into training dataset and testing dataset
def split_dataset(data):
    data_keys = list(data.keys())
    random.shuffle(data_keys)

    train_data = {}
    test_data = {}

    total_count = len(data_keys)
    train_data_count = int(0.8*total_count)
    
    for i in range(train_data_count):
        train_data[data_keys[i]] = data[data_keys[i]]
    
    for i in range(train_data_count, total_count):
        test_data[data_keys[i]] = data[data_keys[i]]
    
    return train_data, test_data

#Function to train the model and return the weights. It plots the epoch vs. error graph.
def training(train_data):
    weights = np.zeros((epochs+1,num_taps))
    errors = []

    for epoch in range(epochs):
        w = weights[epoch]
        error = 0
        for key, value in train_data.items():
            x = value[0]
            y = value[1]
            y_pred = np.dot(w.T, x)
            e = y - y_pred
            error += np.power(e,2)
            correction = np.dot(learning_rate*e, x)
            w = w + correction
        weights[epoch+1] = w
        error = error/len(train_data)

        errors.append(error)

        print('epoch {}, loss {}, w {}'.format(epoch+1,error,weights[epoch+1]))
    
    fig = plt.figure()
    plt.plot([i for i in range(epochs)], errors)
    fig.suptitle('Epochs vs Error')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.show()
    
    return weights[-1]

#Function to evaluate the model and plot the graphs
def evaluate(testing_data, weights):
    test_keys = sorted(testing_data)
    test_data = {}
    for i in test_keys:
        test_data[i] = testing_data[i]
    x_values = []
    y_values = []
    y_predicted = []
    error = 0
    for key, value in test_data.items():
        y_predict = np.dot(weights.T, value[0])
        error += np.power((value[1] - y_predict), 2)
        x_values.append(key)
        y_values.append(value[1])
        y_predicted.append(y_predict)

    plt.plot(x_values, y_values, label = 'From Data', alpha = .5)
    plt.plot(x_values, y_predicted, label = 'Prediction', alpha = 0.5)
    plt.legend()
    plt.show()

    return error/len(testing_data)

if __name__ == "__main__":

    data = load_dataset()
    train_data, test_data = split_dataset(data)

    weights = training(train_data)

    error_test = evaluate(test_data, weights)
    print('\n\nError in prediction of the test data = {}'.format(error_test))

    error_data = evaluate(data, weights)
    print('\n\nError in prediction of the full dataset = {}\n\n'.format(error_data))
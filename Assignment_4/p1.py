import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
## ========================================= Dataset Class =========================================== ##
class p1Dataset(Dataset):
    def __init__(self, csv_file):
        xy = np.loadtxt(csv_file, delimiter=',', dtype=np.float32)
        self.n_samples = xy.shape[0]
        self.y_data = torch.from_numpy(xy[:, 1:]).float()
        #self.x_data = np.squeeze(xy[:, [0]])
        self.x_data = torch.from_numpy(xy[:, [0]]).float()
        #print(self.x_data, type(self.x_data))
        #print(self.y_data, type(self.y_data))

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

## =======================================  NeuralNet Class  ========================================== ##
# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self,  hidden_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(1, hidden_size)
        self.softplus = nn.Softplus()
        self.l2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.l1(x)
        out = self.softplus(out)
        out = self.l2(out)
        return out

## ===================================== K-Fold ================================================== ##
def kfold( K, num_epochs, nP, dataset, verbose = False):
    #K = 10
    #n_samples = len(dataset)
    batch_size = int(len(dataset) / K)
    avg_acc = 0
    mse_array = list()
    model = NeuralNet(nP).to(device)
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
    for k in range(1, K+1):
        if verbose:
            print('-----------------------------------------------------------------')
            print('K = ', k)
        features_val, labels_val = dataset[(k-1)*batch_size : k*batch_size]
        if k == 1:
            features_train, labels_train = dataset[k*batch_size:]
        elif k == K:
            features_train, labels_train = dataset[:(k-1)*batch_size]
        else:
            features_train1, labels_train1 = dataset[:(k - 1) * batch_size]
            features_train2, labels_train2 = dataset[k * batch_size:]
            features_train = torch.cat([features_train1, features_train2], dim=0)
            labels_train = torch.cat([labels_train1, labels_train2], dim=0)

        #moving data to selected device
        features_val = features_val.to(device)
        labels_val = labels_val.to(device)
        features_train = features_train.to(device)
        labels_train = labels_train.to(device)

        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(features_train)
            loss = loss_fn(outputs, labels_train)

            # Backward and optimize
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if verbose and (epoch + 1) % (num_epochs/10) == 0:
                print('Epoch ', (epoch + 1), '/', num_epochs, ' Loss: ', loss.item())
        '''
        # Print weights
        for name, param in model.named_parameters():
            print(name, param.data)
         '''
        with torch.no_grad():
            outputs_val = model(features_val)
            mse = loss_fn(outputs_val, labels_val)
            mse_array.append(mse.numpy())

        if verbose:
            print('MSE: ', mse.numpy())

    #print('MSE in each fold: ', mse_array)
    avg_mse = sum(mse_array) / len(mse_array)
    if verbose:
        print('Average MSE: ', avg_mse)
    return avg_mse
## ====================================== Plotting function =========================================== ##
def show_plot(y):
    y_values = [round(val, 4) for val in y]
    x = np.arange(len(y_values))  # the label locations
    width = 0.35  # the width of the bars
    labels = [str(n+1) for n in x]
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, y_values, width)
    #rects2 = ax.bar(x + width / 2, elu_array, width, label='elu activation fn')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('MSE')
    ax.set_xlabel('No. of Perceptrons In First Layer')
    ax.set_title('MSE in K-fold cross validation ')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    #autolabel(rects2)
    fig.tight_layout()

## =========================== choosing a model from performace measure from kfold  ================================= ##
def select_a_model(p_array):
    #max_acc_sigmoid = max(sigmoid_array)
    max_acc_perceptron = np.argmin(p_array) + 1
    return max_acc_perceptron

## ================================= Model Order Selection Using Dtrain ======================================= ##
device = 'cpu'
Dtrain_dataset = p1Dataset('csv/Dtrain.csv')

no_of_perceptron = 10
p_measure = []
for nPerceptron in range(0, no_of_perceptron):
    num_epochs = 1000
    p = kfold(10, num_epochs, nPerceptron+1,  Dtrain_dataset, verbose=False)
    p_measure.append(p)
    print(nPerceptron+1, p)


show_plot(p_measure)
'''
plt.figure()
x_axis = [n+1 for n in range(0, nPerceptron+1)]
plt.bar(x_axis, p_measure)
plt.title('MSE on Dtrain')
plt.xlabel('No. of Perceptrons In First Layer')
plt.ylabel('MSE')

for i in range(0, len(p_measure)):
    plt.annotate(str(p_measure[i]),  xy=(x_axis[i], p_measure[i]+0.25))
'''
selected_model = select_a_model(p_measure)
print('## ============= Model Order Selection Result for Dtrain =============== ##')
print('Competing Models: '+str(no_of_perceptron)+' perceptrons')
print('Selected No. of Perceptrons: ', selected_model)
print('## ========================================================================= ##')

## ============================= Training the selected model and Evaluating with Dtest ============================= ##
net_test = NeuralNet(selected_model).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net_test.parameters(), lr=0.01)
num_epochs = 1500
features, labels = Dtrain_dataset[:]
features = features.to(device)
labels = labels.to(device)

Dtest_dataset = p1Dataset('csv/Dtest.csv')
features_test, labels_test = Dtest_dataset[:]
features_test = features_test.to(device)
labels_test = labels_test.to(device)


for epoch in range(num_epochs):
    # Forward pass
    outputs = net_test(features)
    loss = criterion(outputs, labels)
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    dtrain_mse = loss.item()
print('Dtrain mse:', dtrain_mse)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    outputs_test = net_test(features_test)
    mse = criterion(outputs_test, labels_test)
    print('MSE on Dtest: ', mse.item())

    plt.figure()
    plt.title('Prediction vs Dtest')
    plt.scatter(features_test, labels_test)
    plt.scatter(features_test, outputs_test, cmap=cm.gray)
    plt.legend(['Dtest', 'Prediction'])
    plt.xlabel('x1')
    plt.ylabel('x2')


print('## ========================================================================= ##')
## ====================================== Display Plots ===================================== ##
plt.show()

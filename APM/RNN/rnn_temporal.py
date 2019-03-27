
# coding: utf-8

# https://www.cpuheater.com/deep-learning/introduction-to-recurrent-neural-networks-in-pytorch/

# In[1]:


import torch as tc
from torch.autograd import Variable
import pandas as pd
import numpy as np
import pylab as pl


# In[2]:


dtype = tc.FloatTensor
input_size, hidden_size, output_size = 10, 60, 10
epochs = 300
seq_length = 20
lr = 0.1


# In[3]:


#data_time_steps = np.linspace(2, 10, seq_length + 1)
#data = np.sin(data_time_steps)
data_df = pd.read_csv('../../1808-12.csv', names=["mid", "Time", "cpu_usage"])
data = data_df['cpu_usage']
data = list(data)
data = np.array(data)
data = tc.Tensor(data)
data = data.view(1,-1)
#data.resize((len(data)+output_size, 1))


# In[4]:


x = Variable(tc.Tensor(data[:-1*output_size]).type(dtype), requires_grad=False)
y = Variable(tc.Tensor(data[output_size:]).type(dtype), requires_grad=False)


# In[5]:


class RNN(tc.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.i2h = tc.nn.Linear(input_size+hidden_size, hidden_size)
#         self.i2h.weight = tc.nn.Parameter(tc.Tensor(np.random.normal(0,0.4,hidden_size*(input_size+hidden_size))).view(hidden_size, -1))
#         self.i2o = tc.nn.Linear(input_size+hidden_size, output_size)
        self.h2o = tc.nn.Linear(hidden_size, output_size)
#         self.h2o.weight = tc.nn.Parameter(tc.Tensor(np.random.normal(0,0.3,hidden_size*output_size)).view(output_size, -1))
        self.tanh = tc.sigmoid

    def forward(self, input, hidden):
        combined = tc.cat((input, hidden), 1)
        hidden = self.i2h(combined)
#         output = self.i2o(combined)
        hidden = self.tanh(hidden)
        output = self.h2o(hidden)
        return output, hidden


# In[6]:


rnn = RNN(input_size, hidden_size, output_size)
criterion = tc.nn.MSELoss()
optimizer = tc.optim.SGD(rnn.parameters(), lr=lr)


# In[7]:


for i in range(epochs):
    total_loss = 0
    hidden = Variable(tc.zeros((1, hidden_size)).type(dtype), requires_grad=True)
    for j in range(x.size(0)):
        optimizer.zero_grad()
        
        input = x[j:(j+input_size)]
        target = y[j:(j+output_size)]
        output, hidden = rnn(input, hidden)
        loss = criterion(output, target)
        total_loss += loss
        loss.backward(requires_grad=True)
#         print(j, hidden)
        optimizer.step()
#         hidden = Variable(hidden.data, requires_grad=True)

    if i%10 == 0:
        print("Epoch: {}\tLoss: {}".format(i, total_loss))


# In[8]:


hidden = Variable(tc.zeros((1, hidden_size)).type(dtype), requires_grad=False)
predictions = []
for i in range(x.size(0)):
    input = x[i:i+1]
    output, hidden = rnn(input, hidden)
    predictions.append(output.data.numpy().ravel()[0])


# In[9]:


import matplotlib.pyplot as plt
plt.plot(y.view(1,-1).data[0].tolist(), 'o', label="Real")
plt.plot(predictions, 'o', label="Pred")
plt.legend()
plt.show()


# `retain_graph=True vs Variable(hidden.data, requires_grad=True)`

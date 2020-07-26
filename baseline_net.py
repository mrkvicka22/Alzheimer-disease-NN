import pandas as pd
import matplotlib.pyplot as plt
from torch import optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
print("importing finished")

csv_path = r"C:\Users\Matej\Documents\machine learn stuff\TADPOLE_D1_D2.csv"
data = pd.read_csv(csv_path)
df = pd.DataFrame(data, columns=["DX", "ADAS13", "AGE", "PTGENDER", "CDRSB", "Hippocampus", "WholeBrain",
                                 "Entorhinal", "MidTemp"])

TRANSLATOR = {"Dementia":0, "MCI to Dementia":1, "MCI": 2, "NL to MCI":3, "NL": 4}
BATCH_SIZE = 1
'''
The main measures to be predicted: DX, ADAS13, Ventricles
Cognitive tests: CDRSB, ADAS11, MMSE, RAVLT_immediate
MRI measures: Hippocampus, WholeBrain, Entorhinal, MidTemp
PET measures: FDG, AV45
CSF measures: ABETA_UPENNBIOMK9_04_19_17  (amyloid-beta level in CSF), TAU_UPENNBIOMK9_04_19_17 (tau level), PTAU_UPENNBIOMK9_04_19_17 (phosphorylated tau level)
Risk factors: APOE4, AGE
'''


'''
ARCHITECTURE:
---------------
Inputs:

4 inputs for cognitive tests
4 inputs for MRI 
2 inputs for PET
2 inputs for risk factors
----
12 inputs


Outputs:
(DX)
- Dementia
- MCI
- NL
5 outputs
'''


# normalize data to be between 0 and 1
df = df.replace("Female", 1)
df = df.replace("Male", 0)
df = df.replace("Dementia to MCI", "MCI to Dementia")
df = df.replace("MCI to NL", "NL to MCI")

df = df.dropna()
print(df["DX"].value_counts())
for key in df.keys():
    try:
        df[key] = df[key] / max(df[key])
    except:
        print(key)



def create_batches(dataframe, batchsize):
    batches = []
    print(len(dataframe.index))
    for i in range(0,len(dataframe.index)-batchsize,batchsize):
        labels = []
        inputs = []
        for j in range(batchsize):
            inputs.append(dataframe.iloc[[i]].loc[:, dataframe.columns != "DX"].values.flatten())
            labels.append(TRANSLATOR[dataframe.iloc[i]["DX"]])
        labels = torch.tensor(labels)
        inputs = torch.tensor(inputs).float()
        batches.append([labels,inputs])
    print("finished batching up")
    return batches



size = round(len(df.index)*0.75)


train_data, test_data = create_batches(df[:size].sample(frac=1),BATCH_SIZE), create_batches(df[size:],1)
print(train_data)

#define a neural net

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(8,40)
        self.lin2 = nn.Linear(40, 80)
        self.lin3 = nn.Linear(80, 120)
        self.lin4 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 5)

    def forward(self, x):

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        x = self.output(x)
        return x


#training
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0)
start = time.time()
for epoch in range(4):

    running_loss = 0.0
    for i in range(0,len(train_data)):
        labels, inputs = train_data[i]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 500 == 499:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0

# testing
running_loss = 0.0
correct = 0
for i in range(0, len(test_data)):
    labels, inputs = test_data[i]

    optimizer.zero_grad()

    correct += labels == torch.max(net(inputs),1)[1]


print(torch.true_divide(correct,len(test_data)))

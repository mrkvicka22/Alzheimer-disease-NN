from sklearn.ensemble import RandomForestClassifier
import pandas as pd

PERCENT = 0.8

# PREPROCESSING

csv_path = r"C:\Users\Matej\Documents\machine learn stuff\TADPOLE_D1_D2.csv"
data = pd.read_csv(csv_path)
df = pd.DataFrame(data, columns=["DX", "ADAS11", "MMSE", "RAVLT_immediate", "AGE", "PTGENDER", "CDRSB", "Hippocampus",
                                 "WholeBrain",
                                 "Entorhinal", "MidTemp", "APOE4"])

# converting male and female to 1 and 0
df = df.replace("Female", 1)
df = df.replace("Male", 0)

# getting rid of inconsistencies
df = df.replace("Dementia to MCI", "MCI to Dementia")
df = df.replace("MCI to NL", "NL to MCI")
df = df.dropna()

inputs = df.iloc[:, 1:].to_numpy()
labels = df.iloc[:, 0].to_numpy()

test_scores = []
for crit in ["gini","entropy"]:
    for depth in range(5,30):
        for i in range(1,10):
            PERCENT = i/10

            # splitting into training and testing data sets
            len_inputs = len(inputs)
            cut = round(PERCENT * len_inputs)
            train_inputs, test_inputs = inputs[:cut], inputs[cut:]
            train_labels, test_labels = labels[:cut], labels[cut:]


            clf = RandomForestClassifier(max_depth=10,criterion=crit)
            clf = clf.fit(train_inputs, train_labels)

            # test
            correct = 0
            for test_inp, test_lab in zip(test_inputs, test_labels):
                correct += clf.predict([test_inp]) == [test_lab]

            result = (correct / len(test_labels) * 100)[0]
            print(result, "%", f"when trained on {PERCENT * 100}% of the data with depth {depth} and {crit}")
            test_scores.append([result,PERCENT,depth,crit])

test_scores.sort(key=lambda x: x[0])
print(f"The best random forest achieved an accuracy of {test_scores[-1][0]}% when trained on {test_scores[-1][1] * 100}% of the data with depth {test_scores[-1][2]} and {test_scores[-1][3]}")

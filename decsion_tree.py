from sklearn import tree
import pandas as pd
import graphviz

PERCENT = 0.7

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


inputs = df.iloc[:,1:].to_numpy()
labels = df.iloc[:, 0].to_numpy()


#splitting into training and testing data sets
len_inputs = len(inputs)
cut = round(PERCENT * len_inputs)
train_inputs, test_inputs = inputs[:cut], inputs[cut:]
train_labels,test_labels = labels[:cut], labels[cut:]

clf = tree.DecisionTreeClassifier(max_depth=10)
clf = clf.fit(train_inputs, train_labels)

dot_data = tree.export_graphviz(clf, out_file=None,
                                class_names=["Dementia", "MCI to Dementia", "MCI", "NL to MCI", "NL"],feature_names=["ADAS11", "MMSE", "RAVLT_immediate", "AGE",
                                                                  "PTGENDER", "CDRSB", "Hippocampus", "WholeBrain",
                                                                  "Entorhinal", "MidTemp", "APOE4"])
graph = graphviz.Source(dot_data)
graph.render("decision_tree")
r = tree.export_text(clf,feature_names=["ADAS11", "MMSE", "RAVLT_immediate", "AGE", "PTGENDER", "CDRSB", "Hippocampus","WholeBrain","Entorhinal", "MidTemp", "APOE4"])


#test
correct = 0
for test_inp,test_lab in zip(test_inputs,test_labels):
    correct += clf.predict([test_inp]) == [test_lab]

print((correct/len(test_labels)*100)[0], "%", f"when trained on {PERCENT*100}% of the data and depth {clf.tree_.max_depth}")



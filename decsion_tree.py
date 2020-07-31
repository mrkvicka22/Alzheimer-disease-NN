from sklearn import tree
import pandas as pd
import graphviz
import pickle

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

max_result = 82.77
best_tree = None
for depth in range(3,10):
    for min_s in range(5,100,5):
        for crit in ["gini","entropy"]:
            for percent in range(40,86,5):
                PERCENT = percent/100
                cut = round(PERCENT * len_inputs)
                train_inputs, test_inputs = inputs[:cut], inputs[cut:]
                train_labels, test_labels = labels[:cut], labels[cut:]
                clf = tree.DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_s,criterion=crit)
                clf = clf.fit(train_inputs, train_labels)
                #test
                correct = 0
                for test_inp,test_lab in zip(test_inputs,test_labels):
                    correct += clf.predict([test_inp]) == [test_lab]

                result = (correct / len(test_labels) * 100)[0]

                if result > max_result:
                    max_result = result
                    best_tree = clf
                    print("new best tree with performance:",end=" ")
                print(result, "%", f"when trained on {percent}% of the data and depth {depth} and {crit} and min_leaf_samples:{min_s}")

# DISPLAY TREE

dot_data = tree.export_graphviz(best_tree, out_file=None,
                                class_names=["Dementia", "MCI to Dementia", "MCI", "NL to MCI", "NL"],feature_names=["ADAS11", "MMSE", "RAVLT_immediate", "AGE",
                                                                  "PTGENDER", "CDRSB", "Hippocampus", "WholeBrain",
                                                                  "Entorhinal", "MidTemp", "APOE4"])
graph = graphviz.Source(dot_data)
graph.render()
r = tree.export_text(best_tree,feature_names=["ADAS11", "MMSE", "RAVLT_immediate", "AGE", "PTGENDER", "CDRSB", "Hippocampus","WholeBrain","Entorhinal", "MidTemp", "APOE4"])
with open('best_tree.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(best_tree, f, pickle.HIGHEST_PROTOCOL)

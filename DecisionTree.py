
# coding: utf-8

# In[ ]:

import sys
trainFile = sys.argv[1]
testFile = sys.argv[2]
with open(trainFile) as f:
    trainlines = f.readlines()

with open(testFile) as o:
    testlines = o.readlines()

if 'synthetic.social' in sys.argv[1]:
    parameter = 120
elif 'nursery' in sys.argv[1]:
    parameter = 7
elif 'led' in sys.argv[1]:
    parameter = 7
else:
    parameter = 0


# In[ ]:

import re
from collections import Counter
def most_common(lst):
    """
    Returns most common value in list
    """
    data = Counter(lst)
    return max(lst, key=data.get)


# In[ ]:

traindf = []
for line in trainlines:
    traindf.append(line.rstrip('\n'))


# In[ ]:

testdf = []
for line in testlines:
    testdf.append(line.rstrip('\n'))


# In[ ]:

def count(name, value, target_label, labels):
    """
    Returns a dictionary with the all the attributes and a count of their unique values
    """
    dictionary = {} 
    for att in name[0]:
        dictionary[att] = {}
    for idx, label in enumerate(labels):
        for col in range(len(name[idx])):
            if not value[idx][col] in dictionary[name[idx][col]]:
                dictionary[name[idx][col]][value[idx][col]] = 0
            if label==target_label:
                dictionary[name[idx][col]][value[idx][col]] += 1
    return dictionary


# In[ ]:

def preprocess(data):
    """
    Preprocesses the data to retrieve important information
    """
    # list of train labels
    labels = []
    for line in range(len(data)):
        labels.append(data[line][0])
    # define size of training set
    total = len(labels)
    # tuples of the <index>:<value>...<index>:<value> of each line of data as type string
    tups = []
    for line in range(len(data)):
        tups.append(tuple(re.match("(.*):(.*)", data[line][2:]).group().replace(":",",").replace(" ",",").split(",")))
    
    return labels, total, tups


# In[ ]:

def get_count(labels, tups):
    """
    Returns class dictionary, attribute dictionary, attributes, attribute & values matrices
    """
    # list of tuples( attribute names ) for each line of data
    attribute_name = []
    for line in range(len(tups)):
        line_atts = tuple(tups[line][value] for value in range(len(tups[0])) if (value%2==0))
        attribute_name.append(line_atts)
    attribute_list = list(attribute_name[0])
    attribute_val = []
    for line in range(len(tups)):
        line_vals = tuple(tups[line][value] for value in range(len(tups[0])) if (value%2!=0))
        attribute_val.append(line_vals)
    # Count the unique values for each attribute out of four total
    count_val = {}
    for attribute in attribute_name[0]:
        count_val[attribute]={}
    for line in range(len(attribute_name)):
        for col in range(len(attribute_name[line])):
            if not attribute_val[line][col] in count_val[attribute_name[line][col]]:
                count_val[attribute_name[line][col]][attribute_val[line][col]] = 1
            else: count_val[attribute_name[line][col]][attribute_val[line][col]] += 1
    # Count the number of values in a attribute given a class label
    count_class = {}
    for lab in set(labels):
        count_class[lab] = count(attribute_name, attribute_val, lab, labels)

    return count_class, count_val, attribute_list, attribute_name, attribute_val


# In[ ]:

train_labels, train_total, train_tups = preprocess(traindf)
train_dc, train_dv, train_attributes, train_name, train_values = get_count(train_labels, train_tups)


# In[ ]:

test_labels, test_total, test_tups = preprocess(testdf)
test_dc, test_dv, test_attributes, test_name, test_values = get_count(test_labels, test_tups)


# In[ ]:

def filter_tuples(tuples, a_names, a_values, labels, best, value):
    new_tuples = []
    new_labels = []
    for i in range(len(a_names)):
        for j in range(len(a_names[0])):
            if ((best==a_names[i][j]) and (value==a_values[i][j])):
                new_tuples.append(tuples[i])
                new_labels.append(labels[i])
    return new_tuples, new_labels


# In[ ]:

def get_values(values, target_att):
    """
    values is a dict; target_att will be splitting attribute for decision tree
    """
    return values[target_att].keys()


# In[ ]:

def gini(classes, values, target_att, att_value):
    """
    Calculate gini(D) without dividing by total D
    """
    b = []
    for cl in classes.keys():
        b.append(classes[cl][target_att][att_value])
    g = values[target_att][att_value]
    t = 1
    for j in range(len(b)):
        t -= (b[j]/values[target_att][att_value])**2
    f = g*t
    return f


# In[ ]:

def gini_split(classes, values, target_att):
    """
    Calculates gini split.
    classes = dictionary of labels and count of attribute values (count_class)
    values = dictionary of unique values for each attribute (count_val)
    """
    g_split = []
    for key in values[target_att].keys():
        g_split.append(gini(classes, values, target_att, key))
    return sum(g_split)


# In[ ]:

def splitting_attribute(classes, values, att_list):
    """
    Choose attribute to split on 
    """
    a_list = att_list
    ls = []
    a = str()
    for att in a_list:
        ls.append(gini_split(classes, values, att))
    for idx, v in enumerate(ls):
        if v==min(ls):
            a = a_list[idx]
    return a


# In[ ]:

def majority_class(classes, labels, target_attr):
    """
    Returns majority class of target attribute 
    """
    count_list = []
    label = []
    if (target_attr == 0):
        lab = most_common(labels)
    else:
        for cls in classes.keys():
            temp = classes[cls]
            count_list.append(sum(temp[target_attr].values()))
            label.append(cls)
        for idx, val in enumerate(count_list):
            if val==max(count_list):
                lab = label[idx]
    return lab


# In[ ]:

def create_tree(tuples, labels, attributes, height):
    """
    Returns a new decision tree based on the examples given.
    """ 
    max_depth = height
#     classes, values, attributes, labels, total, attribute_name, attribute_values = preprocess(data)
    dclasses, dvalues, new_attributes, a_names, a_values = get_count(labels, tuples)
    #     Create a node N
    tree = {}
#     If tuples in D are all from same class, C, return N as a leaf node labeled class C;
    if (labels.count(labels[0]) == len(labels)):
        return labels[0]
#     If attribute list empty, return N as a leaf node labeled w/majority class in D;majority voting
    elif not tuples or (len(new_attributes) <= 0) or max_depth==len(new_attributes):
        return majority_class(dclasses, labels, 0)
    else:
        max_depth += 1
#     Apply Attribute selection method(D, attribute list) to ﬁnd the “best” splitting criterion
        best = splitting_attribute(dclasses, dvalues, new_attributes)
#     Label node N with splitting criterion
        tree= {best: {}}
#     For full split, attribute list = att list −splitting att; //remove splitting attribute 
        if best in new_attributes:
            attr_list = [attr for attr in new_attributes if attr != best]
#     For each outcome j of splitting criterion // partition tuples, grow subtrees for each partition 
        for val in get_values(dvalues, best):
#         Let Dj be the set of data tuples in D satisfying outcome j; // a partition 
            new_tuples, new_labels = filter_tuples(tuples, a_names, a_values, labels, best, val)
            subtree = create_tree(new_tuples, new_labels, attr_list, max_depth)
#         if Dj is empty then attach a leaf labeled with the majority class in D to node N
            if not subtree:
                tree[best][val] = majority_class(dclasses, labels, 0)
#         else attach the node returned by Generate decision tree(Dj, attribute list) to node N; 
            else: tree[best][val] = subtree
#     endfor
#     return N;
    return tree


# In[ ]:

def classify(tree, tup, labels):
    t = tup
    class_label = tree
    a_name = tuple(tup[index] for index in range(len(tup)) if (index%2 == 0))
    a_value = tuple(tup[index] for index in range(len(tup)) if (index%2 != 0))
    if(isinstance(tree, dict)):
        copy_tree = tree.copy()
        for k, v in copy_tree.items():
            index = a_name.index(k)
            value = a_value[index]
            try:
                class_label = classify(copy_tree[k][value], t, labels)
            except KeyError:
                class_label = most_common(labels)
                    
    return class_label


# In[ ]:

def predict_labels(tree, tuplist, trn_labels):
    labels = []
    for i in range(len(tuplist)):
        labels.append(classify(tree, tuplist[i], trn_labels))
    return labels


# In[ ]:

train_tree = create_tree(train_tups, train_labels, train_attributes, 3)


# In[ ]:

predictions = predict_labels(train_tree, test_tups, train_labels)
k = len(set(train_labels))
uniq_labels= sorted(list(set(train_labels)))


# In[ ]:

actual_idx = [uniq_labels.index(test_labels[x]) for x in range(len(test_labels))]
predicted_idx = [uniq_labels.index(predictions[x]) for x in range(len(predictions))]


# In[ ]:

combine_idx =[]
for i in range(len(actual_idx)):
    combine_idx.append(tuple([actual_idx[i], predicted_idx[i]]))

count_idx = []
for i in set(combine_idx):
    count_idx.append(combine_idx.count(i))
count_idx

row = []
l = 0
while(l!=len(uniq_labels)):
    m = 0
    while(m!=len(uniq_labels)):
        row.append(combine_idx.count((l,m)))
        m +=1
    l +=1


# In[ ]:

def createMatrix(rowCount, colCount, dataList):
    """
    Returns a matrix in list form
    """ 
    matrix = []
    for i in range(rowCount):
        rowList = []
        for j in range(colCount):
            rowList.append(dataList[rowCount * i + j])
        matrix.append(rowList)
    return matrix


# In[ ]:

confusion_matrix = createMatrix(k, k, row)


# In[ ]:

for row in confusion_matrix:
    print(' '.join(map(str,row)))


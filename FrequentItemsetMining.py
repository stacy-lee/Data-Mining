# Import data and set it up for data manipulation
get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np

f = open('data.txt','r')
contents = pd.Series(f.read())
srs = pd.Series([j for i in contents for j in i.split('\n')])
seq = pd.Series([j for i in contents for j in i.split('\n')]).drop(srs.index[0])

transition = seq.str.split(', ').tolist()
support = float(srs.str.split(',').str.get(0)[0])
resilience = float(srs.str.split(',').str.get(1)[0])

ls = []
for i in transition:
    ls.append([int(x) for x in i])

sequences = np.asarray(ls)


#Creates a list of frequent 1-itemsets
def find_freq_one_item(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))


# Scans dataset to find the number of times the subsets in the frequent k-itemset appear
# Returns a list with the itemsets satisfying minsup, and the support for all data stored in a dict
def scanD(D, Ck, minSupport):
    suppCount = {}
    for transaction in D:
        for candidate in Ck:
            if candidate.issubset(transaction):
                if not candidate in suppCount: suppCount[candidate]=1
                else: suppCount[candidate] += 1
    numSeq = float(len(D))
    itemsets = []
    for key in suppCount:
        support = suppCount[key]/numSeq
        if support >= minSupport:
            itemsets.insert(0,key)
    return itemsets, suppCount


# Creates Ck
def aprioriGen(Lk, k):
    Ck = []
    all_Lk = len(Lk)
    for i in range(all_Lk):
        for j in range(i+1, all_Lk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1==L2: #if first k-2 elements are equal
                Ck.append(Lk[i] | Lk[j]) #set union
    return Ck


# Implements Apriori algorithm
def apriori(dataSet, minSupport):
    C1 = find_freq_one_item(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)#scan DB to get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


freqlist, suppData1 = apriori(ls, support)


def contains_sublist(lst, sublist):
    n = len(sublist)
    return any(set(sublist).issubset(lst[i:i+n]) for i in range(len(lst)-n+1))


def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))   # indices = 0, 1
    yield tuple(pool[i] for i in indices)
    while True:
        for i in (range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)


def count_min(S, item):
    # Create combinations of all possible sized sequences in sequence (with correct order)
    matches = {}
    for size in range(2,len(S)+1):
        for tup in combinations(S, size):
            if set(item).issubset(tup):
                while(len(matches)==0):
                    matches[0] = list(tup)
    count = [x for x in matches[0] if x not in item]
    return len(count)


# Create item list of all the frequent itemsets
itemlists = {}
rk = 0
kitem = {}
for i in freqlist:
    for j in i:
        kitem[rk]= sorted(list(j))
        itemlists.update(kitem)
        rk += 1


# Find the minimum number of outliers and save values to a matrix with default values of -1
min_outliers = np.full((len(sequences), len(itemlists)), -1) # matrix(3 rows,19 columns) of -1


for idx in range(len(itemlists)):  # Index of Frequent Itemsets = 0,...,18
    for line in range(len(sequences)): # line = 0,1,2
        if (set(itemlists[idx]).issubset(sequences[line])):  # If the itemset values are in the sequence
                                         # Initialize list of number of outliers in each interval
            if (len(itemlists[idx]) == 1) or contains_sublist(sequences[line],itemlists[idx]):
                min_outliers[line][idx] = 0          # if its a 1-itemset or no numbers in between, there are zero outliers
            else:
                min_outliers[line][idx]= count_min(sequences[line],itemlists[idx])


window = np.zeros(len(itemlists))
for idx in range(len(itemlists)):  # Index of Frequent Itemsets = 0,...,18
    for line in range(len(sequences)): # line = 0,1,2
        if (min_outliers[line][idx] >= 0 and min_outliers[line][idx]/len(itemlists[idx]) <= resilience):
            window[idx] += 1


resilient = []
for idx in range(len(itemlists)):
    if (window[idx]/len(sequences) >= support):
        resilient.append(itemlists[idx])


with open("output.txt", "w") as f:
    for item in resilient:
        f.write("%s\n" % str(item)[1:-1])

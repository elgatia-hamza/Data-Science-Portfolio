# Association Rule
#########################
# Apriori Algorithm
#########################


# Importing the libraries
import pandas as pd
import numpy as np

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
#dataset.drop(np.nan, axis=1,inplace=False)
#r=list(dataset.values)

# Data Preprocessing
transactions = []
for i in range(dataset.shape[0]):
    transactions.append([str(dataset.values[i,j]) for j in range(dataset.shape[1]) if str(dataset.values[i,j])  != 'nan' ])
            
# Training Apriori Algorithme on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence=0.2, min_lift=3, min_hength=2)


# Exporting the results
results = list(rules)
results_list = []
for i in range(len(results)):
    results_list.append('RULE:\t'+str(results[i][0])+'\nSUPPROT:\t'+str(results[i][1]))
            

#print(results)
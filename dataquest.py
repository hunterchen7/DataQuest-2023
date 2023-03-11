import csv
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

source = []
target = []

with open('data/train_data.csv') as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  i = False
  for row in csv_reader:
    colums = []
    rowModified = []
    if not i: 
      colums = row
      i = True
      continue # skip the first row
    for j,cell in zip(range(19),row):
      if j in {1,2,3,4,5,6,8,10,11,13,14,15,17}: 
        # leadTime, arrivalYear, arrivalMonth, arrivalDate, numWeekendNights, numWeekNights, parking, etc (already numeric values)
        rowModified.append(int(cell))
      elif j == 7:
        rowModified += [int(cell[-1] == x) for x in ['1', '2']]
      elif j in {9}: # room type
        rowModified.append(int(cell[-1]))
      elif j == 12: # market segment
        rowModified += [int(cell == x) for x in ['Aviation', 'Complementary', 'Corporate', 'Offline', 'Online']]
      elif j == 16: #
        rowModified.append(float(cell))
      elif j == 18: # cancelled col
        target.append(int(cell == 'cancelled'))
    source.append(rowModified)

'''for s,t in zip(source,target):
  print(s,t)
'''

rf = RandomForestClassifier()
rf.fit(source, target)
      
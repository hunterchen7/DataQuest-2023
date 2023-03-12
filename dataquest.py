import csv
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from scipy.stats import randint

# visualization
from sklearn.tree import export_graphviz
from IPython.display import Image, display
import graphviz

source = []
target = []

columns = [
  'leadTime',
  'arrivalDayOfYear',
  'numWeekendNights',
  'numWeekNights',
  'totalNights'
  'mealPlan1',
  'mealPlan2',
  'roomType',
  'aviation',
  'complementary',
  'corporate',
  'offline',
  'online',
  'numAdults',
  'numChildren',
  'numGuests',
  'hasChildren',
  'repeatedGuests',
  'numPrevCancellations',
  'numPRevNonCancellations',
  'avgRoomPrice',
  'specialRequests',
]

def process_data(csv_reader, isTraining=True):
  csv_reader = list(csv_reader)
  source = []
  if not isTraining:
    for i,row in enumerate(csv_reader):
      copy_row = row.copy()
      copy_row[10], copy_row[11] = copy_row[11], copy_row[10]
      copy_row[11], copy_row[12] = copy_row[12], copy_row[11]
      csv_reader[i] = copy_row
  first = False
  for row in csv_reader:
    rowModified = []
    if not first:
      # columns = row
      # print(row)
      first = True
      continue # skip the first row
    day = 0
    totalDays = 0
    guests = 0
    cancelRatio = 0
    for j,cell in zip(range(19 if isTraining else 18),row):
      # doesn't include bookingID, parking and year

      if j in {1,5,6,11,13,14,15,17}: 
        rowModified.append(int(cell))
      elif j == 3: # month
        day = int(cell) * 30 # close enough
        # rowModified.append(int(cell))
      elif j == 4: # day
        day += int(cell)
        rowModified.append(day)
      elif j == 5: # num weekend nights
        totalDays += int(cell)
      elif j == 6: # num week nights
        totalDays += int(cell)
        rowModified.append(totalDays) # total nights
      elif j == 7: # skip meal plan
        # continue
        rowModified += [int(cell[-1] == x) for x in ['1', '2']] # one hot encoding if meal plan is 1 or 2 (or something else)
      elif j in {9}: # room type
        # ranking from worst room type to best room type
        # 1: standard (1)
        # 2: connecting (7)
        # 3: deluxe (4)
        # 4: suite (6)
        # 5: boutique (5)
        # 6: executive room (2)
        # 7: presidential suite (3)
        '''room_conversion = {
          1: 1,
          2: 7,
          3: 4,
          4: 6,
          5: 5,
          6: 2,
          7: 3,
        }
        rowModified.append(int(cell[-1]))'''
        # try grouping into 3 categories:
        # 1: standard (accounts for 2/3+)
        # 2: deluxe, suite and connecting (accounts for ~1/5)
        # 3: executive room, presidential suite and boutique (accounts for the rest)
        room_conversion = {
          1: 1,
          2: 3,
          3: 3,
          4: 2,
          5: 3,
          6: 2,
          7: 2,
        }
        # rowModified += [int(room_conversion[int(cell[-1])] == x) for x in [1,2,3]] # one hot encoding
        rowModified.append(room_conversion[int(cell[-1])]) # label encoding
      elif j == 12: # market segment
        rowModified += [int(cell == x) for x in ['Aviation', 'Complementary', 'Corporate', 'Offline', 'Online']]
      elif j == 10: # adults
        guests += int(cell)
        rowModified.append(int(cell))
      elif j == 11: # children
        guests += int(cell)
        rowModified.append(guests) # total guests
        rowModified.append(int(cell >= 1)) # has children
      elif j == 14: # prev cancellations
        cancelRatio = int(cell)
      elif j == 15: # prev non cancels
        rowModified.append(min(1, cancelRatio) if int(cell) == 0 else cancelRatio/int(cell)) # add ratio
      elif j == 16: # avg room price
        rowModified.append(float(cell))
      elif j == 18: # canceled col
        target.append(int(cell == 'Canceled'))
    source.append(rowModified)
  return source


train_source = []
with open('data/train_data.csv') as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  train_source = process_data(csv_reader)

test_source = []
with open('data/test_data.csv') as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  test_source = process_data(csv_reader, isTraining=False)

rf = RandomForestClassifier(max_depth=24, n_estimators=1000)
rf.fit(train_source, target)
test_pred = rf.predict(test_source)
# print('accuracy: ', accuracy_score(test_pred, target))
# print('precision: ', precision_score(test_pred, target))
# print('recall: ', recall_score(test_pred, target))
# print('f1: ', f1_score(test_pred, target))

with open('test_data_predicted.csv', 'w', newline='') as f:
  writer = csv.writer(f)
  with open('data/test_data.csv') as csv_file:
    csv_reader = list(csv.reader(csv_file, delimiter=','))

    for i, row in enumerate(csv_reader):
      if i == 0:
        writer.writerow(row)
      else:
        row[-1] = 'Canceled' if test_pred[i-1] == 1 else 'Not_Canceled'
        writer.writerow(row)


'''for _ in range(5): # testing
  X_train, X_test, y_train, y_test = train_test_split(source, target, test_size=0.2)
  
  gbc = GradientBoostingClassifier()
  erf = ExtraTreesClassifier()
  rf.fit(X_train, y_train)
  y_pred = rf.predict(X_test)
  print('accuracy: ', accuracy_score(y_test, y_pred))
  print('precision: ', precision_score(y_test, y_pred))
  print('recall: ', recall_score(y_test, y_pred))
  print('f1: ', f1_score(y_test, y_pred))'''

'''


gbc.fit(X_train, y_train)



y_pred_gbc = gbc.predict(X_test) 

#print(X_test[0:10], rf.predict(X_test[0:10]))



accuracy_gbc = accuracy_score(y_test, y_pred_gbc)
print('gbc accuracy: ', accuracy_gbc)
'''

param_dist = { # for randomized search
  'n_estimators': randint(900,1100),
  'max_depth': randint(20, 35),
}

param_grid = { # for grid search
  'n_estimators': list(range(1000, 1050, 1)),
  'max_depth': list(range(17, 22, 1)),
}

# f.write(str(search_rf))
'''search_gbc = RandomizedSearchCV(gbc, param_distributions=param_dist, scoring='accuracy',n_jobs=-1,iter=30)
#search_gbc = GridSearchCV(gbc, param_grid=param_grid, scoring='accuracy',n_jobs=-1)
search_gbc.fit(X_train, y_train)

best_gbc = search_gbc.best_estimator_

print('Best hyperparams gbc:', search_gbc.best_params_)
print('Best accuracy gbc:', search_gbc.best_score_)'''

# testing parameters
#for _ in range(10):
#  '''
#  # Extra Random Forest
#  search_erf = RandomizedSearchCV(erf, param_distributions=param_dist, scoring='accuracy',n_jobs=-1,n_iter=20)
#  # search_rf = GridSearchCV(rf, param_grid=param_grid, scoring='accuracy',n_jobs=-1)
#  search_erf.fit(X_train, y_train)
#
#  best_erf = search_erf.best_estimator_
#  print('Best hyperparams erf:', search_erf.best_params_)
#  print('Best accuracy erf:', search_erf.best_score_)
#
#  # f.write(str(search_gbc))
#
#  f.write('\nbest hyperparams erf: ' + str(search_erf.best_params_) + 'best accuracy rf: ' + str(search_erf.best_score_))
#  '''
#  
#
#  # Random Forest
#
#  for score_type in ['accuracy', 'precision', 'recall', 'f1']:
#    #search_rf = RandomizedSearchCV(rf, param_distributions=param_dist, scoring=score_type,n_jobs=-1,n_iter=10)
#    search_rf = GridSearchCV(rf, param_grid=param_grid, scoring='accuracy',n_jobs=-1)
#    search_rf.fit(X_train, y_train)
#
#    best_rf = search_rf.best_estimator_
#    print('Best hyperparams rf:', search_rf.best_params_)
#    print('Best ' + score_type + ':', search_rf.best_score_)
#
#    # f.write(str(search_gbc))
#     
#    f = open('params.txt', 'a')
#
#    f.write('\nbest hyperparams rf: ' + str(search_rf.best_params_) + 'best ' + score_type + ': ' + str(search_rf.best_score_))
#
#    f.close() # write every iteration separately
#
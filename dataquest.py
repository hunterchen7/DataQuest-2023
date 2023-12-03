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
  header = next(csv_reader)  # Get the column names from the first row
  col_index = {col: header.index(col) for col in header}

  processed_data = []
  for row in csv_reader:
    day, totalDays, guests, cancelRatio = 0, 0, 0, 0
    row_modified = []

    for col, item in zip(header, row):
      cell = item.strip()
      if col in {'LeadTime', 'NumWeekendNights', 'NumWeekNights', 'NumAdults', 'RepeatedGuest', 'NumPrevCancellations', 'NumPreviousNonCancelled', 'SpecialRequests'}:
        row_modified.append(int(cell))
      elif col == 'ArrivalMonth':  # Month
        day = int(cell) * 30  # Approximate days in a month
      elif col == 'ArrivalDate':  # Day
        day += int(cell)
        row_modified.append(day)
      elif col == 'RoomType':  # Room type
        # try grouping into 3 categories:
        # 1: standard (accounts for 2/3+)
        # 2: deluxe, suite and connecting (accounts for ~1/5)
        # 3: executive room, presidential suite and boutique (accounts for the rest)
        room_conversion = {1: 1, 2: 3, 3: 3, 4: 2, 5: 3, 6: 2, 7: 2}
        row_modified.append(room_conversion[int(cell[-1])]) # label encoding
      elif col == 'MarketSegment':  # Market segment
        row_modified.extend(int(cell == x) for x in ['Aviation', 'Complementary', 'Corporate', 'Offline', 'Online'])
      elif col == 'NumAdults':  # Adults
        guests += int(cell)
        row_modified.append(int(cell))
      elif col == 'NumChildren':  # Children
        guests += int(cell)
        row_modified.append(guests)  # Total guests
        row_modified.append(int(cell) >= 1)  # Has children
      elif col == 'NumPrevCancellations':  # Previous cancellations
        cancelRatio = int(cell)
      elif col == 'NumPrevNonCancellations':  # Previous non-cancellations
        row_modified.append(min(1, cancelRatio) if int(cell) == 0 else cancelRatio / int(cell))
      elif col == 'AvgRoomPrice':  # Average room price
        row_modified.append(float(cell))
      elif col == 'BookingStatus' and isTraining:  # Canceled column (only for training)
        target.append(int(cell == 'Canceled'))

    processed_data.append(row_modified)

  return processed_data


train_source = []
with open('data/train_data.csv') as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  train_source = process_data(csv_reader)

test_source = []
with open('data/test_data.csv') as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  test_source = process_data(csv_reader, isTraining=False)

rf = RandomForestClassifier(max_depth=22, n_estimators=996)
X_train, X_test, y_train, y_test = train_test_split(train_source, target, test_size=0.2)
rf.fit(X_train, y_train)
test_pred = rf.predict(X_test)

print('accuracy: ', accuracy_score(y_test, test_pred))
print('precision: ', precision_score(y_test, test_pred))
print('recall: ', recall_score(y_test, test_pred))
print('f1: ', f1_score(y_test, test_pred))

'''rf.fit(train_source, target)
test_pred = rf.predict(test_source)



with open('test_data_predicted.csv', 'w', newline='') as f:
  writer = csv.writer(f)
  with open('data/test_data.csv') as csv_file:
    csv_reader = list(csv.reader(csv_file, delimiter=','))

    for i, row in enumerate(csv_reader):
      if i == 0:
        writer.writerow(row)
      else:
        row[-1] = 'Canceled' if test_pred[i-1] == 1 else 'Not_Canceled'
        writer.writerow(row)'''

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
  'n_estimators': randint(100,150),
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
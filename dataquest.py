import csv

dataModified = []

with open('data/train_data.csv') as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  for i,row in zip(range(5),csv_reader):
    colums = []
    dataModified = []
    if not i: 
      colums = row
      continue # skip the first row
    for j,data in row:
      if j in {1,2,3,4,5,6,8,10,11}: 
        # leadTime, arrivalYear, arrivalMonth, arrivalDate, numWeekendNights, numWeekNights, parking,
        dataModified.append(int(data))
      elif j in {7,9}: # meal plan and room type
        dataModified.append(int(data[-1]))
    elif j == 12: # market segment
      print(row)
      
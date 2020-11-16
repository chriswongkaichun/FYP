import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import time
import PySimpleGUI as sg
import pandas as pd
import gzip
import json
import os
import subprocess

f = ['Times New Roman',20]
save_path = './Result/'

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def input():
  sg.theme('SandyBeach')   # Add a touch of color
  # All the stuff inside your window.
  layout = [  
            [sg.Text('Weclome to the matrix completion platform',font = f)],
            [sg.Button('Get start')]
           ]

  # Create the Window
  window = sg.Window('Welcome', layout)
  # Event Loop to process "events" and get the "values" of the inputs
  while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Get start': # if user closes window or clicks cancel
        break

  window.close()
  
  folder = [name for name in os.listdir(".") if os.path.isdir(name) and os.path.exists('./'+ name +'/'+ name +'_5.json.gz')]
  sg.theme('DarkRed1')   # Add a touch of color
  # All the stuff inside your window.

  first_column = [
                    [sg.Text('Please choose the file name (R)',font = f)],
                    [sg.Listbox(values=folder, enable_events=True, size=(40, 20), key="-FILE LIST-")],
                    [sg.Button('Ok')]
                  ]

  second_column = [
                    [sg.Text('Please fill in the following information for matrix completion.',font = f)],
                    [sg.HorizontalSeparator()],
                    [sg.Text('No. of initializations of user latent factors (L):',font = f)],
                    [sg.InputText(key="-INPUT0-")],
                    [sg.Text('No. of rank',font = f)],
                    [sg.InputText(key="-INPUT1-")],
                    [sg.Text('No. of iterations (T):',font = f)],
                    [sg.InputText(key="-INPUT2-")],
                    [sg.Text('Learning rate for first 2 iterations (c):',font = f)],
                    [sg.InputText(key="-INPUT3-")],
                    [sg.Text('Divided integer for learning rate for other iterations (b):',font = f)],
                    [sg.InputText(key="-INPUT4-")]
                  ]

  third_column = [
                    [sg.Text('The algorithm',font = f)],
                    [sg.Image(filename="./螢幕截圖 2020-11-14 下午4.56.40.png")],
                  ]

  layout = [[sg.Column(first_column), sg.VSeperator(), sg.Column(second_column), sg.VSeperator(), sg.Column(third_column)]]

  # Create the Window
  window = sg.Window('Choose the file', layout)
  # Event Loop to process "events" and get the "values" of the inputs
  while True:
    event, values = window.read()
    if event == 'Ok' or event == sg.WIN_CLOSED:
      e = input1()
      if e == 'Yes':
        break

  window.close()
  return values["-FILE LIST-"][0],values["-INPUT0-"],values["-INPUT1-"],values["-INPUT2-"],values["-INPUT3-"],values["-INPUT4-"]

def input1():
  sg.theme('LightBlue2')   # Add a touch of color
  # All the stuff inside your window.
  layout = [  
              [sg.Text('Start the matrix completion now?',font = f)],
              [sg.Button('Yes'),sg.Button('No')]
           ]

  # Create the Window
  window = sg.Window('Confirm', layout)
  # Event Loop to process "events" and get the "values" of the inputs
  while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Yes' or event == 'No': # if user closes window or clicks cancel
      break

  window.close()
  return event

def showResult(matrix,line_rating,tab,matrixName):
  sg.theme('Topanga')

  first_column = [
                    [sg.Text('Background information of ' + matrixName,font = f)],
                    [sg.HorizontalSeparator()],
                    [sg.Text('Dimension of the matrix: ' + str(matrix.shape[0]) + ' x ' + str(matrix.shape[1]),font = f)],
                    [sg.Text('No. of ratings: ' + str(np.count_nonzero(matrix)),font = f)],
                    [sg.HorizontalSeparator()],
                  ]
  
  for i in range(len(line_rating)):
    first_column.append([sg.Text(str(i+1) + ': ' + str(line_rating[i]),font = f)])

  mean = matrix[matrix != 0].mean()
  first_column.append([sg.HorizontalSeparator()])
  first_column.append([sg.Text('Mean: ' + str(mean),font = f)])
  first_column.append([sg.Text('The matrix is created and separated.',font = f)])
  first_column.append([sg.Button('Open the original matrix'),sg.Button('Open the first matrix'), sg.Button('Open the second matrix')])
  first_column.append([sg.Button('Quit')])
    
  layout = [[sg.Column(first_column), sg.VSeperator(), sg.TabGroup([tab])]]

  # Create the Window
  window = sg.Window('Results', layout)
  # Event Loop to process "events" and get the "values" of the inputs
  while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == "Quit": # if user closes window or clicks cancel
      e = showResult1()
      if e == 'Yes':
        break
    elif event == 'Open the original matrix': # if user closes window or clicks cancel
      subprocess.call(["open", save_path + 'matrix.csv'])
    elif event == 'Open the first matrix':
      subprocess.call(["open", save_path + 'matrix1.csv'])
    elif event == 'Open the second matrix':
      subprocess.call(["open", save_path + 'matrix2.csv'])
    for l in range(L):
      if event == "Open the final matrix 1 with constraint for set %d" %(l+1):
        subprocess.call(["open", save_path + 'matrix_GD_constraint_1_%d.csv' %(l+1)])
      elif event == "Open the final matrix 2 with constraint for set %d" %(l+1):
        subprocess.call(["open", save_path + 'matrix_GD_constraint_2_%d.csv' %(l+1)])

  window.close()
  
  sg.theme('LightPurple')   # Add a touch of color
  # All the stuff inside your window.

  layout = [  [sg.Text('Thank you for using.',font = f)],
              [sg.Button('Bye Bye')]
           ]

  # Create the Window
  window = sg.Window('The End', layout)
  # Event Loop to process "events" and get the "values" of the inputs
  while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Bye Bye': # if user closes window or clicks cancel
      break
    

  window.close()

def showResult1():
  sg.theme('DarkTeal3')   # Add a touch of color
  # All the stuff inside your window.
  layout = [  
              [sg.Text('Are you sure you want to quit?',font = f)],
              [sg.Button('Yes'),sg.Button('No')]
           ]

  # Create the Window
  window = sg.Window('Confirm', layout)
  # Event Loop to process "events" and get the "values" of the inputs
  while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Yes' or event == 'No': # if user closes window or clicks cancel
      break

  window.close()
  return event

def deleteRowAndColumn(matrix):
  # Get the no. of entries filled on each column and rows
  nz0 = np.count_nonzero(matrix, axis=0)
  nz5 = np.count_nonzero(matrix, axis=1)

  while True:
    allOne = False
    notAllTwo1 = []
    notAllTwo0 = []
    allTwo = True

    # Check for each column
    for j in range(nz0.shape[0]):
      if nz0[j] < 2:
        notAllTwo1.append(j)
        allTwo = False
        
    # Check for each row
    for i in range(nz5.shape[0]):
      if nz5[i] < 2:
        notAllTwo0.append(i)
        allTwo = False
    
    # Start the division of the incompleted R if all rows and columns have at least 2 entries filled
    if allTwo == True:
      break

    # Delete the columns that have only one entry
    matrix = np.delete(matrix,notAllTwo1,axis=1)
    matrix = np.delete(matrix,notAllTwo0,axis=0)

    # Get the number of entries filled on each row and columns
    nz0 = np.count_nonzero(matrix, axis=0)
    nz5 = np.count_nonzero(matrix, axis=1)

    # Check whether each of column has only one entry
    for j in range(nz0.shape[0]):
      if nz0[j] < 2:
        allOne = True
      else:
        allOne = False
        break
        
    for i in range(nz5.shape[0]):
      if nz5[i] < 2:
        allOne = True
      else:
        allOne = False
        break
  
    if allOne == True:
      break

  return allOne, matrix
    
def errorMessage():
  sg.theme('LightGrey3')   # Add a touch of color
  # All the stuff inside your window.
  layout = [  [sg.Text('The matrix cannot be filled because each row and column has only one or even no entry filled',font = f)],
           ]
  # Create the Window
  window = sg.Window('Error', layout)
  # Event Loop to process "events" and get the "values" of the inputs
  while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED: # if user closes window or clicks cancel
      break
  window.close()
    
def separateMatrix(matrix):
  nz = np.count_nonzero(matrix)
  matrix1 = np.zeros((matrix.shape[0],matrix.shape[1]))
  matrix2 = np.zeros((matrix.shape[0],matrix.shape[1]))

  nz1 = 0
  nz2 = 0

  for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
      if matrix[i,j] > 0:
        ran = np.random.randint(0,2)
        if (ran == 0 and nz1 < nz/2) or nz2 >= nz/2:
          matrix1[i,j] = matrix[i,j]
          nz1 += 1
        elif (ran == 1 and nz2 < nz/2) or nz1 >= nz/2:
          matrix2[i,j] = matrix[i,j]
          nz2 += 1

  # Make sure that each row and column has at least 1 entry
  for i in range(matrix.shape[0]):
    if matrix1[i,:].any() == False:
      for j in range(matrix.shape[1]):
        nz4 = np.count_nonzero(matrix2[:,j])
        if matrix2[i,j] > 0 and nz4 > 1:
          matrix1[i,j] = matrix2[i,j]
          matrix2[i,j] = 0
          for k in range(matrix.shape[0]):
            nz3 = np.count_nonzero(matrix1[k,:])
            if nz3 > 1:
              switched = False
              for l in range(matrix.shape[1]):
                nz4 = np.count_nonzero(matrix1[:,l])
                if matrix1[k,l] > 0 and nz4 > 1:
                  matrix2[k,l] = matrix1[k,l]
                  matrix1[k,l] = 0
                  switched = True
                  break
              if switched == True:
                break
          break
    if matrix2[i,:].any() == False:
      for j in range(matrix.shape[1]):
        nz4 = np.count_nonzero(matrix1[:,j])
        if matrix1[i,j] > 0 and nz4 > 1:
          matrix2[i,j] = matrix1[i,j]
          matrix1[i,j] = 0
          for k in range(matrix.shape[0]):
            nz3 = np.count_nonzero(matrix2[k,:])
            if nz3 > 1:
              switched = False
              for l in range(matrix.shape[1]):
                nz4 = np.count_nonzero(matrix2[:,l])
                if matrix2[k,l] > 0 and nz4 > 1:
                  matrix1[k,l] = matrix2[k,l]
                  matrix2[k,l] = 0
                  switched = True
                  break
              if switched == True:
                break
          break

  for j in range(matrix.shape[1]):
    if matrix1[:,j].any() == False:
      for i in range(matrix.shape[0]):
        nz4 = np.count_nonzero(matrix2[i,:])
        if matrix2[i,j] > 0 and nz4 > 1:
          matrix1[i,j] = matrix2[i,j]
          matrix2[i,j] = 0
          for l in range(matrix.shape[1]):
            nz3 = np.count_nonzero(matrix1[:,l])
            if nz3 > 1:
              switched = False
              for k in range(matrix.shape[0]):
                nz4 = np.count_nonzero(matrix1[k,:])
                if matrix1[k,l] > 0 and nz4 > 1:
                  matrix2[k,l] = matrix1[k,l]
                  matrix1[k,l] = 0
                  switched = True
                  break
              if switched == True:
                break
          break
    if matrix2[:,j].any() == False:
      for i in range(matrix.shape[0]):
        nz4 = np.count_nonzero(matrix1[i,:])
        if matrix1[i,j] > 0 and nz4 > 1:
          matrix2[i,j] = matrix1[i,j]
          matrix1[i,j] = 0
          for l in range(matrix.shape[1]):
            nz3 = np.count_nonzero(matrix2[:,l])
            if nz3 > 1:
              switched = False
              for k in range(matrix.shape[0]):
                nz4 = np.count_nonzero(matrix2[k,:])
                if matrix2[k,l] > 0 and nz4 > 1:
                  matrix1[k,l] = matrix2[k,l]
                  matrix2[k,l] = 0
                  switched = True
                  break
              if switched == True:
                break
          break
  return matrix2, matrix1
    
def initialization(matrix,R):
  temp = np.random.sample((matrix.shape[0],R))
  P = np.copy(temp)
  Q = np.zeros((matrix.shape[1],R))
  return P, Q
    
def update(matrix, P, Q, X, Y, R, learning_rate):
  for i, j in zip(X,Y):
    predicted_entry = 0
    for r in range(R):
      predicted_entry += P[i,r] * Q[j,r]
    error = matrix[i,j] - predicted_entry
    for r in range(R):
      t2 = P[i,r]
      P[i,r] += learning_rate * (2 * error * Q[j,r])
      Q[j,r] += learning_rate * (2 * error * t2)

  return P,Q

def getError(matrix, P, Q, X, Y, nz, R):
  cost1 = 0
  cost2 = 0
  for x, y in zip(X, Y):
    predicted_entry = 0
    for r in range(R):
      predicted_entry += P[x,r] * Q[y,r]
    cost1 += abs(matrix[x, y] - predicted_entry)
    cost2 += pow(matrix[x, y] - predicted_entry,2)
  mae = cost1 / nz / 4
  mse = cost2 / nz
  return mae, mse

def getErrorAfterConstraint(matrix, predicted, X, Y, nz):
  cost1 = 0
  cost2 = 0
  for x, y in zip(X, Y):
    cost1 += abs(matrix[x, y] - predicted[x, y])
    cost2 += pow(matrix[x, y] - predicted[x, y],2)
  mae = cost1 / nz / 4
  mse = cost2 / nz
  return mae, mse

def getPredictedMatrix(matrix, P, Q, R):
  predicted = np.zeros(matrix.shape)
  for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
      for r in range(R):
        predicted[i,j] += P[i,r] * Q[j,r]
  return predicted
        
def constraint(predicted):
  for i in range(predicted.shape[0]):
    for j in range(predicted.shape[1]):
      if predicted[i,j] < 1.5:
        predicted[i,j] = 1
      elif (predicted[i,j] >= 1.5 and predicted[i,j] < 5.5):
        predicted[i,j] = int(round(predicted[i,j]))
      else:
        predicted[i,j] = 5
  return predicted

values = input()
filename = './'+ values[0] +'/'+ values[0] +'_5.json.gz'
df = getDF(filename)

product = np.array(df.asin.unique().tolist())
len1 = len(product)

reviewer = np.array(df.reviewerID.unique().tolist())
len2 = len(reviewer)

matrix = np.zeros((len2,len1))
for review in parse(filename):
  for p in range(product.shape[0]):
    if product[p] == review['asin']:
      for r in range(reviewer.shape[0]):
        if reviewer[r] == review['reviewerID']:
          matrix[r][p] = review['overall']
          break
      break
    
allOne, matrix = deleteRowAndColumn(matrix)

line_rating = []
    
rating = 1
while True:
  if np.count_nonzero(matrix == rating) == 0:
    break
  line_rating.append(np.count_nonzero(matrix == rating))
  rating += 1
    
np.savetxt(save_path + 'matrix.csv', matrix, delimiter=',')

# Give up to filling the matrix R if each row and column has only one or even no entry filled
if allOne == True:
  errorMessage()
else:
  matrix2, matrix1 = separateMatrix(matrix)
        
  X2, Y2 = matrix2.nonzero()
  X1, Y1 = matrix1.nonzero()
    
  nz3 = np.count_nonzero(matrix2)
  nz4 = np.count_nonzero(matrix1)
    
  np.savetxt(save_path + 'matrix1.csv', matrix1, delimiter=',')
  np.savetxt(save_path + 'matrix2.csv', matrix2, delimiter=',')

  # No. of versions of P1
  L = int(values[1])

  # No. of rank
  R = int(values[2])
    
  # No. of iterations
  T = int(values[3])
    
  # learning rate for first 2 iterations
  lr = float(values[4])
    
  # Divided number
  d = int(values[5])
    
  sg.theme('LightYellow')
    
  tab = []
    
  for l in range(L):
    P1, Q1 = initialization(matrix, R)
    P2 = np.copy(P1)
    Q2 = np.copy(Q1)
    
    line_iter = []
    line_mae2 = []
    line_mse2 = []
    line_mae1 = []
    line_mse1 = []
    line_learning_rate = []
    
    start = time.time()
    
    # Complete R by using GD
    for t in range(T):
      if t <= 1:
        learning_rate = lr
      else:
        diff = ((line_mse2[t-2] - line_mse2[t-1])/ line_mse2[t-2] + (line_mse1[t-2] - line_mse1[t-1])/ line_mse1[t-2])/2
        learning_rate = diff / d
    
      P1, Q1 = update(matrix1, P1, Q1, X1, Y1, R, learning_rate)
      P2, Q2 = update(matrix2, P2, Q2, X2, Y2, R, learning_rate)
      mae2, mse2 = getError(matrix2, P1, Q1, X2, Y2, nz3, R)
      mae1, mse1 = getError(matrix1, P2, Q2, X1, Y1, nz4, R)
    
      line_iter.append(t+1)
      line_mae2.append(mae2)
      line_mse2.append(mse2)
      line_mae1.append(mae1)
      line_mse1.append(mse1)
      line_learning_rate.append(learning_rate)
    
    end = time.time()
    
    output = open(save_path + "before_constraint_%d.txt" %(l+1), "w")
    output.writelines(["MAE of matrix2: %.4f" %mae2,
                       "\n",
                       "MSE of matrix2: %.4f" %mse2,
                       "\n",
                       "MAE of matrix1: %.4f" %mae1,
                       "\n",
                       "MSE of matrix1: %.4f" %mse1])
    output.close()
    
    first_column = [
                    [sg.Text("Before constraining,",font = f)],
                    [sg.Text("MAE of matrix2: %.4f" %mae2,font = f)],
                    [sg.Text("MSE of matrix2: %.4f" %mse2,font = f)],
                    [sg.Text("MAE of matrix1: %.4f" %mae1,font = f)],
                    [sg.Text("MSE of matrix1: %.4f" %mse1,font = f)],
                    [sg.HorizontalSeparator()],
                   ]
    
    predicted1 = getPredictedMatrix(matrix1, P1, Q1, R)
    predicted2 = getPredictedMatrix(matrix2, P2, Q2, R)
    
    np.savetxt(save_path + 'matrix_GD_1_%d.csv' %(l+1), predicted1, delimiter=',')
    np.savetxt(save_path + 'matrix_GD_2_%d.csv' %(l+1), predicted2, delimiter=',')
    
    predicted1 = constraint(predicted1)
    predicted2 = constraint(predicted2)
    
    plt.plot(line_iter, line_mae2, color='blue', lw='1')
    plt.xlabel('iterations')  
    plt.ylabel('MAE')
    plt.xticks(np.arange(0, T+1, 1))
    plt.title("MAE2 vs iterations by GD")
    figure = plt.gcf()
    figure.set_size_inches(4, 3)
    plt.savefig(save_path + 'MAE2_GD_%d.png' %(l+1))
    plt.close(figure)
    
    plt.plot(line_iter, line_mse2, color='blue', lw='1')
    plt.xlabel('iterations')  
    plt.ylabel('MSE')
    plt.xticks(np.arange(0, T+1, 1))
    plt.title("MSE2 vs iterations by GD")
    figure = plt.gcf()
    figure.set_size_inches(4, 3)
    plt.savefig(save_path + 'MSE2_GD_%d.png' %(l+1))
    plt.close(figure)
    
    plt.plot(line_iter, line_mae1, color='blue', lw='1')
    plt.xlabel('iterations')  
    plt.ylabel('MAE')
    plt.xticks(np.arange(0, T+1, 1))
    plt.title("MAE1 vs iterations by GD")
    figure = plt.gcf()
    figure.set_size_inches(4, 3)
    plt.savefig(save_path + 'MAE1_GD_%d.png' %(l+1))
    plt.close(figure)
    
    plt.plot(line_iter, line_mse1, color='blue', lw='1')
    plt.xlabel('iterations')  
    plt.ylabel('MSE')
    plt.xticks(np.arange(0, T+1, 1))
    plt.title("MSE1 vs iterations by GD")
    figure = plt.gcf()
    figure.set_size_inches(4, 3)
    plt.savefig(save_path + 'MSE1_GD_%d.png' %(l+1))
    plt.close(figure)
    
    plt.plot(line_iter, line_learning_rate, color='blue', lw='1')
    plt.xlabel('iterations')  
    plt.ylabel('learning rate')
    plt.xticks(np.arange(0, T+1, 1))
    plt.title("learning rate vs iterations by GD")
    figure = plt.gcf()
    figure.set_size_inches(4, 3)
    plt.savefig(save_path + 'learning_rate_GD_%d.png' %(l+1))
    plt.close(figure)
    
    mae2, mse2 = getErrorAfterConstraint(matrix2, predicted1, X2, Y2, nz3)
    mae1, mse1 = getErrorAfterConstraint(matrix1, predicted2, X1, Y1, nz4)
    
    output = open(save_path + "after_constraint_%d.txt" %(l+1), "w")
    output.writelines(["MAE of matrix2: %.4f" %mae2,
                       "\n",
                       "MSE of matrix2: %.4f" %mse2,
                       "\n",
                       "MAE of matrix1: %.4f" %mae1,
                       "\n",
                       "MSE of matrix1: %.4f" %mse1])
    output.close()
    
    np.savetxt(save_path + 'matrix_GD_constraint_1_%d.csv' %(l+1), predicted1, delimiter=',')
    np.savetxt(save_path + 'matrix_GD_constraint_2_%d.csv' %(l+1), predicted2, delimiter=',')
    
    # All the stuff inside your window.
    first_column.append([sg.Text("After constraining,",font = f)])
    first_column.append([sg.Text("MAE of matrix2: %.4f" %mae2,font = f)])
    first_column.append([sg.Text("MSE of matrix2: %.4f" %mse2,font = f)])
    first_column.append([sg.Text("MAE of matrix1: %.4f" %mae1,font = f)])
    first_column.append([sg.Text("MSE of matrix1: %.4f" %mse1,font = f)])
    first_column.append([sg.HorizontalSeparator()])
    first_column.append([sg.Text("Time used: %f"%(end-start),font = f)])
    first_column.append([sg.HorizontalSeparator()])
    first_column.append([sg.Text("Change of learning rate",font = f)])
    first_column.append([sg.Image(save_path + 'learning_rate_GD_%d.png' %(l+1))])
    
    second_column = [
                     [sg.Text("Errors of " + values[0] + "2",font = f)],
                     [sg.HorizontalSeparator()],
                     [sg.Text("MAE",font = f)],
                     [sg.Image(save_path + 'MAE2_GD_%d.png' %(l+1))],
                     [sg.HorizontalSeparator()],
                     [sg.Text("MSE",font = f)],
                     [sg.Image(save_path + 'MSE2_GD_%d.png' %(l+1))],
                     [sg.HorizontalSeparator()],
                     [sg.Button("Open the final matrix 2 with constraint for set %d" %(l+1))]
                    ]
    
    third_column = [
                     [sg.Text("Errors of " + values[0] + "1",font = f)],
                     [sg.HorizontalSeparator()],
                     [sg.Text("MAE",font = f)],
                     [sg.Image(save_path + 'MAE1_GD_%d.png' %(l+1))],
                     [sg.HorizontalSeparator()],
                     [sg.Text("MSE",font = f)],
                     [sg.Image(save_path + 'MSE1_GD_%d.png' %(l+1))],
                     [sg.HorizontalSeparator()],
                     [sg.Button("Open the final matrix 1 with constraint for set %d" %(l+1))]
                    ]
    
    layout = [[sg.Column(first_column), sg.VSeperator(), sg.Column(second_column), sg.VSeperator(), sg.Column(third_column)]]

    tab.append(sg.Tab('Result of set %d' %(l+1), layout))

  showResult(matrix,line_rating,tab,values[0])
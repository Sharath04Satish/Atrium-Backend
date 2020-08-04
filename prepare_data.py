from glob import glob
import numpy as np

traindata = np.empty(shape=[0, 188])
testdata = np.empty(shape=[0, 188])

paths_train = glob('data_ecg/mitdb/train/*.csv')
paths_test = glob('data_ecg/mitdb/test/*.csv')

for path in paths_train:
    print('Loading ', path)
    csvrows = np.loadtxt(path, delimiter=',')
    traindata = np.append(traindata, csvrows, axis=0)


for path in paths_test:
    print('Loading ', path)
    csvrows = np.loadtxt(path, delimiter=',')
    testdata = np.append(testdata, csvrows, axis=0)

# Randomly mix rows
np.random.shuffle(traindata)
trainrows = len(traindata)

np.random.shuffle(testdata)
testrows = len(testdata)

with open('train.csv', "wb") as fin:
    np.savetxt(fin, traindata, delimiter=",", fmt='%f')
    
with open('test.csv', "wb") as fin:
    np.savetxt(fin, testdata, delimiter=",", fmt='%f')


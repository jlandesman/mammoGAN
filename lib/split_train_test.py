#! /usr/bin/python

import os
import pandas as pd
import sys

if len(sys.argv) != 3:
    print "Enter in the format python split_train_test.py <csv file> <png or tif>"
    exit()
FILE_TYPE = sys.argv[2]
CSV_FILE = sys.argv[1]

mias_mini = pd.read_csv(CSV_FILE)
mias_mini = mias_mini.drop_duplicates('ref_num')
mias_mini.reset_index(drop=True, inplace =True)
abnormality = mias_mini['abnormaility_class'].value_counts().keys()
severity = mias_mini[mias_mini['abnormaility_class']=='CALC']['severity'].value_counts().keys()

mias_mini_train = pd.DataFrame([])
mias_mini_test = pd.DataFrame([])

for item in abnormality:
    abClass = mias_mini[mias_mini['abnormaility_class'] == item]
    if item == "NORM": 
        train_count = int(round(abClass.shape[0] * 0.8))
        mias_mini_train = mias_mini_train.append(abClass.head(train_count))
        mias_mini_test = mias_mini_test.append(abClass.tail(abClass.shape[0] - train_count))
    else:
        for sev in severity:
            sevClass = abClass[abClass['severity'] == sev]
            train_count = int(round(sevClass.shape[0] * 0.8))
            mias_mini_train = mias_mini_train.append(sevClass.head(train_count))
            mias_mini_test = mias_mini_test.append(sevClass.tail(sevClass.shape[0] - train_count))
            
mias_mini_trainSel = mias_mini_train[['ref_num','severity']].copy()
mias_mini_testSel = mias_mini_test[['ref_num','severity']].copy()            

mias_mini_trainSel = mias_mini_trainSel.fillna(0)
convert = {"severity": {'B': 1, 'M': 2 }}
mias_mini_trainSel.replace(convert, inplace=True)
mias_mini_testSel = mias_mini_testSel.fillna(0)
mias_mini_testSel.replace(convert, inplace=True)

if os.path.exists('train'):
    os.system('rm -rf train')
os.system('mkdir train')
os.system('chmod +wx train')
if os.path.exists('test'):
    os.system('rm -rf test')
os.system('mkdir test')
os.system('chmod +wx test')

for name in mias_mini_trainSel['ref_num']:
    #print name

    opFile = "train/" + name +  '.' + FILE_TYPE
    ipFile =  name + '.' + FILE_TYPE
    
    if os.path.isfile(ipFile):
        os.system('cp ' + ipFile + ' ' +opFile)
        
for name in mias_mini_testSel['ref_num']:
    #print name

    opFile = "test/" + name + '.' + FILE_TYPE
    ipFile =  name + '.' + FILE_TYPE
    
    if os.path.isfile(ipFile):
        os.system('cp ' + ipFile + ' ' +opFile)
        
print "training files are in the directory train" 
print "test files are in the directory test"

mias_mini_trainSel['severity'].to_csv("train_labels.csv",index=False)
mias_mini_testSel['severity'].to_csv("test_labels.csv",index=False)
print
print "Train labels in train_labels.csv"
print "Test labels in test_labels.csv"

#! /usr/bin/python

import os
import pandas as pd
import sys

if len(sys.argv) != 4:
    print "Enter in the format python split_train_test.py mias-mini <csv file> <png or tif>"
    exit()
IP_DIR = sys.argv[1]
FILE_TYPE = sys.argv[3]
CSV_FILE = sys.argv[2]
TRAIN_DIR = IP_DIR + '-train'
TEST_DIR = IP_DIR + '-test'

mias_mini = pd.read_csv(CSV_FILE)
mias_mini = mias_mini.drop_duplicates('ref_num')
mias_mini.reset_index(drop=True, inplace =True)
abnormality = mias_mini['abnormaility_class'].value_counts().keys()
severity = mias_mini[mias_mini['abnormaility_class']=='CALC']['severity'].value_counts().keys()

mias_mini_train = pd.DataFrame([])
mias_mini_test = pd.DataFrame([])

'''
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
'''
            
mias_mini_trainSel = mias_mini[['ref_num','severity']].copy()
#mias_mini_testSel = mias_mini[['ref_num','severity']].copy()            

mias_mini_trainSel = mias_mini_trainSel.fillna(0)
convert = {"severity": {'B': 1, 'M': 2 }}
mias_mini_trainSel.replace(convert, inplace=True)

severity = mias_mini_trainSel['severity'].value_counts().keys()
#mias_mini_testSel = mias_mini_testSel.fillna(0)
#mias_mini_testSel.replace(convert, inplace=True)

if os.path.exists(TRAIN_DIR):
    os.system('rm -rf ' + TRAIN_DIR)
os.system('mkdir ' + TRAIN_DIR)
os.system('chmod +wx ' + TRAIN_DIR)

for sev in severity:
    print sev, TRAIN_DIR
    os.system('mkdir ' + TRAIN_DIR + '/' + str(sev))
    os.system('chmod +wx ' + TRAIN_DIR + '/' + str(sev))
#if os.path.exists(TEST_DIR):
#    os.system('rm -rf ' + TEST_DIR)
#os.system('mkdir '+ TEST_DIR)
#os.system('chmod +wx '  + TEST_DIR)

for name in mias_mini_trainSel['ref_num']:
    #print name

    classLabel = mias_mini_trainSel[mias_mini_trainSel['ref_num'] == name]['severity'].item()

    #print classLabel
    opFile = TRAIN_DIR + '/' + str(classLabel) + '/' + name +  '.' + FILE_TYPE
    ipFile =  IP_DIR + '/' +name + '.' + FILE_TYPE
    #print opFile 
    if os.path.isfile(ipFile):
        os.system('cp ' + ipFile + ' ' +opFile)
        
'''
for name in mias_mini_testSel['ref_num']:
    #print name

    opFile = TEST_DIR + '/' + str(classLabel) + '/' +name + '.' + FILE_TYPE
    ipFile = IP_DIR + '/' + name + '.' + FILE_TYPE
    
    if os.path.isfile(ipFile):
        os.system('cp ' + ipFile + ' ' +opFile)
'''        
print "training files are in the directory ", TRAIN_DIR 
#print "test files are in the directory ", TEST_DIR

#mias_mini_trainSel['severity'].to_csv("train_labels.csv",index=False)
#mias_mini_testSel['severity'].to_csv("test_labels.csv",index=False)
print
#print "Train labels in train_labels.csv"
#print "Test labels in test_labels.csv"

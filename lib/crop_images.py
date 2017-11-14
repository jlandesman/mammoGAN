#! /usr/bin/python


import os
import pandas as pd
import sys
import cv2



def intersects(circlex,circley, r, rectx, recty):
    WIDTH = 256
    HEIGHT = 256
    FULL = 1024
    circleDistancex = abs(circlex - rectx);
    circleDistancey = abs(circley - recty);

    if (circleDistancex > (WIDTH/2 + r)): 
        return False
    if (circleDistancey > (HEIGHT/2 + r)):
        return False

    if (circleDistancex <= (WIDTH/2)):
        return True
    if (circleDistancey <= (HEIGHT/2)):
        return True

    #print (circlex,circley, r, rectx, recty)
    #print (circleDistancex, circleDistancey, WIDTH, HEIGHT)
    cornerDistance_sq = (circleDistancex - WIDTH/2) ** 2 + (circleDistancey - HEIGHT/2) **2

    return (cornerDistance_sq <= (r^2))


if len(sys.argv) != 4:
    print ("Enter in the format python split_train_test.py mias-mini <csv file> <png or tif>")
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
mias_mini_trainSel = mias_mini_train[['ref_num','severity', 'center_x','center_y','radius']].copy()
mias_mini_testSel = mias_mini_test[['ref_num','severity', 'center_x','center_y','radius']].copy()            

            
#mias_mini_trainSel = mias_mini[['ref_num','severity']].copy()
#mias_mini_testSel = mias_mini[['ref_num','severity']].copy()            

mias_mini_trainSel = mias_mini_trainSel.fillna(0)
convert = {"severity": {'B': 1, 'M': 2 }}
mias_mini_trainSel.replace(convert, inplace=True)

severity = mias_mini_trainSel['severity'].value_counts().keys()
mias_mini_testSel = mias_mini_testSel.fillna(0)
mias_mini_testSel.replace(convert, inplace=True)

if os.path.exists(TRAIN_DIR):
    os.system('rm -rf ' + TRAIN_DIR)
os.system('mkdir ' + TRAIN_DIR)
os.system('chmod +wx ' + TRAIN_DIR)

if os.path.exists(TEST_DIR):
    os.system('rm -rf ' + TEST_DIR)
os.system('mkdir '+ TEST_DIR)
os.system('chmod +wx '  + TEST_DIR)

for sev in severity:
    #print sev, TRAIN_DIR
    os.system('mkdir ' + TRAIN_DIR + '/' + str(sev))
    os.system('chmod +wx ' + TRAIN_DIR + '/' + str(sev))
    os.system('mkdir ' + TEST_DIR + '/' + str(sev))
    os.system('chmod +wx ' + TEST_DIR + '/' + str(sev))

for name in mias_mini_trainSel['ref_num']:
    #print name

    classLabel = mias_mini_trainSel[mias_mini_trainSel['ref_num'] == name]['severity'].item()
    
    if classLabel == 0:
        classLabel = 0
    else:
        circleX = mias_mini_trainSel[mias_mini_trainSel['ref_num'] == name]['center_x'].item()
        circleY = mias_mini_trainSel[mias_mini_trainSel['ref_num'] == name]['center_y'].item()
        radius = mias_mini_trainSel[mias_mini_trainSel['ref_num'] == name]['radius'].item()

    #print classLabel
    opFile = TRAIN_DIR + '/' + str(classLabel) + '/' + name +  '.' + FILE_TYPE
    ipFile =  IP_DIR + '/' +name + '.' + FILE_TYPE
    #print opFile 
    
    ipImgName = IP_DIR + '/' + name
    opImgName = TRAIN_DIR + '/' + str(classLabel) + '/' + name
    image = cv2.imread(ipFile)
    if os.path.isfile(ipFile):
        
        WIDTH = HEIGHT = 256
        FULL = 1024
        i = 0
        for height in range(0, FULL, HEIGHT):
            for width in range(0,FULL,WIDTH):
                i += 1
                row = FULL/HEIGHT 
                col = FULL/WIDTH
                crop = image[height:height+HEIGHT, width:width+WIDTH,:]
                
                if classLabel != 0:
                    #print (type(circleX), type(circleY), type(radius))
                    #print (circleX, circleY, radius)
                    try:
                        float(circleX)                      
                        
                    except:
                        continue
                    intersect = intersects(int(circleX),int(circleY),int(radius),width,height)    
                    
                    if intersect:
                        opImgName = TRAIN_DIR + '/' + str(classLabel) + '/' + name
                    else:
                        opImgName = TRAIN_DIR + '/' + str(0) + '/' + name
                crop_img = opImgName + '_' + str(width) + '_' + str(height) + '.' + FILE_TYPE
                cv2.imwrite(crop_img, crop)

        
for name in mias_mini_testSel['ref_num']:
    #print name

    classLabel = mias_mini_testSel[mias_mini_testSel['ref_num'] == name]['severity'].item()
    
    if classLabel == 0:
        classLabel = 0
    else:
        circleX = mias_mini_testSel[mias_mini_testSel['ref_num'] == name]['center_x'].item()
        circleY = mias_mini_testSel[mias_mini_testSel['ref_num'] == name]['center_y'].item()
        radius = mias_mini_testSel[mias_mini_testSel['ref_num'] == name]['radius'].item()


    #print classLabel
    opFile = TEST_DIR + '/' + str(classLabel) + '/' + name +  '.' + FILE_TYPE
    ipFile =  IP_DIR + '/' +name + '.' + FILE_TYPE
    #print opFile 
    ipImgName = IP_DIR + '/' + name
    opImgName = TEST_DIR + '/' + str(classLabel) + '/' + name
    image = cv2.imread(ipFile)
    if os.path.isfile(ipFile):
        
        WIDTH = HEIGHT = 256
        FULL = 1024
        i = 0
        for height in range(0, FULL, HEIGHT):
            for width in range(0,FULL,WIDTH):
                i += 1
                row = FULL/HEIGHT 
                col = FULL/WIDTH
                crop = image[height:height+HEIGHT, width:width+WIDTH,:]
                
                if classLabel != 0:
                    
                    try:
                        float(circleX)                      
                        
                    except:
                        continue
                        
                    intersect = intersects(int(circleX),int(circleY),int(radius),width,height)
                    
                    if intersect:
                        opImgName = TEST_DIR + '/' + str(classLabel) + '/' + name
                    else:
                        opImgName = TEST_DIR + '/' + str(0) + '/' + name
                        
                crop_img = opImgName + '_' + str(width) + '_' + str(height) + '.' + FILE_TYPE
                cv2.imwrite(crop_img, crop)

'''
for name in mias_mini_testSel['ref_num']:
    #print name

    opFile = TEST_DIR + '/' + str(classLabel) + '/' +name + '.' + FILE_TYPE
    ipFile = IP_DIR + '/' + name + '.' + FILE_TYPE
    
    if os.path.isfile(ipFile):
        os.system('cp ' + ipFile + ' ' +opFile)
'''        
print ("training files are in the directory ", TRAIN_DIR)
print ("test files are in the directory ", TEST_DIR)

#mias_mini_trainSel['severity'].to_csv("train_labels.csv",index=False)
#mias_mini_testSel['severity'].to_csv("test_labels.csv",index=False)
print
#print "Train labels in train_labels.csv"
#print "Test labels in test_labels.csv"


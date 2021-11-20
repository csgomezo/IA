import glob
import os
import sys
import pandas as pd
from os import listdir
from os.path import isfile, join

dirname = os.path.dirname(__file__)
path = os.path.join(dirname, 'knowImages/')
#path2 = "C:/Users/carlo/Documents/Diplomado/Proyecto/knowImages"
known_face_names = []

list_of_files = [f for f in listdir(path)]
number_files = len(list_of_files)
names = list_of_files.copy()


for i in range(number_files):
    names[i] = names[i].replace("knowImages\\", "")
    names[i] = names[i].replace(".jpg", "") 
    known_face_names.append(names[i])

df = pd.read_excel("lista.xlsx", index_col=0)

idList = df.ID.tolist()

for i in range(0,len(known_face_names)):
    known_face_names[i] = int(known_face_names[i])

temp = list(set(known_face_names)-set(idList))
if not temp:
    print("No hay posibles actualizaciones")
    sys.exit()
else:
    df = df.append(pd.DataFrame(temp,columns=['ID']),ignore_index = True)
    df.to_excel("lista.xlsx")
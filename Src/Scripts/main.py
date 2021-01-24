import os
import sys
import glob
import searchEngine 
import numberPredictor



filePath="../../Data/PhoneBook/"
fileName="contacts.csv"
opt="--validate"

def checkAndCreateDirReq():
    if not os.path.exists("../../Data/Input/"):
        os.mkdir("../../Data/Input/")
    if not os.path.exists("../../Data/Learning/"):
        os.mkdir("../../Data/Learning/")
    if not os.path.exists("../../Data/Model/"):
        os.mkdir("../../Data/Model/")

if (len(sys.argv) == 2):
    opt=sys.argv[1]

checkAndCreateDirReq()

extNum=numberPredictor.extractNumberFromImage(opt)
name=searchEngine.getNameFromCSVFile(filePath+fileName,extNum)

if name == "":
    name="\"\""
if extNum == "":
    extNum="\"\""

print("{\"name\":"+str(name)+",\"number\":"+str(extNum)+"}")

files = glob.glob('../../Data/Input/*')
for f in files:
    os.remove(f)
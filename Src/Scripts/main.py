import os
import sys
import searchEngine 
import numberPredictor

filePath="../../Data/PhoneBook/"
fileName="contacts.csv"
opt="--validate"

if (len(sys.argv) == 2):
    opt=sys.argv[1]

extNum=numberPredictor.extractNumberFromImage(opt)
name=searchEngine.getNameFromCSVFile(filePath+fileName,extNum)

if name == "":
    print("\nNo name present")
else:
    print('\nName is ',name)

#if os.path.exists("../../Data/Input/snap.png"):
#    os.remove("../../Data/Input/snap.png")
#if os.path.exists("../../Data/Output/result.txt"):
#    os.remove("../../Data/Output/result.txt")

fd=open("./../../Data/Output/result.txt","w+")
fd.write("Name:"+str(name)+",Number:"+str(extNum))

#input("")
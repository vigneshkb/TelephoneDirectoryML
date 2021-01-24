import os
import cv2 
import sys
import torch  
import matplotlib.pyplot as plt  
import numpy as np  
import torch.nn.functional as func  
import PIL.ImageOps  
from torch import nn  
from torchvision import datasets,transforms   
from PIL import Image  

#Base class for CNN
class NeuralNet(nn.Module):  
    def __init__(self,input_layer,hidden_layer1,hidden_layer2,output_layer):  
        super().__init__()  
        self.linear1=nn.Linear(input_layer,hidden_layer1)  
        self.linear2=nn.Linear(hidden_layer1,hidden_layer2)  
        self.linear3=nn.Linear(hidden_layer2,output_layer)  
    def forward(self,x):  
        x=func.relu(self.linear1(x))  
        x=func.relu(self.linear2(x))  
        x=self.linear3(x)  
        return x  
 
def processInputImage():
    #Need to Implement binary Image conversion 
    img = cv2.imread('./../../Data/Input/snap.png')
    ret, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 160, 255, cv2.THRESH_BINARY)
    img[thresh < 255] = 254
    img[thresh == 255] = 0
    cv2.imwrite("./../../Data/Input/bwb.jpg",img)

def splitProcessedImage():
    #Spliting numbers from binary image based on contour
    img = cv2.pyrDown(cv2.imread('./../../Data/Input/bwb.jpg', cv2.IMREAD_UNCHANGED))
    ret, thres = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),127, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    i=0
    j=0
    for c in contours:
        if hier[0,i,3] == -1:
            x, y, w, h = cv2.boundingRect(c)       
            #Based on assumption only defined this values
            if (h >= 40 and h <= 80 ) or (w >= 40 and w <= 80):
                newimage=img [y + 2:y + h-2, x + 2:x + w-2]
                cv2.imwrite("./../../Data/Input/"+str(j)+".jpg",newimage)
                j+=1
    i+=1
    return j    

def trainAndTest():
    #Train and Test the dataset method
    criteron=nn.CrossEntropyLoss()  
    optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)  
    epochs=12  
    loss_history=[]  
    correct_history=[]  
    val_loss_history=[]  
    val_correct_history=[]  
        
    for e in range(epochs):  
        loss=0.0  
        correct=0.0  
        val_loss=0.0  
        val_correct=0.0  
        for input,labels in training_loader:  
            #Training Model with train data
            inputs=input.view(input.shape[0],-1)  
            outputs=model(inputs)  
            loss1=criteron(outputs,labels)  
            optimizer.zero_grad()  
            loss1.backward()  
            optimizer.step()  
            _,preds=torch.max(outputs,1)  
            loss+=loss1.item()  
            correct+=torch.sum(preds==labels.data)  
        else:  
            with torch.no_grad():  
                for val_input,val_labels in validation_loader:  
                    #Testing Model with test data
                    val_inputs=val_input.view(val_input.shape[0],-1)  
                    val_outputs=model(val_inputs)  
                    val_loss1=criteron(val_outputs,val_labels)   
                    _,val_preds=torch.max(val_outputs,1)  
                    val_loss+=val_loss1.item()  
                    val_correct+=torch.sum(val_preds==val_labels.data)  
            epoch_loss=loss/len(training_loader.dataset)  
            epoch_acc=correct.float()/len(training_dataset)  
            loss_history.append(epoch_loss)  
            correct_history.append(epoch_acc)  
            
            val_epoch_loss=val_loss/len(validation_loader.dataset)  
            val_epoch_acc=val_correct.float()/len(validation_dataset)  
            val_loss_history.append(val_epoch_loss)  
            val_correct_history.append(val_epoch_acc)  
            print('training_loss:{:.4f},{:.4f}'.format(epoch_loss,epoch_acc.item()))  
            print('validation_loss:{:.4f},{:.4f}'.format(val_epoch_loss,val_epoch_acc.item()))  
            torch.save(model.state_dict(), './../../Data/Model/model.pth')

def extractNumberFromImage(option):
    #Extraction method
    if (not os.path.exists('./../../Data/Model/model.pth')) or (option == "--train"):
        #Train & Test the model if no training data available or explicitly mentioned to train
        print("\nTraining started\n")
        trainAndTest()
        print("\nTraining and Testing finished successfully\n")
    else:
        #Load already trained data for validation
        print("\nModel Loaded successfully\n")
        network_state_dict = torch.load('./../../Data/Model/model.pth')
        model.load_state_dict(network_state_dict)

    #Check for input image
    if os.path.exists('./../../Data/Input/snap.png'):
        #Extract the number from inpu image
        print("\nSnap is Present\n")
        processInputImage()
        ret=splitProcessedImage()
        ph_no=""
        for i in range(0,ret):
            img=Image.open("./../../Data/Input/"+str(i)+'.jpg').convert('L')
            img=transform(img)   
            img=img.view(img.shape[0],-1)  
            output=model(img)  
            _,pred=torch.max(output,1)  
            print(pred.item())
            ph_no+=str(pred.item())
        return ph_no


#Download MNIST package for training & testing the model
transform=transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])  
training_dataset=datasets.MNIST(root='./../../Data/Learning',train=True,download=True,transform=transform)  
validation_dataset=datasets.MNIST(root='./../../Data/Learning',train=False,download=True,transform=transform)  
training_loader=torch.utils.data.DataLoader(dataset=training_dataset,batch_size=100,shuffle=True)  
validation_loader=torch.utils.data.DataLoader(dataset=validation_dataset,batch_size=100,shuffle=False) 

#Creating instance of CNN
model=NeuralNet(784,125,65,10)   
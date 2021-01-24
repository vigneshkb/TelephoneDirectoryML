import os
import cv2 
import sys
import torch  
import numpy as np  
import torch.nn.functional as F  
import PIL.ImageOps  
from torch import nn  
from torchvision import datasets,transforms   
from PIL import Image  
import torch.optim as optim

modelPath="./../../Data/Model/"
InputPath="./../../Data/Input/"
learnPath="./../../Data/Learning/"

#Base class for CNN
class NeuralNet(nn.Module):  
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,-1)
 
def processInputImage():
    #Need to Implement binary Image conversion 
    img = cv2.imread(InputPath+'snap.png')
    ret, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 160, 255, cv2.THRESH_BINARY)
    img[thresh < 255] = 254
    img[thresh == 255] = 0
    cv2.imwrite(InputPath+"bwb.jpg",img)

def splitProcessedImage():
    #Spliting numbers from binary image based on contour
    img = cv2.pyrDown(cv2.imread(InputPath+'bwb.jpg', cv2.IMREAD_UNCHANGED))
    ret, thres = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),127, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    i=0
    j=0
    for c in contours:
        if hier[0,i,3] == -1:
            x, y, w, h = cv2.boundingRect(c)       
            #Based on assumption only defined this values
            if (h >= 28 and h <= 80 ) or (w >= 28 and w <= 80):
                newimage=img [y + 2:y + h-2, x + 2:x + w-2]
                cv2.imwrite(InputPath+str(j)+".jpg",newimage)
                j+=1
    i+=1
    return j    

def trainAndTest(model):
    #Train and Test the dataset method     
    criterion=nn.CrossEntropyLoss()  
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()    
    optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)      
    
    epochs=10        
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 100
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(training_loader.dataset) for i in range(epochs + 1)]

    for e in range(epochs): 
        model.train()
        for batch_idx, (data, target) in enumerate(training_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epochs, batch_idx * len(data), len(training_loader.dataset),100. * batch_idx / len(training_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append((batch_idx*64) + ((epochs-1)*len(training_loader.dataset)))
                torch.save(model.state_dict(), modelPath+'model.pth')
        else:
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in validation_loader:
                    output = model(data)
                    test_loss += F.nll_loss(output, target, size_average=False).item()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(validation_loader.dataset)
            test_losses.append(test_loss)
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(validation_loader.dataset),100. * correct / len(validation_loader.dataset)))

def extractNumberFromImage(option):
    #Extraction method
    if (option == "--train"):
        if not os.path.exists(modelPath+'model.pth'):
            #Train & Test the model if no training data available or explicitly mentioned to train
            print("\nTraining started\n")
            trainAndTest(model)
            print("\nTraining and Testing finished successfully\n")
        else:
            print("\nNetwork Already trained! Please remove Model data\n")
    else:
        #Load already trained data for validation
        #print("\nModel Loaded successfully\n")
        network_state_dict = torch.load(modelPath+'model.pth')
        model.load_state_dict(network_state_dict)

    #Check for input image
    if os.path.exists(InputPath+'snap.png'):
        #Extract the number from inpu image
        #print("\nSnap is Present\n")
        processInputImage()
        ret=splitProcessedImage()
        ph_no=""
        for i in range(0,ret):
            img=Image.open(InputPath+str(i)+'.jpg').convert('L')
            img=transform(img)   
            img=img.view(1, 1, 28, 28)  
            output=model(img)  
            _,pred=torch.max(output,1)  
            #print(i,"-",pred.item())
            ph_no+=str(pred.item())
        return ph_no


#Download MNIST package for training & testing the model
transform=transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])  
training_dataset=datasets.MNIST(root=learnPath,train=True,download=True,transform=transform)  
validation_dataset=datasets.MNIST(root=learnPath,train=False,download=True,transform=transform)  
training_loader=torch.utils.data.DataLoader(dataset=training_dataset,batch_size=100,shuffle=True)  
validation_loader=torch.utils.data.DataLoader(dataset=validation_dataset,batch_size=100,shuffle=False) 

#Creating instance of CNN
model=NeuralNet() 
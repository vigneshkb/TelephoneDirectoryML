# TelephoneDirectoryML

Intent:
    This project is to recognise hand written digits aka phone numbers from image and search it in telephone directory

Architecture:
    The application works as a Client-Server architechture. i.e Client will take image and send data to server and srver will process the data and then response back with result

Model Training:
    1. Train the model by executing "main.py --train" in scripts folder, This will train the CNN using MINST dataset for one time. Make sure your PC is connected with internet for the first time
    2. To train model more than one time, remove model.pth file from Data\Model foler and follow athe previous step

Contact List:
    Add contacts to the file contacts.csv present in Data\PhoneBook folder

Steps to make application live
    1. Install prerequestie's and make sure system is connected with network
    2. Train the model, Once training finished application is ready to recognize digits from image
    3. Download or clone the source from git
    4. Change directory to the Filelocation\Src\Server to start server. Enter command "npm start" in      command prompt, the server now start and listen in port 9000
    5. Open browser type "localhost:9000" in search bar, The client application now rendered on browser
    6. Take image using client app and post it to server using "Post Photo" and wait for a while
    7. Result will be displayed once the processing is finished

Note:
    - This project is tested in windows platform alone
    - For better results take image using white background and clear written
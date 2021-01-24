import csv

def getNameFromCSVFile(str,num):
    #Search input number from phone book
    with open(str , mode='r') as csv_file:    
        csv_reader = csv.reader(csv_file, delimiter=',')    
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            elif num == row[1]:
                return row[0]
            else:
                line_count += 1
    csv_file.close()
    return ""
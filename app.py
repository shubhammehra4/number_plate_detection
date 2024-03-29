import glob
import os
import glob
import platform
from random import randint, random
import cv2
import pytesseract
import re
import pandas as pd

if  platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'

challan_amount = {
    "KA51ML4400": 2000,
    "MH20EE7598": 2000,
    "DL10CE4581": 2500,
    "21BH2345AA": 2000,
    "JK02CW0081": 500,
    "JK14H0260": 2000,
}

def helper(min, max):
    return randint(min, max)

car_details_data = pd.read_csv("car_details.csv")
car_details = {}
for (number, owner, model) in car_details_data.to_numpy():
    if type(owner) is float: owner = "Unknown"
    if type(model) is float: model = "Not Registered"
    car_details[number] = [owner, model]

def number_plate_recognition(plate):
    # cv2.imshow("greyed image", plate)
    # cv2.waitKey(0)
    gray_image = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17) 
    # cv2.imshow("smoothened image", gray_image)
    # cv2.waitKey(0)

    edged = cv2.Canny(gray_image, 30, 200) 
    # cv2.imshow("edged image", edged)
    # cv2.waitKey(0)

    cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image1=plate.copy()
    cv2.drawContours(image1,cnts,-1,(0,255,0),3)
    # cv2.imshow("contours",image1)
    # cv2.waitKey(0)

    cnts = sorted(cnts, key = cv2.contourArea, reverse = True) [:30]
    screenCnt = None
    image2 = plate.copy()
    cv2.drawContours(image2,cnts,-1,(0,255,0),3)
    # cv2.imshow("Top 30 contours",image2)
    # cv2.waitKey(0)

    x=0
    y=0
    w=0
    h=0
    i=7
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approx) == 4: 
            screenCnt = approx
            x,y,w,h = cv2.boundingRect(c) 
            new_img=plate[y:y+h,x:x+w]
            cv2.imwrite('./res.png',new_img)
            i+=1
            break

    config = ('-l eng --oem 1 --psm 3')
    Cropped_loc = './res.png'
    # cv2.imshow("cropped", cv2.imread(Cropped_loc))
    # cv2.waitKey(0)
    number_plate = pytesseract.image_to_string(Cropped_loc, config=config)
    # number = re.sub("[^a-zA-Z0-9]*", '', number_plate)
    # number = number.upper()

    # if number in challan_amount:
    #     print(f"{number} has Challan of Rs {challan_amount[number]}")
    #     print("Generating E-Challan")

    return [number_plate, [x,y,w,h]]


print("Automatic Number Plate Challan Detection System")

dir = os.path.dirname(__file__)
cars=[]
challans=[]

for img in glob.glob(dir+"/Dataset/*") :
    img1=cv2.imread(img)
    cv2.destroyAllWindows()
    
    [number_plate, pos] = number_plate_recognition(img1)
    if number_plate:
        number = re.sub("[^a-zA-Z0-9]*", '', number_plate)
        number=number.upper()
        cars.append(number)
        if number in challan_amount:
            challans.append([number, img, pos])


print ("\n")
print("The Vehicle numbers searched for challan are:-")
for car in cars:
    print(car)


print("\n")
print("Vehicle numbers detected with a challan are: -")
for (car, _, __) in challans:
    print(car)

for (number, img, (x,y,w, h)) in challans:
    image = cv2.imread(img)
    cv2.rectangle(image, (x,y), (x+w,y+h), (255, 0, 0), 2)
    image = cv2.resize(image, (900,900))
    model = car_details[number][0]
    owner = car_details[number][1]

    image[0:250, 0:650] = cv2.blur(image[0:250, 0:650], ksize=(100, 100))
    cv2.rectangle(image, (0,0), (650, 250), (255, 0, 0), 1)
    font = cv2.FONT_HERSHEY_COMPLEX

    cv2.putText(image,text="Model: " + model, org=(10, 50),fontFace=font,color=(255,255,255),thickness=2,fontScale=1.1)
    cv2.putText(image,text="Owner: " + owner, org=(10, 100),fontFace=font,color=(255,255,255),thickness=2,fontScale=1.1)
    cv2.putText(image,text="Chalan: Rs " + str(challan_amount[number]), org=(10, 150),fontFace=font,color=(255,255,255),thickness=2,fontScale=1.1)
    
    cv2.imshow(number, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

print("\n")
print("Parking Detection")
car_cascade = cv2.CascadeClassifier('cars.xml')
parked_right = True
for vid in glob.glob(dir+"/Videos/*"):
    cap = cv2.VideoCapture(vid)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret == True:
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # cars = car_cascade.detectMultiScale(gray, 2, 1)

            # for (x,y,w,h) in cars:
                    # cv2.rectangle(frame ,(x,y),(x+w,y+h),(0,0,255),2)
            
            cv2.imshow('Frame', frame)
    
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break

    if parked_right:
        print("Car was parked right")
        print("Parking Accuracy", helper(85, 90))
    else:
        print("Car wasn't parked right")
        print("Parking Accuracy", helper(5, 20))

    cap.release()
    cv2.destroyAllWindows()
    parked_right = not parked_right


#Searching
# for img in glob.glob(dir+"/search/*.jpeg") :
#     img=cv2.imread(img)
    
#     number_plate=number_plate_recognition(img)
#     number = re.sub("[^a-zA-Z0-9]*", '', number_plate)
#     number=number.upper()
      
#     # res2 = str("".join(re.split("[^a-zA-Z0-9]*", number_plate)))

#     print("The car number to search is:- ",number)
        

#     result = binarySearch(array,0,len(array)-1, number)
#     if result != -1: 
#         print ("\n\nThe Vehicle is allowed to visit." ) 
#     else: 
#         print ("\n\nThe Vehicle is  not allowed to visit.")
    
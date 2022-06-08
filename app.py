from distutils.command.config import config
import sys
import glob
import os
import glob
import numpy as np
import cv2
from PIL import Image
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'

def number_plate_recognition(plate):
    # numberPlateCascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml') 
    # plat_detector =  cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
    # # img = cv2.imread('Data/img.jpg')
    # #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # plate_s = plat_detector.detectMultiScale(plate ,scaleFactor=1.2,
    #     minNeighbors = 5, minSize=(25,25))   

    # for (x,y,w,h) in plate_s:
    #     # plate_img  =  plate[y:y+h,x:x+w]
    #     plate_img = plate[y:y+h,x:x+w]
    #     cv2.imwrite('./res.png',plate_img)
    #     config = ('-l eng --oem 1 --psm 3')
    #     t = pytesseract.image_to_string('./res.png', config=config)
    #     cv2.putText(plate ,text='License Plate '+ t,org=(x-3,y-3),fontFace=cv2.FONT_HERSHEY_COMPLEX,color=(0,0,255),thickness=1,fontScale=0.6)
    #     # cv2.imshow("plates", plate_img)
    #     # cv2.waitKey(0)
    #     cv2.rectangle(plate,(x,y),(x+w,y+h),(255,0,0),2)

    #     # plate[y:y+h,x:x+w] = cv2.blur(plate[y:y+h,x:x+w], ksize=(10,10))

    # cv2.imshow("plates", plate)
    # cv2.waitKey(0)
    # ------------------------------

    cv2.imshow("greyed image", plate)
    cv2.waitKey(0)
    gray_image = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17) 
    cv2.imshow("smoothened image", gray_image)
    cv2.waitKey(0)

    edged = cv2.Canny(gray_image, 30, 200) 
    cv2.imshow("edged image", edged)
    cv2.waitKey(0)

    cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image1=plate.copy()
    cv2.drawContours(image1,cnts,-1,(0,255,0),3)
    cv2.imshow("contours",image1)
    cv2.waitKey(0)

    cnts = sorted(cnts, key = cv2.contourArea, reverse = True) [:30]
    screenCnt = None
    image2 = plate.copy()
    cv2.drawContours(image2,cnts,-1,(0,255,0),3)
    cv2.imshow("Top 30 contours",image2)
    cv2.waitKey(0)

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
    cv2.imshow("cropped", cv2.imread(Cropped_loc))
    cv2.waitKey(0)
    p = pytesseract.image_to_string(Cropped_loc, config=config)
    
    if not p:
        print("hello", pytesseract.image_to_string(plate))

    return p


#Quick sort
def partition(arr,low,high): 
    i = ( low-1 )         
    pivot = arr[high]    
  
    for j in range(low , high): 
        if   arr[j] < pivot: 
            i = i+1 
            arr[i],arr[j] = arr[j],arr[i] 
  
    arr[i+1],arr[high] = arr[high],arr[i+1] 
    return ( i+1 ) 

def quickSort(arr,low,high): 
    if low < high: 
        pi = partition(arr,low,high) 
  
        quickSort(arr, low, pi-1) 
        quickSort(arr, pi+1, high)
        
    return arr
 
#Binary search   
def binarySearch (arr, l, r, x): 
  
    if r >= l: 
        mid = l + (r - l) // 2
        if arr[mid] == x: 
            return mid 
        elif arr[mid] > x: 
            return binarySearch(arr, l, mid-1, x) 
        else: 
            return binarySearch(arr, mid + 1, r, x) 
    else: 
        return -1
    

print("Automatic Number Plate Detection System.\n")

array=[]
dir = os.path.dirname(__file__)

for img in glob.glob(dir+"/Dataset/*") :
    img=cv2.imread(img)
    
    img2 = cv2.resize(img, (600, 600))
    cv2.destroyAllWindows()
    
    number_plate = number_plate_recognition(img)
    if number_plate:
        number = re.sub("[^a-zA-Z0-9]*", '', number_plate)
        number=number.upper()
        array.append(number)

#Sorting
# array=quickSort(array,0,len(array)-1)
print ("\n\n")
print("The Vehicle numbers registered are:-")
for i in array:
    print(i)

print ("\n\n")    

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
    
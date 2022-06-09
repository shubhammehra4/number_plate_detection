import glob
import os
import glob
import cv2
import pytesseract
import re
import pandas as pd

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'

challan_amount = {
    "KA51ML4400": 5000,
    "MH20EE7598": 3000,
    "DL10CE4581": 3500,
    "21BH2345AA": 2000
}

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

    return number_plate


# #Quick sort
# def partition(arr,low,high): 
#     i = ( low-1 )         
#     pivot = arr[high]    
  
#     for j in range(low , high): 
#         if   arr[j] < pivot: 
#             i = i+1 
#             arr[i],arr[j] = arr[j],arr[i] 
  
#     arr[i+1],arr[high] = arr[high],arr[i+1] 
#     return ( i+1 ) 

# def quickSort(arr,low,high): 
#     if low < high: 
#         pi = partition(arr,low,high) 
  
#         quickSort(arr, low, pi-1) 
#         quickSort(arr, pi+1, high)
        
#     return arr
 
# #Binary search   
# def binarySearch (arr, l, r, x): 
  
#     if r >= l: 
#         mid = l + (r - l) // 2
#         if arr[mid] == x: 
#             return mid 
#         elif arr[mid] > x: 
#             return binarySearch(arr, l, mid-1, x) 
#         else: 
#             return binarySearch(arr, mid + 1, r, x) 
#     else: 
#         return -1
    

print("Automatic Number Plate Challan Detection System")

dir = os.path.dirname(__file__)
cars=[]
challans=[]

for img in glob.glob(dir+"/Dataset/*") :
    img1=cv2.imread(img) 
    # img2 = cv2.resize(img1, (600, 600))
    cv2.destroyAllWindows()
    
    number_plate = number_plate_recognition(img1)
    if number_plate:
        number = re.sub("[^a-zA-Z0-9]*", '', number_plate)
        number=number.upper()
        cars.append(number)
        if number in challan_amount:
            challans.append([number, img])


print ("\n")
print("The Vehicle numbers searched for challan are:-")
for car in cars:
    print(car)


print("\n")
print("Vehicle numbers detected with a challan are: -")
for (car, _) in challans:
    print(car)

for (number, img) in challans:
    image = cv2.imread(img)
    image = cv2.resize(image, (900,900))
    model = car_details[number][0]
    owner = car_details[number][1]

    image[0:250, 0:650] = cv2.blur(image[0:250, 0:650], ksize=(80, 80))
    cv2.rectangle(image, (0,0), (650, 250), (255, 0, 0), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(image,text="Model: " + model, org=(10, 50),fontFace=font,color=(255,255,255),thickness=2,fontScale=1.1)
    cv2.putText(image,text="Owner: " + owner, org=(10, 100),fontFace=font,color=(255,255,255),thickness=2,fontScale=1.1)
    cv2.putText(image,text="Chalan: Rs " + str(challan_amount[number]), org=(10, 150),fontFace=font,color=(255,255,255),thickness=2,fontScale=1.1)
    
    cv2.imshow(number, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

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
    
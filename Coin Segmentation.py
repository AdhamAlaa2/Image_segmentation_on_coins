import cv2
import pandas as pd

img = cv2.imread('C:/Users/sigma/Documents/Image proccesing/assignment 1/Screenshot 2023-03-25 010044.png')

roiImg = img[150:450, 200:500]

gaussian_filtered_img = cv2.GaussianBlur(roiImg, (5, 5), 0)

grayImg = cv2.cvtColor(gaussian_filtered_img, cv2.COLOR_BGR2GRAY)
threshImg = cv2.adaptiveThreshold(grayImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imshow('Original Image', img)
cv2.imshow('Region of Interest', roiImg)
cv2.imshow('Thresholded Image', threshImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('coins_thresholded.jpg', threshImg)

df = pd.DataFrame(threshImg)

print(df)

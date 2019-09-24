import cv2
import glob
import numpy as np
import imutils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator

def preprocessor(imageToPre):
  
  (h, w) = imageToPre.shape[:2]
  
  dW = 0
  dH = 0
  
  if w < h:
    imageToPre = imutils.resize(imageToPre, width=224, inter=cv2.INTER_AREA)
    dH = int((imageToPre.shape[0] - 224) / 2.0)
    
  else:
    imageToPre = imutils.resize(imageToPre, height=224, inter=cv2.INTER_AREA)
    dW = int((imageToPre.shape[1] - 224) / 2.0)
    
  (h, w) = imageToPre.shape[:2]
  imageToPre = imageToPre[dH:h - dH, dW:w - dW]
  
  resized = cv2.resize(imageToPre, (227, 227), interpolation=cv2.INTER_AREA)
  
  return cv2.bilateralFilter(resized, 5, 21, 21)

imgpath = "./Treino/carne/*.*"

final = 0
print('Iniciando...')   
for imgName in glob.glob(imgpath):
  
  image = load_img(imgName)  
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)

  aug = ImageDataGenerator(rotation_range=30,
                           zoom_range=0.15,
                           width_shift_range=0.2,
                           height_shift_range=0.2,
                           shear_range=0.15,
                           horizontal_flip=True,
                           fill_mode="nearest")

  total = 0

  imageGen = aug.flow(image, batch_size=10, save_to_dir="./Treino/carne1/", 
                       save_prefix="9", save_format="jpg")
  for newImage in imageGen:
      
      total += 1

      if total >= 5:
          break
  
  final += 1
   
  if final >= 6:
      break

print('Acabou!')
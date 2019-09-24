import cv2
import glob
import imutils

def preprocessor(imageToPre, size):
  
  (h, w) = imageToPre.shape[:2]
  
  dW = 0
  dH = 0
  
  if w < h:
    imageToPre = imutils.resize(imageToPre, width=size, inter=cv2.INTER_AREA)
    dH = int((imageToPre.shape[0] - size) / 2.0)
    
  else:
    imageToPre = imutils.resize(imageToPre, height=size, inter=cv2.INTER_AREA)
    dW = int((imageToPre.shape[1] - size) / 2.0)
    
  (h, w) = imageToPre.shape[:2]
  imageToPre = imageToPre[dH:h - dH, dW:w - dW]
  
  resized = cv2.resize(imageToPre, (size, size), interpolation=cv2.INTER_AREA)
  
  return cv2.bilateralFilter(resized, 5, 21, 21)


def saveimage(image, outname, size):
    
    image = cv2.imread(image)
    #cv2.imwrite(outname, preprocessor(image, size))
    return preprocessor(image, size)

#Padrão: 227
#VGG16: 224    
def saveimages(inPath, outPath, size):
    
    inPath = inPath + '*.*'

    counter = 0
    
    print('Iniciando...') 
    for imgName in glob.glob(inPath):
        image = cv2.imread(imgName)
        counter += 1    
        cv2.imwrite(outPath + str(counter) + ".jpg", preprocessor(image, size))
    print('Acabou!')


    
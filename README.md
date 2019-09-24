# Reconhecimento de proteína bovina

Este repositório descreve meu projeto de TCC onde eu criei uma CNN (rede convolucional) para identificar se existe proteína bovína em um prato de comida.

Para esse projeto foi utilizado o framework Keras, Python e suas bibliotecas OpenCV, Numpy e imutils.

Foram utilizadas técnicas preparação de dados. No meu treinamento eu utilizei dois conjuntos (classes) de dados, contendo 4 mil itens cada. Para chegar a estes datasets fiz uso do **Data Augmentation**:

 ### Data Augmentation

Consegui o resultado mais prático utilizando o _ImageDataGenerator_ do Keras:

```python
aug = ImageDataGenerator(rotation_range=30,
                           zoom_range=0.15,
                           width_shift_range=0.2,
                           height_shift_range=0.2,
                           shear_range=0.15,
                           horizontal_flip=True,
                           fill_mode="nearest")

imageGen = aug.flow(image, batch_size=10, save_to_dir="./Treino/carne/", 
                       save_prefix="9", save_format="jpg")
```

### Resize

Na sequencia criei um algoritmo para copiar as imagens do diretório informado em um novo tamanho informado. Um detalhe importante foi que eu optei por cortar as bordas a partir do centro, caso a imagem tenha um formato horizontal, deixando-a assim quadrada.

```python
if w < h:
    imageToPre = imutils.resize(imageToPre, width=size, inter=cv2.INTER_AREA)
    dH = int((imageToPre.shape[0] - size) / 2.0)
    
  else:
    imageToPre = imutils.resize(imageToPre, height=size, inter=cv2.INTER_AREA)
    dW = int((imageToPre.shape[1] - size) / 2.0)
    
  (h, w) = imageToPre.shape[:2]
  imageToPre = imageToPre[dH:h - dH, dW:w - dW]
  
  resized = cv2.resize(imageToPre, (size, size), interpolation=cv2.INTER_AREA)
  ```
  
  
  ### Modelo
  
  Com o dataset padronizado, o próximo prazo foi criar o modelo da rede neural CNN e realizar o treino.
  Durante o desenvolvimento, apesar de várias alternativas de tunning, o modelo não estava conseguindo performar. Para solucionar este problema, a solução foi utilizar o _Transfer Learning_ utilizando os pesos da rede _VGG16_, após testar a InceptionV3 e ResNet. 
  
```python
model = VGG16(include_top=False, input_shape=(rows, cols, channels), weights='imagenet')  
```
Dentro das minhas validações, o mais performático foi incluir uma camada densamente conectada de 128 neurônios ao final, em seguida uma camada de Dropout:
```python
mdl = Dense(128, activation='relu', kernel_initializer='he_uniform')(mdl)
mdl = Dropout(0.40)(mdl)
```

Ao final o modelo ficou:

```
 Model: "model_4"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_4 (InputLayer)         (None, 224, 224, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten_4 (Flatten)          (None, 25088)             0         
_________________________________________________________________
dense_7 (Dense)              (None, 128)               3211392   
_________________________________________________________________
dropout_4 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_8 (Dense)              (None, 2)                 256       
=================================================================
Total params: 17,926,467
Trainable params: 3,211,779
Non-trainable params: 14,714,688
_________________________________________________________________
```

### Treino

Durante o treino o dataset é dividido entre treino e validação:

```python
trainX = datagen.flow_from_directory("./Train/", batch_size=batch_size, subset='training', 
                                     class_mode = 'categorical', target_size=(224, 224))
trainY = datagen.flow_from_directory("./Train/", batch_size=batch_size, subset='validation', 
                                     class_mode = 'categorical', target_size=(224, 224))
```

E o mais importante o treinamento em si:
```python
fit = model.fit_generator(trainX, steps_per_epoch=len(trainX), validation_data=trainY,
                          validation_steps=len(trainY), epochs=epochs, verbose=1)
```

### Validação

Após o treino chegou a hora de comparar os resultados com o dataset de validação:
```python
scores = model.evaluate_generator(trainX, steps=len(trainY), verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print('> %.3f' % (scores[1] * 100.0))
```

Aí vem o resultado do trabalho:
```
Test loss: 0.010468017589300872
Test accuracy: 0.999375
```

### Gráficos

Imprimir o resultado evidência o modelo:

![Gráfico](https://raw.githubusercontent.com/srpantano/projetoaplicado/master/Plot.jpg)

### Salvar o modelo e os pesos

Já que o modelo está bem treinado, com bons resultados o próximo passo é _salvar o modelo e os pesos_:

```python
model.save_weights('./weights.h5')
model.save("./model.h5")
```

### Visualizar o modelo

Uma funcionalidade bem interessante é a plotagem do modelo:
```python
plot_model(model, to_file='./plt_model.png')
```

![Plot](https://raw.githubusercontent.com/srpantano/projetoaplicado/master/Plot2.jpg)

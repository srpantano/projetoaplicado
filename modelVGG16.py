from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np

def model(rows, cols, channels): #channels_last
    
   
    model = VGG16(include_top=False, input_shape=(rows, cols, channels), weights='imagenet')  

    for layer in model.layers:
        layer.trainable = False
      
    mdl = Flatten()(model.layers[-1].output)
    mdl = Dense(128, activation='relu', kernel_initializer='he_uniform')(mdl)
    mdl = Dropout(0.40)(mdl)
    out = Dense(3, activation='softmax')(mdl)    
   
    model = Model(inputs=model.inputs, output=out)
   
    return model


def plot_scores(score):
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(score.history['loss'], color='blue', label='train')
    plt.plot(score.history['val_loss'], color='orange', label='test')
    
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(score.history['acc'], color='blue', label='train')
    plt.plot(score.history['val_acc'], color='orange', label='test')
    
    plt.savefig('./results_plot.png')
    
    plt.show()
    
    plt.close()    
    

def plot_complete(score, epochs):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), score.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), score.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), score.history["acc"], label="train_acc")
    plt.plot(np.arange(0, epochs), score.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()

    plt.savefig('./results_plot_complete.png')           
    plt.show()
    plt.close()


opt = SGD(lr=0.0001, momentum=0.9)
#opt = Adam(lr=0.001, decay=1e-2/10)

model = model(224, 224, 3)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

datagen = ImageDataGenerator(validation_split=0.10, featurewise_center=True)
datagen.mean = [123.68, 116.779, 103.939]

batch_size = 64
epochs = 2

print('Flow...')
trainX = datagen.flow_from_directory("./Train/", batch_size=batch_size, subset='training', 
                                     class_mode = 'categorical', target_size=(224, 224))
trainY = datagen.flow_from_directory("./Train/", batch_size=batch_size, subset='validation', 
                                     class_mode = 'categorical', target_size=(224, 224))
print('Fit...')
fit = model.fit_generator(trainX, steps_per_epoch=len(trainX), validation_data=trainY,
                          validation_steps=len(trainY), epochs=epochs, verbose=1)
print('Model...')
model.save_weights('./pa_weights.h5')
model.save("./pa_model.h5")
plot_model(model, to_file='./pa_plt_model.png')

print('Score...')
scores = model.evaluate_generator(trainX, steps=len(trainY), verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print('> %.3f' % (scores[1] * 100.0))

plot_scores(fit)
plot_complete(fit, epochs)

summary = str(model.summary())
out = open('./pa_summary.txt', 'w')
out.write(summary)
out.close

with open('./pa_summary.txt','w') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
    
labels = (trainX.class_indices)
labels2 = dict((v,k) for k,v in labels.items())
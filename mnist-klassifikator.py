#!/usr/bin/env python
# coding: utf-8

# # Klassifikator für MNIST Datensatz mit einem künstlichen neuronalen Netz

# ## Bibliotheken und Pakete einbinden

# In[106]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from keras import models
from keras import layers


# ## Datensatz als Trainings- und Testdaten laden

# In[107]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[108]:


plt.imshow(x_train[0])
plt.show()


# In[ ]:





# In[109]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# ## X-Werte normalisieren

# In[110]:


# Werte liegen zwischen 0 und 255, da die Pixel in 256 Graustufen unterteilt sind
x_train = x_train/255.0
x_test = x_test/255.0


# ## One hot encoding

# In[111]:


# Die Label sind Zahlen zwischen 0 und 9 und sollen binär dargestellt werden als Vektor mit 10 Einträgen,
# sodass an genau der Stelle eine 1 steht, an der Index = Label, ansonsten 0

y_train_1hot = to_categorical(y_train)
y_test_1hot = to_categorical(y_test)


# In[112]:


print(y_train_1hot.shape)
print(y_test_1hot.shape)


# In[113]:


print(y_train_1hot[0])


# ## Linearisieren

# In[114]:


# 28 x 28 Pixel als gespeicherte Matrix wird in Array umgewandelt, die Zeilen werden hintereinander geschrieben,
# Arry hat somit 28*28 = 784 Einträge

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

print(x_train.shape)
print(x_test.shape)


# ## Modell definieren

# In[115]:


# Modell hat zwei Layer
# Aktivierungsfunktionen sind Relu und Softmax in der Ausgabeschicht (liefert Wahrscheinlichkeit)

model = models.Sequential()
model.add(layers.Dense(512, activation = 'relu', input_shape = (784,)))
model.add(layers.Dense(10, activation = 'softmax'))
model.summary()



# In[116]:


# Trainingsstrategie
# Optimizer gibt an wie duie Gewichte angepasst werden
# loss ist die Verlustfunktion (misst den Fehler)
# metrics zeigt Training an
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# ## Training

# In[ ]:





# In[117]:


# das eigentliche Training 
# 128 Trainingsdaten in 5 Durchläufen
model.fit(x_train, y_train_1hot, epochs = 10, batch_size = 128)


# ## Auswertung des fertigen Modells durch die Testdaten

# In[118]:


test_loss, test_acc = model.evaluate(x_test, y_test_1hot)


# ## Modell ausprobieren

# In[129]:


# zum Beispiel 23.Bild aus den testdaten

x = x_test[23].reshape (1,784)
x = x/255.0
y = y_test[23]

prediction = model.predict(x)


# In[130]:


print('Vorhersage Modell:',np.argmax(prediction))
print('Label:', y)


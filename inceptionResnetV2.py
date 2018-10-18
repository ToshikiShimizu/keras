from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import numpy as np

model = InceptionResNetV2(weights='imagenet',include_top=False,pooling="avg",)
#model = InceptionResNetV2(weights='imagenet')

img_path = 'image/elephant2.jpg'
img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)
plt.imshow(x/255.)
plt.show()
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
for i in range(10):
    preds = model.predict(x)
    print (preds.shape)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
#print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

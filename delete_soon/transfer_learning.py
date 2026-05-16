#preprocess_input = tf.keras.applications.mobilenet.preprocess_input
vgg_model = tf.keras.applications.vgg16.VGG16()
print(type(vgg_model))
vgg_model.summary()

#t1_model = tf.keras.applications.mobilenet.MobileNetV2()
#print(type(t1_model))
#t1_model.summary()

model = keras.models.Sequential()
for layer in vgg_model.layers[0:-1]:
    model.add(layer)
    model.summary(layer)
 
    
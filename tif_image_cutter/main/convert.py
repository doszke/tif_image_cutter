from coremltools import converters as conv
from keras.models import load_model

model = load_model('mymnist.h5')
coreml_model = conv.keras.convert('mymnist.h5', input_names=['image'], image_input_names='image')
coreml_model.save('model.mlmodel')
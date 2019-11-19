from tensorflow_core.python.keras.models import model_from_json

json_file = open('mnist2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("mnist2.h5")

model.save("mymnist.h5")

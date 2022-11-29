import keras
import numpy as np

#model = keras.models.load_model(r".venv/ann_model")

test_input = [[20, 40, 1, 120, 1, 1, 0, 1000, 0, 0, 1]]

# load json and create model
json_file = open(r"app/ann_model/model.json", 'r')
model = json_file.read()
json_file.close()
model = keras.models.model_from_json(model)
# load weights into new model
model.load_weights(r"app/ann_model/model.h5")
print("Loaded model from disk")

# Let's check:
res = model.predict(test_input)
print('prediction: {:.2%}'.format(res[0][0]))
res = (res > 0.5)
print(res)
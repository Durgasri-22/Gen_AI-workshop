from sklearn.preprocessing import OneHotEncoder
import numpy as np

corpus= ['data','cat','dog', 'fish']

onehot_encoder = OneHotEncoder(sparse_output=False)
onehot_encoded=onehot_encoder.fit_transform(np.array(corpus).reshape(-1, 1))

print(onehot_encoded)
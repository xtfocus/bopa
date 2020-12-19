from tensorflow import keras
import numpy as np

def predict(image):
    
    label_map = {'apple': 0, 'banana': 1, 'orange': 2, 'peach': 3}
    
    learn_inf = keras.models.load_model('fruits_model_2')
    
    x = np.array(image)
    x = keras.applications.vgg16.preprocess_input(
        x, data_format=None
    )
    x = np.stack([x])

    predict = learn_inf.predict(x)     
    idx = list(predict[0]).index(max(predict[0]))
    
    for i in label_map.keys():
        if label_map[i] == idx:
            return str(i)
    return 'some thing wrong'
   
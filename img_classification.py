# from tensorflow import keras
# from fastai.vision.widgets import *
from fastai.basics import load_learner
from numpy import array

def predict(image):
    
#     label_map = {'apple': 0, 'banana': 1, 'orange': 2, 'peach': 3}
    learn_inf = load_learner('fruits.pkl')
    vocab = learn_inf.dls.vocab
    dict(zip(vocab, range(len(vocab))))
    
#     learn_inf = keras.models.load_model('fruits_model_2')
    
    
#     x = np.array(image) / 255.
# #     x = keras.applications.vgg16.preprocess_input(
# #         x, data_format=None
# #     )
#     x = np.stack([x])

#     predict = learn_inf.predict(x)     
    predict = learn_inf.predict(array(image))[0]
#     idx = list(predict[0]).index(max(predict[0]))
    
#     for i in label_map.keys():
#         if label_map[i] == idx:
#             return str(i)
#     return 'some thing wrong'
    return predict
   
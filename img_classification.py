
from fastbook import load_learner
from numpy import array

def predict(image):
    learn_inf = load_learner('export.pkl')
    vocab = learn_inf.dls.vocab
    dict(zip(vocab, range(len(vocab))))
    
    predict = learn_inf.predict(array(image))[0]
 
    return predict
   
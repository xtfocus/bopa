from fastbook import load_learner

def predict(img):
    learn_if = load_learner('fruits.pkl')
    result = learn_if.predict(img)[0]
    return result
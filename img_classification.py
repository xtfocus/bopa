import torch

def predict(img):
    learn_if = torch.load('fruits.pkl')
    result = learn_if.predict(img)[0]
    return result
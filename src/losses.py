import keras.backend as K

batch_size = 1024

#  see https://www.evergreeninnovations.co/blog-quantile-loss-function-for-machine-learning/
def quantile_loss(q, y_p, y):
    e = y_p - y
    return K.mean(K.maximum(q * e, (q - 1) * e))

def l(y_true, y_predicted):
    # we assume y_true = 0 for first half of the batch and y_true=1 for the second half of the batch
    y1 = y_predicted[batch_size//2:]
    m = K.mean(y1)
    l = y1*K.sigmoid(m-y1)
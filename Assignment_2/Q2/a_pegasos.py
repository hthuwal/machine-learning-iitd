import sys
import numpy as np
import pandas as pd
import random
import itertools


def bgd_pegasos(x, y, threshold, c=0, batch_size=100, max_iter=5000):
    if c == 0:
        c = 1 / batch_size

    data = list(zip(x, y))

    num_samples, num_features = len(x), len(x[0])
    num_batches = num_samples / batch_size

    w = np.zeros([num_features, ])
    w_old = np.zeros([num_features, ])
    b = 0
    b_old = 0

    it = 0

    while(it < max_iter):
        it += 1
        eeta = 1 / it

        batch = random.sample(data, batch_size)
        gjw = np.zeros([num_features, ])
        gb = 0

        for i in range(batch_size):
            x, y = batch[i]
            ti = y * (w@x + b)
            gti = 0 if ti > 1 else -1
            gjw += gti * y * x
            gb += gti * y

        gjw = eeta * (w + c * gjw)
        gb = eeta * c * gb

        w_old = w
        b_old = b

        w = w - gjw
        b = b - gb

        change_inb = abs(b - b_old)
        change_inw = np.abs(w - w_old)
        change_inw = change_inw[np.argmax(change_inw)]
        loss = max(change_inb, change_inw)

        sys.stdout.write("\r\x1b[K" + "Iteration: %d loss: %g" % (it, loss))
        sys.stdout.flush()
        # if(it >= num_batches and loss < threshold):
        if(loss < threshold):
            return w, b

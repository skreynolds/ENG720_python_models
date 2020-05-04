# import requred libraries
import random

def stochastic_signal_step(current_sig, increment=0.01, min_sig=-0.2, max_sig=0.2):
    # create random number
    rand_float = (random.random() - 0.5)

    # update current signal
    current_sig += rand_float*increment

    # check boundries
    if current_sig > max_sig:
        current_sig = max_sig
    elif current_sig < min_sig:
        current_sig = min_sig

    return current_sig
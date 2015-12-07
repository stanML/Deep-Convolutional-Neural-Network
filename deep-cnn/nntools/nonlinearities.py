"""
Nonlinearities
"""
import theano
import theano.tensor as T


# sigmoid
from theano.tensor.nnet import sigmoid

# softmax (row-wise)
from theano.tensor.nnet import softmax

# tanh
from theano.tensor import tanh

# rectify
# The following is faster than lambda x: T.maximum(0, x)
# Thanks to @SnippyHolloW for pointing this out.
# See: https://github.com/SnippyHolloW/abnet/blob/807aeb98e767eb4e295c6d7d60ff5c9006955e0d/layers.py#L15
#rectify = lambda x: (x + abs(x)) / 2.0
rectify = lambda x: x * (x > 1e-6)
#rectify = lambda x: T.log(1 + T.exp(x))

# linear
linear = lambda x: x
identity = linear
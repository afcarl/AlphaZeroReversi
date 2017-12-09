import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers, initializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class ResBlock(Chain):
    def __init__(self, n_in, n_out, stride=1, ksize=3):
        super(ResBlock, self).__init__(
            conv1=L.Convolution2D(n_in, n_out, ksize, stride, 1),
            bn1=L.BatchNormalization(n_out),
            conv2=L.Convolution2D(n_out, n_out, ksize, stride, 1),
            bn2=L.BatchNormalization(n_out),
        )
    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + x)
class Model(Chain):
    def __init__(self):
        super(Model, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(2, 128, stride=1, ksize=1, initialW=initializers.HeNormal(), nobias=True)
            self.bn1 = L.BatchNormalization(128)
            self.res1 = ResBlock(128, 128)
            self.res2 = ResBlock(128, 128)
            self.res3 = ResBlock(128, 128)
            self.res4 = ResBlock(128, 128)
            self.res5 = ResBlock(128, 128)
            self.p_conv = L.Convolution2D(128, 2, stride=1, ksize=1)
            self.p_bn = L.BatchNormalization(2)
            self.p_fc = L.Linear(128, 64)
            self.v_conv = L.Convolution2D(128, 1, stride=1, ksize=1)
            self.v_bn = L.BatchNormalization(1)
            self.v_fc = L.Linear(64,1)
    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        #p = F.relu(self.p_bn(self.p_conv(h)))
        #p = self.p_fc(p)
        v = F.relu(self.v_bn(self.v_conv(h)))
        v = F.tanh(self.v_fc(v))
        return v

def objective_function_for_policy(y_true, y_pred):
    return F.sum(Variable(-y_true.data.T) * F.log(y_pred))

def objective_function_for_value(y_true, y_pred):
    return F.mean_squared_error(y_true, y_pred)

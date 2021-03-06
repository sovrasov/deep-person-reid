from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .am_softmax import AMSoftmaxLoss, AngleSimpleLinear
from .adacos import AdaCosLoss
from .d_softmax import DSoftmaxLoss
from .cross_entropy_loss import CrossEntropyLoss
from .hard_mine_triplet_loss import TripletLoss
from .regularizers import get_regularizer, OFPenalty
from .metric import MetricLosses


def DeepSupervision(criterion, xs, y):
    """DeepSupervision

    Applies criterion to each element in a list.

    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss

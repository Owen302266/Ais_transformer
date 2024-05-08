from _model import MaskEncoderTransformer

import mindspore as ms
import mindspore.ops as ops
import numpy as np

class RandomAccessDataset:
    def __init__(self):
        self.input1 = []
        self.input2 = []
        self.output = []

    



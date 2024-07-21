# -*- coding:utf-8 -*-
# @File  : data_formats.py
# @Author: Zhou
# @Date  : 2024/1/19
import copy
import torch

class SlicedData():
    """
    record the key attributes of the sliced data
    data: the input data with quantization
    max_data: the max data of the input data, (divided_num, 1, 1)
    slice_method: the slice method of the data, tuple
    sliced_data: the sliced data of the input data, (divided_num, len(slice_method), row, col)
    sliced_weights: the weights of each slice for the sliced data, (len(slice_method),)
    sliced_max_weights: the max weights of each slice for the sliced data, (len(slice_method),)
    sliced_data_recalled: the flag to record the sliced data is calculated or not

    """
    def __init__(self, slice_method:torch.Tensor, bw_e=None, device=None):
        """
        the sliced data for the data slicing method with quantization
        :param data: the input data
        :param slice_method: the data slicing method, bit width of each slice, tuple
        :param bw_e: the bit width of the exponent,
                    if None, the exponent is not used, and the SlicedData is the uint type, the sign is the first bit
                    if not None, the SlicedData is fp type, the exponent is the last several bits
        :param device: use cpu or gpu, default is cpu (None)
        """
        self.bw_e = bw_e
        self.slice_method = slice_method
        self.device = torch.device('cpu') if device is None else device
        self.shape = None
        self._init_data(slice_method, bw_e, device)

    def _init_data(self, slice_method:torch.Tensor, bw_e, device):
        if bw_e is None:
            # optimize the calculation of the sliced_max_weights
            self.sliced_max_weights = torch.zeros(len(slice_method), device=device)
            self.sliced_weights = torch.zeros(len(slice_method), device=device)
            temp_s, i = 0, 0
            for slice in slice_method.flip(0):
                self.sliced_max_weights[i] = 2 ** slice - 1
                self.sliced_weights[i] = 2 ** temp_s
                temp_s += slice
                i += 1
            self.sliced_weights[-1] *= -1
        # fp type
        else:
            assert (slice_method[0] == 1 and slice_method[1] == 1), 'the first two slice bits should be 1'
            new_slice_method = slice_method
            # max weights of each slice
            self.sliced_max_weights = torch.Tensor([2 ** slice - 1 for slice in new_slice_method]).to(device)
            # the weights of each slice
            self.sliced_weights = torch.zeros_like(self.sliced_max_weights, device=device)
            temp_s, i = 0, 0
            for slice in new_slice_method:
                temp_s += slice
                self.sliced_weights[i] = 2 ** (2-temp_s)
                i += 1
            self.sliced_weights[0] *= -1

    def __repr__(self):
        return 'sliced data with slice_method:{}'.format(self.slice_method)

    def __len__(self):
        return len(self.slice_method)

    # @property
    # def shape(self):
    #     return self.quantized_data.shape

    def t(self):
        copy_ = copy.deepcopy(self)
        copy_.sliced_data = self.sliced_data.transpose(2, 3)
        copy_.quantized_data = self.quantized_data.T
        return copy_

    def size(self):
        return self.quantized_data.size()

    def slice_data_imp(self, engine, data, transpose=False):
        # implement the localized slicing of the data
        # the slice is determined by the local max data

        self.sliced_data, self.quantized_data, self.max_data = engine.slice_data(data,
                                                                                self.slice_method,
                                                                                transpose,
                                                                                 self.bw_e)
        self.shape = self.quantized_data.shape

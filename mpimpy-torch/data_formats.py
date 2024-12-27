# -*- coding:utf-8 -*-
# @File  : data_formats.py
# @Author: Zhou
# @Date  : 2024/1/19
import copy
import torch

class SlicedData(object):
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
    def __init__(self, slice_method:torch.Tensor, bw_e=None, input_en=False, dbfp_en=False, device=None, dtype=torch.float32):
        """
        the sliced data for the data slicing method with quantization
        :param data: the input data
        :param slice_method: the data slicing method, bit width of each slice, tuple
        :param bw_e: the bit width of the exponent,
                    if None, the exponent is not used, and the SlicedData is the uint type, the sign is the first bit
                    if not None, the SlicedData is fp type, the exponent is the last several bits
        :param input_en: enables row/column-wise input data quantization, default is False
                        if True, the input data is quantized by the data of input_size in DPETensor
        :param dbfp_en: enables the double bit fixed point, default is False
                        if True, there are two exponents, one for small values and one for large values
        :param device: use cpu or gpu, default is cpu (None)
        """
        self.bw_e = bw_e
        self.input_en = input_en
        self.dbfp_en = dbfp_en
        self.slice_method = slice_method
        self.device = torch.device('cpu') if device is None else device
        self.shape = None

        self.sliced_data = None
        self.quantized_data = None
        self.max_data = None
        self.dtype = dtype
        self.e_bias = None
        self._init_data(slice_method, bw_e, device)

    def _init_data(self, slice_method:torch.Tensor, bw_e, device):
        assert slice_method[0] == 1, 'the first slice should be 1'
        if bw_e is None:
            # optimize the calculation of the sliced_max_weights
            self.sliced_max_weights = torch.zeros(len(slice_method), device=device, dtype=self.dtype)
            self.sliced_weights = torch.zeros(len(slice_method), device=device, dtype=self.dtype)
            temp_s, i = 0, 0
            for s in slice_method.flip(0):
                self.sliced_max_weights[i] = 2 ** s - 1
                self.sliced_weights[i] = 2 ** temp_s
                temp_s += s
                i += 1
            self.sliced_weights[-1] *= -1
        # fp type
        else:
            self.sliced_max_weights = torch.zeros(len(slice_method), device=device, dtype=self.dtype)
            self.sliced_weights = torch.zeros(len(slice_method), device=device, dtype=self.dtype)
            temp_s, i = 0, 0
            for s in slice_method.flip(0):
                self.sliced_max_weights[i] = 2 ** s - 1
                self.sliced_weights[i] = 2 ** temp_s
                temp_s += s
                i += 1
            self.sliced_weights[-1] *= -1

    def __repr__(self):
        return 'sliced data with slice_method:{}'.format(self.slice_method)

    def __len__(self):
        return len(self.slice_method)


    def t(self):
        copy_ = copy.deepcopy(self)
        copy_.sliced_data = self.sliced_data.transpose(-4, -5)
        copy_.quantized_data = self.quantized_data.T
        copy_.max_data = self.max_data.transpose(0,1)
        return copy_

    def size(self):
        return self.quantized_data.size()

    def slice_data_imp(self, engine, data):
        """
        implement the localized slicing of the data
        :param engine: dot product engine, DPETensor
        :param data: tensor, 2D or 3D, if 2D, the shape is (row, col), if 3D, the shape is (batch, row, col)
        :return:
        """

        self.sliced_data, self.quantized_data, self.max_data, self.e_bias, self.dbfp_en = engine.slice_data(data,
                                                                                self.slice_method,
                                                                                self.bw_e,
                                                                                self.input_en,
                                                                                self.dbfp_en)
        self.shape = self.quantized_data.shape
# -*- coding: utf-8 -*-
# @Time    : 2022/6/29 15:08
# @Author  : Zhou
# @FileName: functions.py
# @Software: PyCharm

import numpy as np
import torch


def dot_2d(x, y):
    """
    use einsum to calculate the cross 2D product
    :param x: tensor with shape (batch, num_divide_row_a ,num_divide, slice_a ,m, n)
                             or (num_divide_row_a,num_divide, slice_a ,m, n)
    :param y: tensor with shape (num_divide,num_divide_col_b, slice_b ,n, p)
    """
    if len(x.shape) == 6:
        return torch.einsum("bnmijk, mpskl->bnmpisjl", x, y)
    elif len(x.shape) == 5:
        return torch.einsum("nmijk, mpskl->nmpisjl", x, y)
    else:
        raise ValueError('The input data dimension is not supported!')


def ABSE(ytest, ypred):
    return np.sum(np.abs((ytest-ypred)/ytest))/(ytest.shape[0] * ytest.shape[1])


def RE(ytest, ypred):
    return np.sqrt(np.sum((ytest-ypred)**2))/np.sqrt(np.sum(ytest**2))


def quant_map_tensor(mat, blk=(1, 1, 2, 4)):
    """
    convert the data to the quantized data
    :param mat: 5D tensor (num_divide_row_a,num_divide ,m, n) or 6D tensor (batch,num_divide_row_a,num_divide ,m, n)
    :param blk: slice method
    :return:
        data_int: the quantized data, if mat is 4D, the shape is (num_divide_row_a,num_divide, len(blk) ,m, n),
                    if mat is 5D, the shape is (batch, num_divide_row_a,num_divide, len(blk) ,m, n)
        mat_data: the data after quantization, the same shape as mat
        max_mat: the max value of the mat, the shape is (num_divide_row_a,num_divide 1, 1) or
                    (batch, num_divide_row_a,num_divide, 1, 1)
        e_bias: None, reserved for the block floating point
    """
    quant_data_type = torch.uint8 if max(blk) <= 8 else torch.int16
    e_bias = None
    assert blk[0] == 1
    bits = sum(blk)
    max_mat = torch.max(torch.max(torch.abs(mat), dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
    matq = torch.round(mat / max_mat * (2 ** (bits - 1) - 1)).int()
    mat_data = matq / (2 ** (bits - 1) - 1) * max_mat
    location = torch.where(matq < 0)
    matq[location] = 2 ** bits + matq[location]

    data_int = torch.empty((mat.shape[0], mat.shape[1], mat.shape[2], len(blk), mat.shape[3], mat.shape[4]),
                           device=mat.device, dtype=quant_data_type)
    b = 0
    for idx in range(len(blk)):
        data_int[:, :, :, idx, :, :] = ((matq - matq % 2 ** b) % 2 ** (b + blk[-1 - idx])) >> b
        b += blk[-1 - idx]
    return data_int, mat_data, max_mat, e_bias


def bfp_map_tensor(mat, blk=(1, 1, 2, 4), bw_e=8, dbfp_en=False):
    """
    convert the data to the quantized data with block floating point, which takes the block as the basic unit for quantization
    :param mat: 5D tensor (batch, num_divide_row_a,num_divide, m, n)
    :param blk: slice method
    :param bw_e: the bit width of the exponent, default is 8
    :param dbfp_en: enables the double bit fixed point, default is False
    :return:
    """
    quant_data_type = torch.uint8 if max(blk) <= 8 else torch.int16
    assert blk[0] == 1
    bits = sum(blk)
    abs_mat = torch.abs(mat)
    max_mat = torch.max(torch.max(abs_mat, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
    if dbfp_en is False:
        e_bias = torch.full_like(max_mat, -2**(bw_e-1)+1)
        e_bias[torch.where(max_mat > 0)] = torch.floor(torch.log2(max_mat[torch.where(max_mat > 0)]))
        matq = mat / 2 ** e_bias
        matq = torch.round(matq * 2 ** (bits - 2))
        clip_up = (2 ** (bits - 1) - 1).to(mat.device)
        clip_down = (-2 ** (bits - 1)).to(mat.device)
        matq = torch.clip(matq, clip_down, clip_up)  # round&clip
        mat_data = matq * 2 ** (e_bias + 2 - bits)  # 存储的是反量化后的数据
        location = torch.where(matq < 0)
        matq[location] = 2 ** bits + matq[location]
        if len(mat.shape) == 4:
            data_int = torch.empty((mat.shape[0], mat.shape[1], len(blk), mat.shape[2], mat.shape[3]),
                                   device=mat.device, dtype=quant_data_type)
            b = 0
            for idx in range(len(blk)): 
                data_int[:, :, idx, :, :] = ((matq - matq % 2 ** b) % 2 ** (b + blk[-1 - idx])) / (2**b)
                b += blk[-1 - idx]
        elif len(mat.shape) == 5:
            data_int = torch.empty((mat.shape[0], mat.shape[1], mat.shape[2], len(blk), mat.shape[3], mat.shape[4]),
                                   device=mat.device, dtype=quant_data_type)
            b = 0
            for idx in range(len(blk)):
                data_int[:, :, :, idx, :, :] = ((matq - matq % 2 ** b) % 2 ** (b + blk[-1 - idx])) / (2**b)
                b += blk[-1 - idx]
    else:
        mean_mat = torch.mean(abs_mat,dim=(-1,-2), keepdim=True)
        new_shape = (2,) + mat.shape
        new_mat = torch.zeros(new_shape,device=mat.device)
        # new_mat adds the 0th dimension to the mat dimension as 2
        e_bias = torch.ones(((2,)+max_mat.shape),device=mat.device)*(-2**(bw_e-1)+1)

        new_mat[0] = (abs_mat > mean_mat) * mat
        new_mat[1] = (abs_mat <= mean_mat) * mat
        e_bias[0][torch.where(max_mat > 0)] = torch.floor(torch.log2(max_mat[torch.where(max_mat > 0)]))
        e_bias[1][torch.where(mean_mat > 0)] = torch.floor(torch.log2(mean_mat[torch.where(mean_mat > 0)]))

        if len(mat.shape) == 4:
            e_bias = e_bias.permute(1, 2, 0, 3, 4)
            new_mat = new_mat.permute(1, 2, 0, 3, 4)
            matq = new_mat / 2 ** e_bias
            matq = torch.round(matq * 2 ** (bits-2))
            clip_up = (2 ** (bits - 1) - 1).to(mat.device)
            clip_down = (-2 ** (bits - 1)).to(mat.device)
            matq = torch.clip(matq, clip_down, clip_up)  # round & clip
            mat_data = matq * 2 ** (e_bias + 2 - bits)  # inverse quantization
            location = torch.where(matq < 0)
            mat_data = mat_data[:, :, 0, :, :]+mat_data[:, :, 1, :, :]
            location = torch.where(matq < 0)
            matq[location] = 2 ** bits + matq[location]      # complementary code
            shape_len = len(mat.shape)
            shape = list(mat.shape[:shape_len - 2]) + [len(blk),2] + list(mat.shape[shape_len - 2:])
            data_int = torch.empty(shape, device=mat.device, dtype=quant_data_type)
            b = 0
            for idx in range(len(blk)):
                for i in [0,1]:
                    data_int[:, :, idx, i, :, :] = ((matq[:, :, i, :, :] - matq[:, :, i, :, :] % 2 ** b) % 2 **
                                                    (b + blk[-1 - idx])) / (2 ** b)
                b += blk[-1 - idx]
        elif len(mat.shape) == 5:
            e_bias = e_bias.permute(1, 2, 3, 0, 4, 5)
            new_mat = new_mat.permute(1, 2, 3, 0, 4, 5)
            matq = new_mat / 2**e_bias
            matq = torch.round(matq*2**(bits-2))
            clip_up = (2 ** (bits - 1) - 1).to(mat.device)
            clip_down = (-2 ** (bits - 1)).to(mat.device)
            matq = torch.clip(matq, clip_down, clip_up)
            mat_data = matq * 2 ** (e_bias + 2 - bits)
            mat_data = mat_data[:, :, :, 0, :, :]+mat_data[:, :, :, 1, :, :]
            location = torch.where(matq < 0)
            matq[location] = 2**bits + matq[location]
            shape_len = len(mat.shape)
            shape = list(mat.shape[:shape_len - 2]) + [len(blk),2] + list(mat.shape[shape_len - 2:])
            data_int = torch.empty(shape, device=mat.device, dtype=quant_data_type)
            b = 0
            for idx in range(len(blk)):
                for i in [0,1]:
                    data_int[:, :, :, idx, i, :, :] = ((matq[:,:,:,i,:,:] - matq[:,:,:,i,:,:] % 2 ** b) %
                                                       2 ** (b + blk[-1 - idx])) / (2 ** b)
                b += blk[-1 - idx]
    return data_int, mat_data, max_mat, e_bias

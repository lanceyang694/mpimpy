import torch
import numpy as np

def ABSE(ytest, ypred):
    return np.sum(np.abs((ytest-ypred)/ytest))/(ytest.shape[0] * ytest.shape[1])

def RE(ytest, ypred):
    return np.sqrt(np.sum((ytest-ypred)**2))/np.sqrt(np.sum(ytest**2))

def quant_map_tensor(mat, blk=(1, 1, 2, 4)):
    '''
    convert the data to the quantized data
    :param mat: 4D tensor (batch, divide_num, row, col)
    :param blk: slice method
    :return:
        data_int: the quantized data, if mat is 3D, the shape is (divide_num, len(blk), row, col),
                    if mat is 4D, the shape is (batch, divide_num, len(blk), row, col)
        mat_data: the data after quantization, the same shape as mat
        max_mat: the max value of the mat, the shape is (divide_num, 1, 1) or (batch, divide_num, 1, 1)
        e_bias: None, reserved for the block floating point
    '''
    quant_data_type = torch.int8 if max(blk)<=8 else torch.int16
    e_bias = None
    assert blk[0] == 1
    bits = sum(blk)
    # (batch, divide_num, 1, 1)
    max_mat =  torch.max(torch.max(torch.abs(mat), dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
    matq = torch.round(mat / max_mat * (2 ** (bits - 1) - 1)).int()
    # record quantized data
    mat_data = matq / (2 ** (bits - 1) - 1) * max_mat
    # use location to reduce the function where
    location = torch.where(matq < 0)
    matq[location] = 2 ** bits + matq[location]
    data_int = torch.empty((mat.shape[0], mat.shape[1], len(blk), mat.shape[2], mat.shape[3]),
                           device=mat.device, dtype=quant_data_type)
    b = 0
    for idx, bits in enumerate(blk):
        data_int[:, :, idx, :, :] = ((matq - matq % 2 ** b) % 2 ** (b + blk[-1 - idx])) >> b
        b += blk[-1 - idx]
    return data_int, mat_data, max_mat, e_bias


def bfp_map_tensor(mat, blk=(1, 1, 2, 4), bw_e=8):
    '''
    convert the data to the quantized data with block floating point
    :param mat: 4D tensor (batch, divide_num, row, col)
    :param blk: slice method
    :return:
    '''
    quant_data_type = torch.int8 if max(blk) <= 8 else torch.int16
    assert blk[0] == 1
    bits = sum(blk)
    # (batch, divide_num, 1, 1)
    max_mat =  torch.max(torch.max(torch.abs(mat), dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]

    e_bias = torch.full_like(max_mat, -2**(bw_e - 1))
    # 对 max_mat 中大于 0 的元素取 log2 并 floor
    e_bias[torch.where(max_mat > 0)] = torch.floor(torch.log2(max_mat[torch.where(max_mat > 0)]))

    matq = mat / 2 ** e_bias
    matq = torch.round(matq * 2 ** (bits - 2)).int()
    clip_up = (2 ** (bits - 1) - 1).to(mat.device)
    clip_down = (-2 ** (bits - 1)).to(mat.device)
    matq = torch.clip(matq, clip_down, clip_up)  # round&clip
    mat_data = matq * 2 ** (e_bias + 2 - bits)  # 存储的是反量化后的数据

    location = torch.where(matq < 0)
    matq[location] = 2 ** bits + matq[location]

    data_int = torch.empty((mat.shape[0], mat.shape[1], len(blk), mat.shape[2], mat.shape[3]),
                           device=mat.device, dtype=quant_data_type)
    b = 0
    for idx in range(len(blk)):
        data_int[:, :, idx, :, :] = ((matq - matq % 2 ** b) % 2 ** (b + blk[-1 - idx])) >> b
        b += blk[-1 - idx]

    return data_int, mat_data, max_mat, e_bias

def dot_2d(x, y):
    """
    use einsum to calculate the cross 2D product
    :param x: tensor with shape (batch, divide_num, slice_x, m, n) or (divide_num, slice_x, m, n)
    :param y: tensor with shape (divide_num, slice_y, n, p)
    """
    if len(x.shape) == 5:
        return torch.einsum("bdxmn, dynp->bdxymp", x, y)
    elif len(x.shape) == 4:
        return torch.einsum("dxmn, dynp->dxymp", x, y)
    else:
        raise ValueError('The input data dimension is not supported!')
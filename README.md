# PimTorch: A Software-Hardware Co-Design Simulator for Memristor-Based In-Memory Computing

## *Make in-memory computing as easy as np.dot (Torch version)!*

In memory computing technology is a potential solution to overcome the von Neumann bottleneck and to provide significant improvements in energy efficiency. However, for in-memory computing, the algorithm and hardware are highly intertwined, making it challenging to design an optimal circuit. To address this issue, the author developed a Python package named "mPimPy" that embeds hardware simulation into the algorithm. From a user's perspective, mPimPy serves as the equivalent of the Numpy.dot function to implement the matrix multiplication which is the key operation of neural networks, machine learning, signal processing, and scientific computing. By simply replacing the `np.dot` function in your original code with `MapReducedot` and rerunning your program, you can simulate scenarios such as neural networks or any algorithm involving matrix multiplications. Whether you are a computer engineer or a microelectronic scientist, mPimPy enables you to quickly deploy your code on in-memory computing hardware or verify your circuit design solutions. 

## Note
For more details, please refer to [wiki](https://github.com/lanceyang694/mpimpy/wiki)

## Installing

Currently, mpimpy torch version is in development, and it can only be installed from the source code.


```shell
git clone -b pimtorch https://github.com/lanceyang694/mpimpy.git

cd mpimpy

#---------------------------------------------------
#       method 1: write code in this dictionary
#---------------------------------------------------
# install the package
pip install -r requirements.txt

# change to the files in pimtorch
cd pimtorch
# new files in this dictionary and run

#---------------------------------------------------
#       method 2: install the package
#---------------------------------------------------
# install the package
python setup.py install

#---------------------------------------------------
#       method 3: install the package by pip
#---------------------------------------------------
cd dist
pip install pimtorch-0.1.1-py3-none-any.whl

```

## Version Information
### 0.0.1   
For INT data support only
### 0.0.2
We add the FP data support fot the dataformat
Changed the consideration for the batch input data
### 0.0.3
Update includes:
- Added support for different splitting granularity of input data
  - The input data can be sliced by the matrix mode and row mode
  - It is controlled by "input_en" in class "SlicedData"
- Added support for different splitting granularity of weight
  - For FP data, the data slice can support the double exponet quantization
## Example

```python
# from matplotlib import pyplot as plt
# from memmat_tensor import DPETensor
# from data_formats import SlicedData
# from utils import SNR
# import torch

    
tb_mode = 1
device = torch.device('cuda:0')
if tb_mode == 0:
    torch.manual_seed(42)
    x_data = torch.randn(2, 1000, 1000, dtype=torch.float64, device=device)
    mat_data = torch.randn(1000, 1000, dtype=torch.float64, device=device)
    mblk = torch.tensor([1, 1, 2, 4])
    xblk = torch.tensor([1, 1, 2, 4])
    mat = SlicedData(mblk, device=device, bw_e=8, is_weight=True, quant_gran=(64, 64), paral_size=(64, 64))
    x = SlicedData(xblk, device=device, bw_e=8, quant_gran=(64, 64), paral_size=(64, 64))
    engine = DPETensor(var=0.05, g_level=16, rdac=16, radc=2 ** 8)
    mat.slice_data_imp(engine, mat_data)
    x.slice_data_imp(engine, x_data)
    start = time.time()
    result = engine(x, mat).cpu().numpy()
    rel_result = torch.matmul(x_data, mat_data).cpu().numpy()
    snr_varlue = SNR(result, rel_result)
    print("SNR(dB)", snr_varlue)
    plt.scatter(rel_result.reshape(-1), result.reshape(-1))
    plt.xlabel('Expected Value of Dot Product')
    plt.ylabel('Measured Value of Dot Product')
    plt.show()
elif tb_mode == 1:
    torch.manual_seed(42)
    x_data = torch.randn(1000, 1000, dtype=torch.float64, device=device)
    mat_data = torch.randn(1000, 1000, dtype=torch.float64, device=device)
    mblk = torch.tensor([1, 1, 2, 2])
    xblk = torch.tensor([1, 1, 2, 2])
    size = 64
    paral_size = size
    mat = SlicedData(mblk, device=device, bw_e=None, is_weight=True,
                     paral_size=(paral_size, paral_size), quant_gran=(1,size))
    x = SlicedData(xblk, device=device, bw_e=None, paral_size=(paral_size, paral_size), quant_gran=(1,size))
    engine = DPETensor(var=0.00, g_level=16, rdac=16, radc=2**12)
    mat.slice_data_imp(engine, mat_data)
    x.slice_data_imp(engine, x_data)
    start = time.time()
    result = engine(x, mat).cpu().numpy()
    rel_result = torch.matmul(x_data, mat_data).cpu().numpy()
    snr_varlue = SNR(result, rel_result)
    print("SNR(dB)", snr_varlue)

    plt.scatter(rel_result.reshape(-1), result.reshape(-1))
    plt.xlabel('Expected Value of Dot Product')
    plt.ylabel('Measured Value of Dot Product')
    plt.show()

```

## Author

Author: Ling Yang

Email: [3299285328@qq.com](mailto:3299285328@qq.com)

## Contributors

- Houji Zhou, [1499403578@qq.com](mailto:1499403578@qq.com), [zhouhouji (houjizhou) · GitHub](https://github.com/zhouhouji)

- Zhiwei Zhou, [1548384176@qq.com](mailto:1548384176@qq.com)

- Yuyang Fu, [412983100@qq.com](mailto:412983100@qq.com)

## Maintainer

- Maintainer: Researchers from Prof. Xiangshui Miao and Prof. Yi Li's group at HUST

- Affiliation: Huazhong University of Science and Technology, School of Integrated Circuit,  [Institute of Information Storage Materials and Devices (hust.edu.cn)](http://ismd.hust.edu.cn/)

Any advice and criticism are highly appreciated on this package. Naturally, you can also modify the source code to suit your needs. In upcoming versions, we plan to continually incorporate the latest research findings into mPimPy. Stay tuned!

## References

[1] [Zidan M A, Jeong Y J, Lee J, et al. A general memristor-based partial differential equation solver[J]. Nature Electronics, 2018, 1(7): 411-420.](https://www.nature.com/articles/s41928-018-0100-6)

[2] [Li C, Hu M, Li Y, et al. Analogue signal and image processing with large
 memristor crossbars[J]. Nature electronics, 2018, 1(1): 52-59.](https://www.nature.com/articles/s41928-017-0002-z)

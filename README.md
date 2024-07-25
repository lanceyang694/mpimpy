# mPimPy: A Software-Hardware Co-Design Simulator for Memristor-Based In-Memory Computing

## *Make in-memory computing as easy as np.dot !*

In memory computing technology is a potential solution to overcome the von Neumann bottleneck and to provide significant improvements in energy efficiency. However, for in-memory computing, the algorithm and hardware are highly intertwined, making it challenging to design an optimal circuit. To address this issue, the author developed a Python package named "mPimPy" that embeds hardware simulation into the algorithm. From a user's perspective, mPimPy serves as the equivalent of the Numpy.dot function to implement the matrix multiplication which is the key operation of neural networks, machine learning, signal processing, and scientific computing. By simply replacing the `np.dot` function in your original code with `MapReducedot` and rerunning your program, you can simulate scenarios such as neural networks or any algorithm involving matrix multiplications. Whether you are a computer engineer or a microelectronic scientist, mPimPy enables you to quickly deploy your code on in-memory computing hardware or verify your circuit design solutions. 

## Note
Although this package can realize part of the function in 'main' branch, it lacks the support for FP data and line resistance.

The aim of this torch version mpimpy is to simulate memristive VMM in deep learning layers including the convolution and FC layers. 
The full version of the package will be released recently.

## Installing

Currently, mpimpy torch version is in development, and it can only be installed from the source code.


```shell
git clone -b mpimpy-torch https://github.com/lanceyang694/mpimpy.git

cd mpimpy

# install the package
pip install -r requirements.txt

# change to the files in mpimpy-torch
cd mpimpy-torch
# new files in this dictionary and run

```

## Version Information
### 0.0.1   
For INT data support only
### 0.0.2
We add the FP data support fot the dataformat
Changed the consideration for the batch input data


## Example

```python
from matplotlib import pyplot as plt
from memmat_tensor import DPETensor
from data_formats import SlicedData

    
tb_mode = 1
# test for INT format
if tb_mode == 0:
    torch.manual_seed(42)
    device = torch.device('cuda:0')
    x_data = torch.randn(4000, 1000, device=device)
    mat_data = torch.randn(1000, 1200, device=device)
    xblk = torch.tensor([1, 1, 4, 4])
    mblk = torch.tensor([1, 1, 4, 4])
    mat = SlicedData(mblk, device=device)
    x = SlicedData(xblk, device=device)

    engine = DPETensor(var=0.0)
    x.slice_data_imp(engine, x_data, transpose=True)
    mat.slice_data_imp(engine, mat_data)
    start = time.time()
    result = engine(x, mat)
    end = time.time()
    print("Tensor time: ", end - start)
    # result = engine._test(x, mat)
    result = result.cpu().numpy()

    rel_result = torch.matmul(x_data, mat_data).cpu().numpy()

    print(RE(result, rel_result))
    plt.scatter(rel_result.reshape(-1), result.reshape(-1))
    plt.xlabel('Expected Value of Dot Product')
    plt.ylabel('Measured Value of Dot Product')
    plt.show()
# test for FP format
elif tb_mode == 1:
    torch.manual_seed(42)
    device = torch.device('cuda:0')
    x_data = torch.randn(4000, 1000, device=device)
    mat_data = torch.randn(1000, 1200, device=device)
    xblk = torch.tensor([1, 1, 4, 4])
    mblk = torch.tensor([1, 1, 4, 4])
    mat = SlicedData(mblk, device=device, bw_e=8)
    x = SlicedData(xblk, device=device, bw_e=8)

    engine = DPETensor(var=0.0)
    x.slice_data_imp(engine, x_data, transpose=True)
    mat.slice_data_imp(engine, mat_data)
    start = time.time()
    result = engine(x, mat)
    end = time.time()
    print("Tensor time: ", end - start)
    # result = engine._test(x, mat)
    result = result.cpu().numpy()

    rel_result = torch.matmul(x_data, mat_data).cpu().numpy()

    print(RE(result, rel_result))
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

[1]    [Zidan M A, Jeong Y J, Lee J, et al. A general memristor-based partial differential equation solver[J]. Nature Electronics, 2018, 1(7): 411-420.](https://www.nature.com/articles/s41928-018-0100-6)

[2]     [Li C, Hu M, Li Y, et al. Analogue signal and image processing with large
 memristor crossbars[J]. Nature electronics, 2018, 1(1): 52-59.](https://www.nature.com/articles/s41928-017-0002-z)
# mpimpy
# mpimpy

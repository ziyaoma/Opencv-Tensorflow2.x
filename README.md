# Opencv-Tensorflow2.x
1、用opencv调用tensorflow2.x的keras训练的模型

  tensorflow 2.2.0
  opencv 4.2.0.32

  Python：
    文件里面包含训练和测试，tensorflow的调用和opencv调用的测试都有
    opencv-tfkeras1.py
    简单的线性回归的例子

    opencv-tfkeras2.py
    手写数字的例子

  C++：
    只有测试部分：
    Copencv-tfk.cpp
    C++和opencv的调用的例子
  
  
2、用libtorch 调用torch训练的模型
  1.11.0两个版本保持一致
  deeplabv3图像分割的例子，训练模型：https://github.com/bubbliiiing/deeplabv3-plus-pytorch

  训练pth文件后
  Python：
     pth_pt.py : 将保存的pth转换为pt，并结合opencv调用测试
  c++：
     toch_pttest.cpp ：c++测试，libtorch调用测试


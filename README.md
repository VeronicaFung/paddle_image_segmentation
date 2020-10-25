# Paddlepaddle图像分割学习笔记及UNet应用实例
这篇学习笔记分为两部分，先是对课程进行一个大框架总结，记录一些概念性的东西；第二部分则是对于我来说，学习图像分割的目的是在医学图像数据、或者生物学网络上对特定生物学问题建模和预测，因此学习了AIstudio中大神用Unet对肝脏数据进行图像分割的项目，并试图自己写个简单的版本，第二部分可能因为时间的关系（而且python不是我很擅长的语言）效果可能很差。

**注：**
*以下内容仅为我自己的理解，如果有误，请评论纠正。*
*该部分截图来自于飞桨深度学习学院中“图像分割7日打卡营”课程中朱老师和伍老师的ppt。*
## 定义
**图像分割（Image Segmentation）**


***处理对象**：图片（或视频）*
***stuffs**：无固定形状的物体（可以理解成不可数名词）*
***things**：有固定形状的物体（可以理解成可数名词）*

* 语义分割(Semantic Segmentation)：像素级别的分类，即对每个Pixel进行分类;注重类别间的区分。
* 实例分割(Instance Segmentation): 目标检测+像素级分割；注重个体间的区分。
* 全景分割(Panoptic Segmentation): 语义分割与实例分割的结合，每个 stuff 类别与 things 类别都被分开，且things中的每个个体也被分开。
<br>
* *视频目标分割（VOS）:给定目标mask，求特定目标的mask*
* *视频实例分割（VIS）:根据目标检测框，求目标的mask*

## 应用场景
1. 人像分割
2. 自动驾驶
3. 医学图像（CT、MRI、病理染色）：常用UNet
4. 工业质检、分拣机器人
![image](https://github.com/VeronicaFung/paddle_image_segmentation/blob/main/Image/FCN.PNG)
## 模型流程
![image](https://github.com/VeronicaFung/paddle_image_segmentation/blob/main/Image/workflow.PNG)

这个流程和之前上的“百度架构师手把手带你实现零基础小白到AI工程师的华丽蜕变”讲到的深度学习模型流程图很相似，为了方便理解，把当时课件里的图也贴过来：
![image](https://github.com/VeronicaFung/paddle_image_segmentation/blob/main/Image/workflow2.PNG)


## 评价指标
1. ACC：预测结果与真实（Ground Truth）对应位置像素的分类准确率
   
2. IoU：分割每一类别与真实类别之间的交并比（可以理解为二者之间计算Jaccard index）

## 模型

### FCN (Fully Convolutional Networks)
Backbone: VGG
**模型结构：**
![image](https://github.com/VeronicaFung/paddle_image_segmentation/blob/main/Image/FCN.PNG)

FCN with Multilayer Fusion:
* 集成多层feature map；不同采样率；以Element-wise add为结合方式
![image](https://github.com/VeronicaFung/paddle_image_segmentation/blob/main/Image/FCN2.PNG)
1. 与图像分类区别，替换FC为1x1Conv（只改变Channel大小）
2. Feature map尺寸变大：
   1\) Upsampling: 上采样，双线性插值
   2\) Transpose Conv：反卷积，相当于kernel顺时针旋转180度后对feature map做加padding的卷积（paddle中的api fluid.dygraph.Conv2DTranspose提供了实现的方式）
   3\) Unpooling: pooling的反操作，forward时需要提供index。

**优点：**
1. 任意尺寸输入
2. 效率较高
3. 结合浅层信息

**缺点：**
1. 分割结果不够精细
2. 没有考虑上下文信息

### U-Net
1. U-Net对FCNnet进行了改进，通过“Encoder-Decoder”形式的U型结构实现网络搭建。
2. 输入输出保持大小不变
3. 利用了skip connect机制：SKIP结合方式采用Concatenation；若尺寸发生变化则进行crop；Concat之后再过Conv层
4. 输出层与FCN相同，为一个1x1卷积，kernel个数为类别数。

**模型结构：**
![image](https://github.com/VeronicaFung/paddle_image_segmentation/blob/main/Image/UNet.PNG)

*U-net的主要优点为：*

    * 支持少量的数据训练模型
    * 通过对每个像素点进行分类，获得更高的分割准确率
    * 用训练好的模型分割图像，速度快
    Ref: https://blog.csdn.net/TDhuso/article/details/79948494


### PSPNet
全称：Pyramid Scene Parsing Network
* 如何应对FCN中缺少上下文信息这一缺点？利用全局信息(global information)
* 如何利用全局信息？增大感受野（Receptive Field，用于产生特征的输入图像中区域大小）
* 如何增大感受野？不同scale的“feature pyramid”，通过不同bin size的“adaptive pool + Conv1x1 + BN层 + Upsample(interpolate会输入的size)”concat在一起后再过Conv层得到分割结果。

**模型结构：**
![image](https://github.com/VeronicaFung/paddle_image_segmentation/blob/main/Image/PSPnet.PNG)

PSPnet的Backbone是Dilated ResNet，是在原始的ResNet基础上改用了dilated Conv层（空洞卷积），可以增大感受野，不降低分辨率，同时不引入额外参数和计算量。
![image](https://github.com/VeronicaFung/paddle_image_segmentation/blob/main/Image/PSPnet2.PNG)

**疑问：** 可以理解成对kernel进行插值填0？这里理解的不太好。

![image](https://github.com/VeronicaFung/paddle_image_segmentation/blob/main/Image/PSPnet3.PNG)


### DeepLab系列
#### DeepLab v1
* backbone: VGG 5层空洞卷积
* FC6: 空洞卷积
* FC7: Conv 1x1
* Classification分类：
  * 用Conv(stride)控制feature map尺寸+ReLU+dropout
  * Conv1x1+ReLU+dropout
  * Conv1x1, number_filters = num_classes
* 分类输出的feature map进行Elementwise相加
* Interpolation （8倍）
* Output
**模型结构：**
![image](https://github.com/VeronicaFung/paddle_image_segmentation/blob/main/Image/deeplab_v1.PNG)

#### DeepLab v2
* backbone: ResNet(4 layers), resBlock [3,4,23,3]
* ASPPmodule
* 分类输出的feature map（stride控制feature map大小）进行Elementwise相加
* Label下采样保证size

**模型结构：**
![image](https://github.com/VeronicaFung/paddle_image_segmentation/blob/main/Image/deeplab_v2.PNG)

**ASPP模块**
* 通过设置不同dilation（[6,12,18,24]），去捕捉不同感受野的信息。
* 通过加padding保持生成的feature map尺寸保持不变。
![image](https://github.com/VeronicaFung/paddle_image_segmentation/blob/main/Image/deeplab_v2_ASPP.PNG)

#### DeepLab v3
* 在DeepLab v2基础上进行改进。

**模型结构：**
![image](https://github.com/VeronicaFung/paddle_image_segmentation/blob/main/Image/deeplab_v3.PNG)

* Backbone: Multi-grid ResNet
  * DeepLab v2的backbone基础上增加3个ResBlock。
  * 给每个ResBlock里设置不同的Dilation（按Multi-Grid Rate = [1,2,4]设置)

![image](https://github.com/VeronicaFung/paddle_image_segmentation/blob/main/Image/deeplab_v3_ASPPplus.PNG)

相比于DeepLab v2中的ASPP模块，v3中除了dilated conv层外还增加了一个1x1Conv层和一个Adaptive Pool+Interpolation。
  * 1x1Conv层: 改变channel大小，C维的升降维
  * A daptive Pool（+Interpolation）:  改变HxW的大小，变成C维的HW=1x1的形状。
  * 生成的feature maps用concat。
![image](https://github.com/VeronicaFung/paddle_image_segmentation/blob/main/Image/deeplab_v3_ASPPplus.PNG)

### GCN，实例分割与全景分割
这部分我没太听懂，所以在学习笔记中就暂时不放了，之后再补上。



## 部分课后作业代码集成
* 放置了UNet，PSPnet和DeepLab v3以及train部分代码，该部分不能保证正确...
* FCN在课程学习中我写的没有成功跑出来，因此没有放在这里。
* 具体代码见models文件夹。
* 没有放上数据集，所以可能跑不起来，仅供参考。
* 
```{bash}
# deeplab
python ./work/unet.py
# deeplab
python ./work/unet.py
# deeplab
python ./work/deeplab.py

# train.py --

```

## 实例： 使用UNet实现肝脏-肿瘤的分割

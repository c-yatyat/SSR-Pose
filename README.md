# SSR-Pose

Scale-Sensitive Representation for Multi-Person Pose Estimation
尺度感知的多人姿态估计SSR-Pose

在HigherHRNet的基础上，加入变形卷积的概念，增强模型尺度感知力。


## 安装教程

1.  建议事先参考[mmpose](https://github.com/open-mmlab/mmpose)官网，了解代码库脉络，并安装好mmcv和mmpose。

*注意：这个mmpose的版本为0.16.0，可以在`./mmpose/version.py`里查看。为确保减少安装成本，建议去回看旧版本对应的官方文档。*

2.  安装教程类似，本库的`./requirements`里将所有github链接替换成gitee，网速更快，可以下载替换。
3.  analyze分支里提供coco-analyze误差分析工具及部分结果，可能需要额外下载LaTeX编辑器来获得较好阅读体验的分析报告。

除官方教程外，建议自行下载的工具：
- tensorboardX 便于实时观察训练效果；
- cocoapi 适配coco数据集；
- LaTeX 编辑器，Linux/macOS建议下载TexLive、Windows建议TeXStudio。



## 使用说明

1.  建议参照[mmpose](https://github.com/open-mmlab/mmpose)官网的中文教程，不嫌弃的话也可以看看我曾经写过的知乎专栏[人体姿态咕咕](https://www.zhihu.com/column/c_1329419002742157312)，了解一下代码库大概结构和使用方式等。
2.  关于方法的集成和修改集中在`./configs`里面，比如`./configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/`里面储存用coco数据集做bottom-up系列2d人体姿态估计的算法。
3.  关于模块的修改集中在`./mmpose`内的所有小文件夹里，举个例子`./mmpose/models/backbones/dcn_adaptive.py`就是尝试加入adaptive方法的小模块。
4.  如果想调用修改后的模块方法，需要事先注册，比如`./mmpose/models/backbones/__init__.py`里面替换或重新指定。

*注意此类文件不止一个，建议通过vscode代码调试，摸清代码运行逻辑后，再逐个击破。*

5.  关于输出训练中间结果，建议使用代码调试，手动在调试区，添加输出代码，不破坏整个运行逻辑。
6.  关于可视化出测试demo图片和视频，需要修改的文件主要集中在`./tests/`和`./demo/bottom_up_img_demo.py`附近，比如`./tests/data/ablation_study/test_coco_lan.json`，这部分相关的可以参考官方文档。
7.  关于训练相关的，由于没找到支持可变形卷积的参数量计算函数，而mmpose提供的代码无法正确估算，所以只用运行时间做参考。相关类似的分析工具主要集中在`./tools/analysis/get_flops.py`附近，可以自行摸索。



## 其他说明

1.  非计算机科班出身，代码习惯不好，加上时间紧迫，许多代码没有良好注释习惯。如果大家看不懂别来找我，我毕业了啥都不记得了。
2.  可以看看mmpose官网的更新迭代，这方法比较简单粗暴和陈旧，现在有好多新的backbone和trick面世，值得寻求更好的思路。
3.  写博客是因为有非常多的网络大佬帮助过我，有一句话是说“初学者才是初学者最好的老师”，就是一种同学的感觉。希望我这几年没白干，能帮到一些朋友就是帮到！
4.  如果你关心的话，我现在不搞算法，去做文书工作了。计算机视觉让我知道了自己非常擅长写文书，虽然这篇Readme写的有点粗糙，但是肯定比代码写得好。润了。


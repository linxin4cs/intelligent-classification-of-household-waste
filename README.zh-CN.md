# 生活垃圾智能分类
[English](./README.md) | 中文

一个用来存储我的比赛代码的仓库（第七届全国大学生工程训练综合能力竞赛选拔赛，中国）

## 项目介绍
本项目是 "第七届全国大学生工程训练综合能力竞赛选拔赛" 的参赛作品，是一个基于深度学习的生活垃圾智能分类系统。主要工作是对生活垃圾数据集进行预处理，然后使用深度学习的方法预训练一个模型，同时使用PyQt5进行界面设计，实现生活垃圾智能分类系统的可视化。最后配合单片机联动机械结构，实现生活垃圾智能分类系统的自动化运作。

## 项目结构
本项目的主要结构如下：
- recognition: 图像识别代码
  - assets: 宣传视频
  - utils: 辅助函数
  - weights: 预训练模型
  - main.py: 主程序
  - UI.py: 界面设计
- singlechip: 单片机代码（机器码）
  - gpio_in.hex: 单片机输入
  - gpio_out.hex: 单片机输出
- train: 训练代码
  - data_sort.py: 对原始数据进行处理
  - folder_rename.py: 将文件夹的名称由编号更改为实际名称
  - test.py: 测试模型
  - train.py: 训练模型
  - garbage_classify.json: 作为一个映射表，方便程序快速查找和引用每个类别的具体名称

## 技术栈
- PySerial: 用于串口通信
- OpenCV-Python: 提供图像和视频处理的工具，包括捕捉、操作和保存
- Pytorch: 用于计算机视觉
- threading: 用于执行并发操作
- PyQt5: 用于界面设计
- ...
[TOC]

## numpy_data_cmp

简介：逐个对比两个numpy的每一个数据，输出每个数据的误差，和最后统计的最大误差以及误差比。

## one_step_accuracy_cmp

简介：全流程精度比对工具，基于《开发辅助工具指南（推理）》的《7 精度比对工具使用指南》，串起流程，只需一步操作即可生成中间数据、对比结果。

## oxinterface

简介：一套操作简单的onnx改图接口。

功能：

1. 修改常量节点（Initializer）的名字、数据类型、数据值
2. 新增常量节点（Initializer）
3. 保存常量节点（Initializer）的数据
4. 向模型中插入节点
5. 从模型中截取单节点、从模型中截取一段模型
6. 获取模型输入、输出、中间tensor、Initializer的信息（包括：tensor名字、shape、类型）
7. dump模型所有节点的数据
8. ...

## pyacl_infer

简介：一套封装好的python acl推理脚本，类似benchmark，支持动态shape，支持推理时间统计。

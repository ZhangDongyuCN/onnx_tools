# 功能

逐个对比两个numpy的每一个数据，输出每个数据的误差，和最后统计的最大误差以及误差比（误差比=有误差的数据个数/数据总个数）。

**要求：** 两个numpy的数据个数必须一致。

# 使用示例

## 对比npy类型数据

```shell
python numpy_data_cmp.py --type=npy --left_data_path=./left_data.npy --right_data_path=./right_data.npy
```

## 对比bin类型数据

```shell
python numpy_data_cmp.py --type=bin --left_data_path=./left_data.npy --left_data_type=np.float32 --right_data_path=./right_data.npy --right_data_type=np.float32
```

## 查看帮助

```shell
python numpy_data_cmp.py -h
```


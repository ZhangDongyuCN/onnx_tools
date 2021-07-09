# 参数说明

- --model_path：模型路径
- --device_id：npu id
- --cpu_run：MeasureTime类的cpu_run参数，`True` 或 `False`，默认就用`True`
- --sync_infer：推理方式
  - True：同步推理（若要和GPU推理时间做对比，只能用同步推理模式计算时间）
  - False：异步推理
- --workspace：类似TensorRT `workspace`参数，计算平均推理时间时排除前n次推理
- --input_info_file_path：类似benchmark的bin_info文件
- --input_dtypes：模型输入的类型，用逗号分割（参考后续类型说明）
  - e.g. 模型只有一个输入：`--input_dtypes=float32`
  - e.g. 模型有多个输入：`--input_dtypes=float32,float32,float32`（需要和bin_info文件多输入排列一致）
- --infer_res_save_path：推理结果保存目录
- --res_save_type：推理结果保存类型，`bin`或`npy`

# 类型说明

```python
DTYPE = {
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
    'int16': np.int16,
    'int32': np.int32,
    'int64': np.int64,
    'uint8': np.uint8,
    'uint16': np.uint16,
    'uint32': np.uint32,
    'uint64': np.uint64
}
```

# bin_info文件说明

因为支持动态shape，相比于benchmark的bin_info文件，需要多加一列shape信息，e.g.

```
0 ./bert_bin/input_ids_0.bin (1,512)
0 ./bert_bin/segment_ids_0.bin (1,512)
0 ./bert_bin/input_mask_0.bin (1,512)
1 ./bert_bin/input_ids_1.bin (1,512)
1 ./bert_bin/segment_ids_1.bin (1,512)
1 ./bert_bin/input_mask_1.bin (1,512)
```

**注意：** shape信息不能加空格，例如：`(1,512)`不能写成`(1, 512)`，后者逗号后边有个空格。

# 使用示例

1、导入环境变量

```shell
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/pyACL/python/site-packages/acl:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
```

**注意：** `PYTHONPATH`环境变量中要加入`acl`模块的路径`${install_path}/pyACL/python/site-packages/acl`，否则运行脚本会提示无法找到`acl`模块，上述环境变量中已经加入了。

2、运行脚本，开始推理

```python
python3.7 pyacl_infer.py \
--model_path=./bert_base_uncased.om \
--device_id=0 \
--cpu_run=True \
--sync_infer=True \
--workspace=10 \
--input_info_file_path=./input.info \
--input_dtypes=int64,int64,int64 \
--infer_res_save_path=./infer_res \
--res_save_type=bin
```


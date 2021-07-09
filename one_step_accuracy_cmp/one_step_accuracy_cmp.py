import os
import time
import onnx
import yaml
import shutil
import argparse
import numpy as np
import onnxruntime as rt
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs, \
    select_model_inputs_outputs, save_onnx_model


def dump_om_data(config):
    # 取出配置信息
    input_info = config['input_info']
    om_path = config['om_path']
    dump_save_path = config['dump_save_path']
    npu_device_id = config['npu_device_id']
    msame_path = config['msame_path']
    msaccucmp_py_path = config['msaccucmp_py_path']
    env = config['env']

    # 生成input数据
    om_input = []
    for info in input_info:
        om_input.append(info['bin_path'])
    om_input = ','.join(om_input)

    # 创建dump_save_path
    if not os.path.exists(dump_save_path):
        os.makedirs(dump_save_path)

    # 先删除dump_save_path/dump目录
    dump_dir = os.path.join(dump_save_path, 'dump')
    if os.path.exists(dump_dir):
        shutil.rmtree(dump_dir)

    # dump om模型数据
    cmd = 'chmod +x ' + msame_path + '\n' + \
          msame_path + ' --model ' + om_path + \
          ' --input ' + om_input + \
          ' --dump true ' \
          ' --device ' + str(npu_device_id) + \
          ' --output ' + dump_save_path
    print('[INFO] Shell cmd:\n', cmd)
    cmd = '\n' + env + '\n' + cmd
    error_code = os.system(cmd)
    if error_code != 0:
        raise RuntimeError('[ERROR] Dump om data failed, please check the config.yaml !')

    # 删除运行msame得到的推理结果文件，e.g. dump_save_path/20210610_102637
    dirs = os.listdir(dump_save_path)
    for dir in dirs:
        if len(dir) == 15:  # 「20210610_102637」的长度是15
            shutil.rmtree(os.path.join(dump_save_path, dir))

    # 查看是否dump成功
    dirs = os.listdir(dump_dir)
    if len(dirs) == 0:
        raise RuntimeError('[ERROR] Dump om data failed, please check the config.yaml !')
    elif len(dirs) > 1:
        raise RuntimeError('[ERROR] Please delete dump_save_path/dump dir before run !')
    om_dump_dir_name = dirs[0]

    # 把本次dump数据从dump_save_path/dump目录下移动到dump_save_path下
    shutil.move(os.path.join(dump_dir, om_dump_dir_name), dump_save_path)
    shutil.rmtree(dump_dir)

    # dump数据转numpy
    om_name = os.path.splitext(os.path.basename(om_path))[0]
    om_dump_raw_path = os.path.join(dump_save_path, om_dump_dir_name, str(npu_device_id), om_name, '1/0')
    om_dump_trans_path = os.path.join(dump_save_path, om_dump_dir_name + '_om')
    os.mkdir(om_dump_trans_path)
    cmd = 'python3.7 ' + msaccucmp_py_path + ' convert -d ' + om_dump_raw_path + ' -out ' + om_dump_trans_path
    print('[INFO] Shell cmd:\n', cmd)
    error_code = os.system(cmd)
    if error_code != 0:
        raise RuntimeError('[ERROR] Convert om dump data to numpy failed, please check !')

    return om_dump_raw_path, om_dump_dir_name


def onnx_dump_data(config, om_dump_dir_name):
    # 取出配置信息
    input_info = config['input_info']
    onnx_path = config['onnx_path']
    dump_save_path = config['dump_save_path']

    # 修改模型，增加输出节点
    onnx_name = os.path.basename(onnx_path)
    modify_onnx_path = os.path.splitext(onnx_name)[0] + '-modify.onnx'
    model_onnx = onnx.load(onnx_path)
    outputs = []
    for out in enumerate_model_node_outputs(model_onnx):
        outputs.append(out)
    if not os.path.exists(modify_onnx_path):
        modify_onnx = select_model_inputs_outputs(model_onnx, outputs=outputs)
        save_onnx_model(modify_onnx, modify_onnx_path)

    # 生成input数据
    onnx_input = {}
    for info in input_info:
        data = np.fromfile(info['bin_path'], dtype=eval(info['dtype'])).reshape(eval(info['shape']))
        onnx_input[info['input_name']] = data

    # 推理得到输出
    sess = rt.InferenceSession(modify_onnx_path)
    infer_res = sess.run(outputs, onnx_input)

    # 创建目录
    onnx_dump_path = os.path.join(dump_save_path, om_dump_dir_name + '_onnx')
    os.mkdir(onnx_dump_path)

    # 保存数据
    assert len(infer_res) == len(outputs)
    idx = 0
    for node in model_onnx.graph.node:
        for i in range(len(node.output)):
            file_name = node.name + '.' + str(i) + '.' + str(round(time.time() * 1000000)) + '.npy'
            data_save_path = os.path.join(onnx_dump_path, file_name)
            np.save(data_save_path, infer_res[idx].astype(np.float16))
            idx += 1
    assert idx == len(infer_res)

    return onnx_dump_path


def accuracy_cmp(config, om_dump_dir_name, om_dump_raw_path, onnx_dump_path):
    # 取出配置信息
    om_path = config['om_path']
    dump_save_path = config['dump_save_path']
    msaccucmp_py_path = config['msaccucmp_py_path']
    env = config['env']

    # om转json
    om_name = os.path.splitext(os.path.basename(om_path))[0]
    om_json_path = om_name + '.json'
    if not os.path.exists(om_json_path):
        cmd = 'atc --mode=1 --om=' + om_path + ' --json=' + om_json_path
        print('[INFO] Shell cmd:\n', cmd)
        cmd = '\n' + env + '\n' + cmd
        error_code = os.system(cmd)
        if error_code != 0:
            raise RuntimeError('[ERROR] Convert om model to json failed, please check !')

    # 创建目录
    cmp_path = os.path.join(dump_save_path, om_dump_dir_name + '_cmp')
    os.mkdir(cmp_path)

    # 精度对比
    cmd = 'python3.7 ' + msaccucmp_py_path + \
          ' compare -m ' + om_dump_raw_path + \
          ' -g ' + onnx_dump_path + \
          ' -f ' + om_json_path + \
          ' -out ' + cmp_path
    print('[INFO] Shell cmd:\n', cmd)
    error_code = os.system(cmd)  # 运行成功返回的竟然是512，而不是0，也不知道后期会不会改成0
    if error_code != 512:
        raise RuntimeError('[ERROR] Run msaccucmp.py failed, please check !')


def one_step_accuracy_cmp(config):
    om_dump_raw_path, om_dump_dir_name = dump_om_data(config)
    onnx_dump_path = onnx_dump_data(config, om_dump_dir_name)
    accuracy_cmp(config, om_dump_dir_name, om_dump_raw_path, onnx_dump_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='一步式精度比对工具')
    parser.add_argument('-c', '--config_yaml_path', required=True)
    opt = parser.parse_args()

    config = yaml.load(open(opt.config_yaml_path, 'r', encoding='utf-8'), Loader=yaml.FullLoader)

    one_step_accuracy_cmp(config)

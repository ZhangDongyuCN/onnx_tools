# -*- coding: utf-8 -*-

import numpy as np
import argparse

if __name__ == '__main__':
    # 参数解析
    usage = r'''
Example 1：npy data cmp
python accuracy_cmp.py --type=npy --left_data_path=./left_data.npy --right_data_path=./right_data.npy

Example 2：bin data cmp
python accuracy_cmp.py --type=bin --left_data_path=./left_data.npy --left_data_type=np.float32 --right_data_path=./right_data.npy --right_data_type=np.float32
    '''
    parser = argparse.ArgumentParser(usage='usage', description='numpy数据比对工具')
    parser.add_argument('--type', required=True, choices=['npy', 'bin'])
    parser.add_argument('--left_data_path', required=True)
    parser.add_argument('--left_data_type', required=False, help='当数据是bin文件时提供，e.g --left_data_type=np.float32')
    parser.add_argument('--right_data_path', required=True)
    parser.add_argument('--right_data_type', required=False, help='当数据是bin文件时提供，e.g --right_data_type=np.float32')
    opt = parser.parse_args()
    type = opt.type
    left_data_path = opt.left_data_path
    right_data_path = opt.right_data_path
    left_data_type = opt.left_data_type
    right_data_type = opt.right_data_type

    # 对比数据
    if type == 'npy':
        left_data = np.load(left_data_path)
        right_data = np.load(right_data_path)
        left_data = left_data.reshape(left_data.size)
        right_data = right_data.reshape(right_data.size)
    else:
        left_data = np.fromfile(left_data_path, dtype=eval(left_data_type))
        right_data = np.fromfile(right_data_path, dtype=eval(right_data_type))

    assert left_data.shape == right_data.shape

    max_error = 0
    count = 0
    shape = left_data.shape
    print('[INFO] Data len: {}\n'.format(shape[0]))
    print_align = '<' + str(len(str(shape[0])) + 2)
    for i in range(shape[0]):
        error = abs(left_data[i] - right_data[i])
        print('Idx:', format(i, print_align),
              'L:', format(left_data[i], '<25'),
              'R:', format(right_data[i], '<25'),
              'Err::', format(error, '<25'))
        if error < 1e-4:
            pass
        else:
            count += 1
            max_error = max(max_error, error)
    print('\n[INFO] Error ratio is: {}, max error is: {}'.format(float(count) / shape[0], max_error))

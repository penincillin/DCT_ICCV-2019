# file to store some often use functions

import os, sys, shutil
import os.path as osp
import multiprocessing as mp


def print_vars(_locals, variables, func):
    print('------------------------------------')
    var_list = list()
    for name, variable in list(_locals.items()):
        if variable in variables:
            idx = variables.index(variable)
            var_list.append( (name, variable, idx) )

    sorted_var_list = sorted(var_list, key=lambda a:a[2])
    for name, variable, _ in sorted_var_list:
        print("{}\t{}".format(name, func(variable)))
    print('------------------------------------')


def renew_dir(target_dir):
    if osp.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

def build_dir(target_dir):
    if not osp.exists(target_dir):
        os.makedirs(target_dir)


def remove_swp(in_dir):
    remove_files = list()
    for subdir, dirs, files in os.walk(in_dir):
        for file in files:
            if file.endswith('.swp'):
                full_path = osp.join(subdir,file)
                os.remove(full_path)

def remove_pyc(in_dir):
    remove_files = list()
    for subdir, dirs, files in os.walk(in_dir):
        for file in files:
            if file.endswith('.pyc'):
                full_path = osp.join(subdir,file)
                os.remove(full_path)


def easy_parallel(target_func, all_args):
    process_list = list()
    for args in all_args:
        p = mp.Process(target=target_func, args=args)
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()


def md5sum(file_path):
    import hashlib
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as in_f:
        hash_md5.update(in_f.read())
    return hash_md5.hexdigest()

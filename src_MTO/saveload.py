import json

def load_dict(filename):
    """load dict from json file"""
    with open(filename, "r") as json_file:
        dic = json.load(json_file)
    return dic

def save_dict(filename, dic):
    with open(filename, 'w') as json_file:
        json.dump(dic, json_file, ensure_ascii=False)


def savelist(filename, a):
    filename = open(filename, 'w')
    for value in a:
        filename.write(str(value)+'\n')
    filename.close()

def readlist(filename):
    f = open(filename, "r")
    a = f.read()
    f.close()
    return a


def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录 # 创建目录操作函数
        os.makedirs(path)

        print
        path + ' 创建成功'
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print
        path + ' 目录已存在'
        return False
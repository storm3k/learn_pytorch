# coding:utf8
import warnings
import torch as t


class DefaultConfig(object):
    # 数据集
    train_data_root = '/media/gaoshuai/项目/数据集/catVSdog/all/train/'  # 训练集存放路径
    test_data_root = '/media/gaoshuai/项目/数据集/catVSdog/all/test1'  # 测试集存放路径
    num_workers = 4  # how many workers for loading data

    # 模型
    model = 'SqueezeNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载

    # 超参数
    batch_size = 32  # batch size
    max_epoch = 10
    print_freq = 20  # print info every N batch
    # 学习率
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数

    # 可视化
    env = 'default'  # visdom 环境
    vis_port = 8097  # visdom 端口

    # debug
    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    # 其他
    use_gpu = True  # user GPU or not

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        # gpu or cpu
        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


opt = DefaultConfig()

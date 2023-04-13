# -*- coding: utf-8 -*-
class DefaultConfig(object):
    net_work = 'ResUnNet50'
    num_classes = 9
    num_epochs = 100
    batch_size = 64
    validation_step = 1
    root = "/raid/DTS/Dataset"
    train_file = "Datasets/train.csv"
    val_file = "Datasets/val.csv"
    test_file = "Datasets/test.csv"
    lr = 1e-4
    lr_mode = 'poly'
    momentum = 0.9
    weight_decay = 1e-4
    save_model_path = './Model_Saved'.format(net_work,lr)
    log_dirs = './Logs_Adam_0304'
    pretrained = False
    pretrained_model_path = None
    cuda = 0
    num_workers = 4
    use_gpu = True
    trained_model_path = ''
    predict_fold = 'predict_mask'

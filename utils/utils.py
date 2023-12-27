import logging
import os
import sys
import yaml

def path_to_repo(*args): # REPO/arg1/arg2
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), *args)

def path_to_data(*args): # REPO/data/arg1/arg2
    return path_to_repo("data", *args)

def path_to_experiment(*args): # REPO/experiments/arg1/arg2
    return path_to_repo("experiments", *args)

def path_to_config(*args): #REPO/configs/arg1/arg2
    return path_to_repo("configs", *args)


def create_logger(logdir):
    head = '%(asctime)-15s %(message)s'
    if logdir != '':
        log_file = os.path.join(logdir, 'log.txt')
        logging.basicConfig(filename=log_file, format=head)
        # output to console as well
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    else:
        logging.basicConfig(format=head)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger


def init_output_dirs(exp_name="default"):
    log_dir = path_to_experiment(exp_name)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    runs_dir = os.path.join(log_dir, "tensorboard")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
     
    return log_dir, ckpt_dir, runs_dir


def load_default_config():
    return load_config(path_to_config("default.yaml"))

def load_config(path, exp_name="default"):
    """
    Load the config file and make any dynamic edits.
    """
    with open(path, "rt") as reader:
        config = yaml.load(reader, Loader=yaml.Loader)

    if "OUTPUT" not in config:
        config["OUTPUT"] = {}
    config["OUTPUT"]["log_dir"], config["OUTPUT"]["ckpt_dir"], config["OUTPUT"]["runs_dir"] = init_output_dirs(exp_name=exp_name)

    with open(os.path.join(config["OUTPUT"]["ckpt_dir"], "config.yaml"), 'w') as f:
        yaml.dump(config, f)

    return config

class AverageMeter(object):
    """
    From https://github.com/mkocabas/VIBE/blob/master/lib/core/trainer.py
    Keeps track of a moving average.
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

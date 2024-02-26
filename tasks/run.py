import os

os.environ["OMP_NUM_THREADS"] = "1"

from utils.commons.hparams import hparams, set_hparams
import importlib


def run_task():
    assert hparams['task_cls'] != ''
    pkg = ".".join(hparams["task_cls"].split(".")[:-1]) # tasks.speech_editing.spec_denoiser
    cls_name = hparams["task_cls"].split(".")[-1] # SpeechDenoiserTask
    task_cls = getattr(importlib.import_module(pkg), cls_name) # getattr() 函数是 Python 中的一个内置函数，它用于获取对象的属性值
    task_cls.start()


if __name__ == '__main__':
    set_hparams()
    run_task()

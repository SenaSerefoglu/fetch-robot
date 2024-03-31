import os
from tensorboard import program

class Tensorboard:
    def __init__(self, log_path):
        self.log_path = log_path

    def launch(self):
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.log_path])
        url = tb.launch()
        print(f"Tensorflow listening on {url}")
        return url
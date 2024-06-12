import os

log_path = os.path.join('Reports/Params/batch_size')

from tensorboard import program

tracking_address = log_path # the path of your log file.

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    input("Press any key to exit...")
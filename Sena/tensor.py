import os

log_path = os.path.join('Logs', 'FetchPickAndPlace-v2', 'Tuning')
training_log_path = os.path.join(log_path, 'TQC_1')

from tensorboard import program

tracking_address = training_log_path # the path of your log file.

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
input("Press any key to exit...")
import argparse
from tensorboard import program

from util import log
from config import CHECKPOINTS_DIR

def launch_tensorboard(log_dir):
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir])
    url = tb.launch()
    log(f'TensorBoard launched at (Ctrl + C to quit): {url}')
    try:
        while True:
            pass
    except KeyboardInterrupt:
        log('Stopping TensorBoard...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch TensorBoard')
    parser.add_argument('--logdir', type=str, default=str(CHECKPOINTS_DIR), help='TensorBoard log directory')
    args = parser.parse_args()
    launch_tensorboard(args.log_dir)


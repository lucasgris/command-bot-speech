"""Train and save models script"""
from core.networks import BaseModel


if __name__ == '__main__':
    import json
    import argparse
    import importlib
    # Command line arguments:
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('model', help='Model path (Python file)')
    parser.add_argument('--params', help='Parameters JSON file')
    parser.add_argument('--checkpoints', help='')
    parser.add_argument('--best_checkpoint', help='')
    parser.add_argument('--callbacks', help='')
    parser.add_argument('--tensorboard', help='', action='store_true',
                        default=False)
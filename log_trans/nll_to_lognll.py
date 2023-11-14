########
# Usage:
# python nll_to_lognll.py -i <nll_file> -o <lognll_file>
######## 

import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str, default='', help='input file')
parser.add_argument('--output',
                    '-o',
                    type=str,
                    default='',
                    help='output file or dir')


def main():
    pass
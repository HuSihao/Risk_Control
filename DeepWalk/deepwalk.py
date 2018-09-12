# -*- coding: utf-8 -*-
import os
import sys
from skipgram import skipgram
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import logging
import tensorflow as tf
import math
import graphs
import random
import numpy as np
from six.moves import range
import psutil
from multiprocessing import cpu_count
from skipgram import generate_batch

def main():
    parser = ArgumentParser("deepwalk",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument("--debug", dest="debug", action='store_true', default=False,
                        help="drop a debugger if an exception is raised.")

    parser.add_argument('--format', default='adjlist',
                        help='File format of input file')

    parser.add_argument('--input', nargs='?', required=True,
                        help='Input graph file')

    parser.add_argument("-l", "--log", dest="log", default="INFO",
                        help="log verbosity level")

    parser.add_argument('--matfile-variable-name', default='network',
                        help='variable name of adjacency matrix inside a .mat file.')

    parser.add_argument('--max-memory-data-size', default=1000000000, type=int,
                        help='Size to start dumping walks to disk, instead of keeping them in memory.')

    parser.add_argument('--number-walks', default=10, type=int,
                        help='Number of random walks to start at each node')

    parser.add_argument('--output', required=True,
                        help='Output representation file')

    parser.add_argument('--representation-size', default=64, type=int,
                        help='Number of latent dimensions to learn for each node.')

    parser.add_argument('--seed', default=0, type=int,
                        help='Seed for random walk generator.')

    parser.add_argument('--undirected', default=True, type=bool,
                        help='Treat graph as undirected.')

    parser.add_argument('--vertex-freq-degree', default=False, action='store_true',
                        help='Use vertex degree to estimate the frequency of nodes '
                             'in the random walks. This option is faster than '
                             'calculating the vocabulary.')

    parser.add_argument('--walk-length', default=40, type=int,
                        help='Length of the random walk started at each node')

    parser.add_argument('--window-size', default=5, type=int,
                        help='Window size of skipgram model.')

    parser.add_argument('--workers', default=1, type=int,
                        help='Number of parallel processes.')

    args = parser.parse_args()
    process(args)



def process(args):
    if args.format == "adjlist":
        G = graphs.load_adjacencylist(args.input, undirected=args.undirected)
    elif args.format == "edgelist":
        G = graphs.load_edgelist(args.input, undirected=args.undirected)
    elif args.format == "mat":
        G = graphs.load_matfile(args.input, variable_name=args.matfile_variable_name, undirected=args.undirected)
    else:
        raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', 'mat'" % args.format)

    print("Number of nodes: {}".format(len(G.nodes())))

    num_walks = len(G.nodes()) * args.number_walks

    print("Number of walks: {}".format(num_walks))

    data_size = num_walks * args.walk_length

    print("Data size (walks*length): {}".format(data_size))

    if data_size < args.max_memory_data_size:
        print("Walking...")
        walks = graphs.build_deepwalk_corpus(G, num_paths=args.number_walks,
                                        path_length=args.walk_length, alpha=0, rand=random.Random(args.seed))
        print("Training...")
    vocabulary_size = len(G.nodes())
    num_steps = 100001
    batch_size = 128
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 2  # How many words to consider left and right.
    num_skips = 2  # How many times to reuse an input to generate a label.
    num_sampled = 1  # Number of negative examples to sample.

    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent. These 3 variables are used only for
    # displaying model accuracy, they don't affect calculation.
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    model = skipgram(walks,num_steps, batch_size,vocabulary_size,embedding_size,skip_window,num_skips,num_sampled,valid_size,valid_window)
    model.train()

if __name__ == '__main__':
    sys.exit(main())
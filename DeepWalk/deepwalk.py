#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import random
import numpy as np
from io import open
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import logging
import tensorflow as tf
import math
import graphs
import collections
import walks as serialized_walks
from gensim.models import Word2Vec
#from skipgram import skipgram

from six import text_type as unicode
from six import iteritems
from six.moves import range

import psutil
from multiprocessing import cpu_count

p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

#generate_batch(walks,batch_size, num_skips,skip_window)

def generate_batch(walks,batch_size, num_skips, skip_window):
    #walks must have been depricated.
    global data_index_i
    global data_index_j

    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size,1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin

    if data_index_j+span>len(walks[data_index_i]):
        data_index_j = 0
        data_index_i +=1
        if data_index_i > len(walks):
            data_index_i =  0

    buffer.extend(walks[data_index_i][data_index_j:data_index_j+span])
    data_index_j+=span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w!= skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_words in enumerate(words_to_use):
            batch[i * num_skips + j ] = buffer[skip_window]
            labels[i * num_skips +j, 0] = buffer[context_words]

        if data_index_j == len(walks[data_index_i]):
            data_index_j = 0
            data_index_i +=1
            if data_index_i

        else:
            buffer.append(walks[data_index_i][data_index_j])
            data_index_j+=1

        # Backtrack a little bit to avoid skipping words in the end of a batch
    return batch, labels


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
        # Step 4: Build and train a skip-gram model.

        batch_size = 128
        embedding_size = 128  # Dimension of the embedding vector.
        skip_window = 2  # How many words to consider left and right.
        num_skips = 2  # How many times to reuse an input to generate a label.
        num_sampled = 64  # Number of negative examples to sample.

        # We pick a random validation set to sample nearest neighbors. Here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent. These 3 variables are used only for
        # displaying model accuracy, they don't affect calculation.
        valid_size = 16  # Random set of words to evaluate similarity on.
        valid_window = 100  # Only pick dev samples in the head of the distribution.
        valid_examples = np.random.choice(valid_window, valid_size, replace=False)

        graph = tf.Graph()

        with graph.as_default():

            # Input data.
            with tf.name_scope('inputs'):
                train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
                valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

            # Ops and variables pinned to the CPU because of missing GPU implementation
            with tf.device('/cpu:0'):
                # Look up embeddings for inputs.
                with tf.name_scope('embeddings'):
                    embeddings = tf.Variable(
                        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

                # Construct the variables for the NCE loss
                with tf.name_scope('weights'):
                    nce_weights = tf.Variable(
                        tf.truncated_normal(
                            [vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
                with tf.name_scope('biases'):
                    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            # Explanation of the meaning of NCE loss:
            #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
            with tf.name_scope('loss'):
                loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights=nce_weights,
                        biases=nce_biases,
                        labels=train_labels,
                        inputs=embed,
                        num_sampled=num_sampled,
                        num_classes=vocabulary_size))

            # Add the loss value as a scalar to summary.
            tf.summary.scalar('loss', loss)

            # Construct the SGD optimizer using a learning rate of 1.0.
            with tf.name_scope('optimizer'):
                optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                                      valid_dataset)
            similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True)


            # Add variable initializer.
            init = tf.global_variables_initializer()

        # Step 5: Begin training.
        num_steps = 100001

        with tf.Session(graph=graph) as session:
            # Open a writer to write summaries.

            # We must initialize all variables before we use them.
            init.run()
            print('Initialized')

            average_loss = 0
            for step in xrange(num_steps):
                batch_inputs, batch_labels = generate_batch(walks,batch_size, num_skips,skip_window)
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                # Define metadata variable.
                run_metadata = tf.RunMetadata()

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
                # Feed metadata variable to session for visualizing the graph in TensorBoard.
                _, summary, loss_val = session.run(
                    [optimizer, loss],
                    feed_dict=feed_dict,
                    run_metadata=run_metadata)
                average_loss += loss_val

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0

        # Save the model for checkpoints.

    #model = Word2Vec(walks, size=args.representation_size, window=args.window_size, min_count=0, sg=1, hs=1, workers=args.workers)
    else:
        pass
  #   print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size, args.max_memory_data_size))
  #   print("Walking...")
  #
  #   walks_filebase = args.output + ".walks"
  #   walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=args.number_walks,
  #                                        path_length=args.walk_length, alpha=0, rand=random.Random(args.seed),
  #                                        num_workers=args.workers)
  #
  #   print("Counting vertex frequency...")
  #   if not args.vertex_freq_degree:
  #     vertex_counts = serialized_walks.count_textfiles(walk_files, args.workers)
  #   else:
  #     # use degree distribution for frequency in tree
  #     vertex_counts = G.degree(nodes=G.iterkeys())
  #
  #   print("Training...")
  #   walks_corpus = serialized_walks.WalksCorpus(walk_files)
  #   model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
  #                    size=args.representation_size,
  #                    window=args.window_size, min_count=0, trim_rule=None, workers=args.workers)
  #
  # model.wv.save_word2vec_format(args.output)


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
  numeric_level = getattr(logging, args.log.upper(), None)
  logging.basicConfig(format=LOGFORMAT)
  logger.setLevel(numeric_level)

  process(args)

if __name__ == "__main__":
  sys.exit(main())

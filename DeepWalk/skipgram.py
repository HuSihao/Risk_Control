# -*- coding: utf-8 -*-
import os
import sys
import random
import tensorflow as tf
import numpy as np
import math
import walks as serialized_walks
import collections
data_index_i = 0
data_index_j = 0

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
            if data_index_i == len(walks):
                data_index_i = 0

        else:
            buffer.append(walks[data_index_i][data_index_j])
            data_index_j+=1

        # Backtrack a little bit to avoid skipping words in the end of a batch
    return batch, labels



class skipgram(object):
    def __init__(self,walks,num_steps, batch_size,vocabulary_size,embedding_size,skip_window,num_skips,num_sampled,valid_size,valid_window):

        # bind params to class
        self.walks = walks
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.skip_window = skip_window
        self.num_skips = num_skips
        self.num_sampled = num_sampled
        self.valid_size = valid_size
        self.valid_window = valid_window
        self.vocabulary_size = vocabulary_size
        # We pick a random validation set to sample nearest neighbors. Here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent. These 3 variables are used only for
        # displaying model accuracy, they don't affect calculation.
        self.valid_size = 16  # Random set of words to evaluate similarity on.
        self.valid_window = 100  # Only pick dev samples in the head of the distribution.
        self.valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()

        with self.graph.as_default():
            # Input data.
            with tf.name_scope('inputs'):
                self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
                self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
                self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

            # Ops and variables pinned to the CPU because of missing GPU implementation
            with tf.device('/cpu:0'):
                # Look up embeddings for inputs.
                with tf.name_scope('embeddings'):
                    self.embeddings = tf.Variable(
                        tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
                    self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)

                # Construct the variables for the NCE loss
                with tf.name_scope('weights'):
                    self.nce_weights = tf.Variable(
                        tf.truncated_normal(
                            [self.vocabulary_size, self.embedding_size],
                            stddev=1.0 / math.sqrt(self.embedding_size)))
                with tf.name_scope('biases'):
                    self.nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            # Explanation of the meaning of NCE loss:
            #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights=self.nce_weights,
                        biases=self.nce_biases,
                        labels=self.train_labels,
                        inputs=self.embed,
                        num_sampled=self.num_sampled,
                        num_classes=self.vocabulary_size))

            # Add the loss value as a scalar to summary.
            #tf.summary.scalar('loss', loss)

            # Construct the SGD optimizer using a learning rate of 1.0.
            with tf.name_scope('optimizer'):
                self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            self.norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
            self.normalized_embeddings = self.embeddings / self.norm
            self.valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings,
                                                      self.valid_dataset)
            similarity = tf.matmul(
                self.valid_embeddings, self.normalized_embeddings, transpose_b=True)

            # Merge all summaries.
            self.merged = tf.summary.merge_all()

            # Add variable initializer.
            self.init = tf.global_variables_initializer()

            # Create a saver.
            self.saver = tf.train.Saver()


    def train(self):

        with tf.Session(graph=self.graph) as session:
            # Open a writer to write summaries.

            # We must initialize all variables before we use them.
            self.init.run()
            print('Initialized')

            average_loss = 0
            for step in xrange(self.num_steps):
                batch_inputs, batch_labels = generate_batch(self.walks,self.batch_size, self.num_skips,
                                                            self.skip_window)
                feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

                # Define metadata variable.
                run_metadata = tf.RunMetadata()

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
                # Feed metadata variable to session for visualizing the graph in TensorBoard.
                _, summary, loss_val = session.run(
                    [self.optimizer, self.merged, self.loss],
                    feed_dict=feed_dict,
                    run_metadata=run_metadata)
                average_loss += loss_val

                # Add returned summaries to writer in each step.
                #writer.add_summary(summary, step)
                # Add metadata to visualize the graph for the last run.
                # if step == (num_steps - 1):
                #     writer.add_run_metadata(run_metadata, 'step%d' % step)

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0

            # Save the model for checkpoints.



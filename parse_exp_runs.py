from __future__ import unicode_literals, print_function, division
import os
import subprocess
import logging
import itertools
from collections import defaultdict
import numpy as np
import argh
import scipy
import scipy.sparse.csgraph as csg
from conllu import parse_tree, parse_tree_incr, parse, parse_incr
from io import open
from collections import defaultdict
import json
import string
from joblib import Parallel, delayed
import multiprocessing
import networkx as nx
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import math
from io import open
import unicodedata
import string
import re
import random
import json
import utils.parse_exp_utils as util
import pdb
import glob
import utils.distortions as dis
import warnings
from torch.optim import Optimizer
from torch.optim.optimizer import Optimizer, required

MAX_LENGTH=50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0

    def addSentence(self, sentence):
        for token in sentence:
            self.addWord(token['form'])

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, max_length=MAX_LENGTH):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)


    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        attention_scores = self.attn(torch.cat((embedded[0], hidden.unsqueeze(0)), 1))
        attn_weights = F.softmax(attention_scores, dim=0)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        
        return output


def poincare_grad(p, d_p):
    """
    Calculates Riemannian grad from Euclidean grad.
    Args:
        p (Tensor): Current point in the ball
        d_p (Tensor): Euclidean gradient at p
    """
    p_sqnorm = torch.sum(p.data ** 2, dim=-1, keepdim=True)
    d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)
    return d_p

def _correct(x, eps=1e-10):
    current_norms = torch.norm(x,2,x.dim() - 1)
    mask_idx      = current_norms < 1./(1+eps)
    modified      = 1./((1+eps)*current_norms)
    modified[mask_idx] = 1.0
    return modified.unsqueeze(-1)

def euclidean_grad(p, d_p):
    return d_p

def retraction(p, d_p, lr):
    # Gradient clipping.
    d_p.clamp_(min=-10000, max=10000)
    p.data.add_(-lr, d_p)
    #project back to the manifold.
    p.data = p.data * _correct(p.data)


class RiemannianSGD(Optimizer):
    r"""Riemannian stochastic gradient descent.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rgrad (Function): Function to compute the Riemannian gradient from
            an Euclidean gradient
        retraction (Function): Function to update the parameters via a
            retraction of the Riemannian gradient
        lr (float): learning rate
    """

    def __init__(self, params, lr=required, rgrad=required, retraction=required):
        defaults = dict(lr=lr, rgrad=rgrad, retraction=retraction)
        super(RiemannianSGD, self).__init__(params, defaults)

    def step(self, lr=None):
        """Performs a single optimization step.
        Arguments:
            lr (float, optional): learning rate for the current update.
        """
        loss = None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if lr is None:
                    lr = group['lr']
                d_p = group['rgrad'](p, d_p)
                group['retraction'](p, d_p, lr)

        return loss



def preprocess(create_edges=False):
    """Reads in the conllu training file, trims the sentences (note it's not preprocessed),
    creates the vocab from read-in sentences, creates the edgelist for GT parse trees.
    
    Returns filtered sentences and input vocab.
    """
    sentences = []
    data_file = open("data/en_ewt-ud-train.conllu", "r", encoding="utf-8")
    for sentence in parse_incr(data_file):
        sentences.append(sentence)
        
    MIN_LENGTH = 10
    MAX_LENGTH = 50

    def check_length(sentence):
        return len(sentence) < MAX_LENGTH and len(sentence) > MIN_LENGTH 

    def filterSentences(sentences):
        return [sent for sent in sentences if check_length(sent)]

    input_vocab = Vocab("ewt_train_trimmed")
    filtered_sentences = filterSentences(sentences)
    # print("Length of filtered sentences", len(filtered_sentences))

    # sentences_text = []
    for sent in filtered_sentences:
        input_vocab.addSentence(sent)
        # sentences_text.append(sent.metadata['text'])

    if create_edges:
        os.makedirs("./data/edges/parsing_train/", exist_ok=True)
        for idx in range(0, len(filtered_sentences)):
            curr_tree = filtered_sentences[idx].to_tree()
            G_curr = nx.Graph()
            G_curr = util.unroll(curr_tree, G_curr)
            G = nx.relabel_nodes(G_curr, lambda x: x-1)
            nx.write_edgelist(G, "./data/edges/parsing_train/"+str(idx)+".edges", data=False)

    return filtered_sentences, input_vocab



def run_net(run_name, edge_folder, test_folder, device):
    logging.info("Starting parsing net:")

    filtered_sentences, input_vocab = preprocess(create_edges=False)
    hidden_size = 100
    encoder = EncoderLSTM(input_vocab.n_words, hidden_size).to(device)
    attention = Attention(input_vocab.n_words, hidden_size).to(device)
    
    input_size = 100
    output_size = 10
    mapping = nn.Sequential(
        nn.Linear(input_size, 1000).to(device),
        nn.ReLU().to(device),
        nn.Linear(1000, 100).to(device),
        nn.ReLU().to(device),
        nn.Linear(100, output_size).to(device),
        nn.ReLU().to(device))
    scale = nn.Parameter(torch.cuda.FloatTensor([1.0]), requires_grad=True)
    return trainParseIters(mapping, scale, encoder, attention, filtered_sentences, input_vocab, edge_folder, test_folder, run_name)



def trainParse(input_tensor, target_matrix, n, ground_truth, encoder, encoder_optimizer, attention, attention_optimizer, mapping, mapping_optimizer, scale, scaling_optimizer, sampled_rows, max_length=50):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    attention_optimizer.zero_grad()
    mapping_optimizer.zero_grad()
    scaling_optimizer.zero_grad()
 
    input_length = input_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    encoder_hiddens = torch.zeros(max_length, encoder.hidden_size, device=device)
    euclidean_embeddings = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
        encoder_hiddens[ei] = encoder_hidden[0, 0]

    for idx in range(input_length):
        output = attention(input_tensor[idx], encoder_hiddens[idx], encoder_outputs)
        embedding = output[0]
        norm = embedding.norm(p=2, dim=1, keepdim=True)
        max_norm = torch.max(norm)+1e-10
        normalized_emb = embedding.div(max_norm.expand_as(embedding))
        euclidean_embeddings[idx] = normalized_emb
        
    final_embeddings = mapping(euclidean_embeddings)
    dummy_euclidean = euclidean_embeddings.clone()

    dist_recovered = util.distance_matrix_hyperbolic_parsing(input_length, final_embeddings, sampled_rows, scale) 
    dummy = dist_recovered.clone()
    edge_acc = util.compare_mst(ground_truth, dummy.detach().cpu().numpy())
    loss += util.distortion(target_matrix, dist_recovered, n, sampled_rows)
    loss.backward(retain_graph=True)

    euc_dist_recovered = util.distance_matrix_euclidean_parsing(input_length, dummy_euclidean, sampled_rows)
    euc_edge_acc = util.compare_mst(ground_truth, euc_dist_recovered.detach().cpu().numpy())

    encoder_optimizer.step()
    attention_optimizer.step()
    mapping_optimizer.step()
    scaling_optimizer.step()

    return loss.item(), edge_acc, euc_edge_acc


def full_stats_pass_point(input_matrix, target_matrix, ground_truth, n, encoder, attention, mapping, scale, verbose=False, max_length=MAX_LENGTH):
    st = time.time()
    encoder_hidden = encoder.initHidden()
    input_length = input_matrix.size(0)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    encoder_hiddens = torch.zeros(max_length, encoder.hidden_size, device=device)
    final_embeddings = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_matrix[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
        encoder_hiddens[ei] = encoder_hidden[0, 0]

    for idx in range(input_length):
        output = attention(input_matrix[idx], encoder_hiddens[idx], encoder_outputs)
        embedding = output[0]
        norm = embedding.norm(p=2, dim=1, keepdim=True)
        max_norm = torch.max(norm)+1e-10
        normalized_emb = embedding.div(max_norm.expand_as(embedding))
        final_embeddings[idx] = normalized_emb
    
    final_embeddings = mapping(final_embeddings)
    print("final embeddings shape", final_embeddings.shape)
    sampled_rows = list(range(input_matrix.shape[0]))
    dist_recovered = util.distance_matrix_hyperbolic_parsing(input_length, final_embeddings, sampled_rows, scale)
    print("target matrix", target_matrix)
    print("dist recovered matrix", dist_recovered)
    dis = util.distortion(target_matrix, dist_recovered, n, sampled_rows)
    dummy = dist_recovered.clone()
    edge_acc = util.compare_mst(ground_truth, dummy.detach().cpu().numpy())

    return dis, edge_acc


def trainParseIters(mapping, scale, encoder, attention, filtered_sentences, input_vocab, edge_folder, test_folder, run_name, n_epochs=2000, print_every=5, learning_rate=0.2):
    start = time.time()
    print_loss_total = 0
    n_iters = len(filtered_sentences)
    # subsample_row_num = 10
    # resample_every = 5
    full_stats_every = 100  

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    attention_optimizer = optim.SGD(attention.parameters(), lr=learning_rate)
    mapping_optimizer = RiemannianSGD(mapping.parameters(), lr=learning_rate, rgrad=poincare_grad, retraction=retraction)
    scaling_optimizer = optim.SGD([scale], lr=learning_rate)
    training_pairs = [util.pairfromidx_parsing(idx, input_vocab, filtered_sentences, edge_folder) for idx in range(n_iters)]

    logging.info(f"Started the run with lr {learning_rate}")
    epoch_to_metrics = []

    for n in range(n_epochs):
        iter=1
        full_distortion = 0
        full_edge_acc = 0
        euc_full_edge_acc = 0
        if n!=0 and n%10==0 and n<200:
            mapping_optimizer = RiemannianSGD(mapping.parameters(), lr=learning_rate*0.5, rgrad=poincare_grad, retraction=retraction)
            encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate*0.5)
            attention_optimizer = optim.SGD(attention.parameters(), lr=learning_rate*0.5)
        for iter in range(1, n_iters + 1):     
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_matrix = training_pair[1]
            size = training_pair[2]
            G = training_pair[3]

            subsampled_rows = list(range(input_tensor.shape[0]))
            # if (iter-1) % resample_every == 0:
            #     subsampled_rows = np.random.permutation(G.order())[:subsample_row_num]   

            loss, edge_acc, euc_edge_acc = trainParse(input_tensor, target_matrix, size, G, encoder, encoder_optimizer, attention, attention_optimizer, mapping, mapping_optimizer, scale, scaling_optimizer, subsampled_rows)
            print_loss_total += loss
            full_distortion += loss
            full_edge_acc  += edge_acc
            euc_full_edge_acc += euc_edge_acc

            if True: 
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                logging.info('%s (%d %d%%) %.4f' % (util.timeSince(start, iter / n_iters),
                                            n, iter / n_iters * 100, print_loss_avg))
            print(iter)
            iter+=1

        if n % full_stats_every == 0 or n == n_epochs-1:
            full_distortion /= n_iters
            full_edge_acc /= n_iters
            euc_full_edge_acc /= n_iters
            logging.info(f"\nFull distortion = , {full_distortion}, Edge Acc. = , {full_edge_acc}, Euclidean Edge Acc = {euc_full_edge_acc}\n")
            logging.info(f"Scale = {scale.data}")
            epoch_to_metrics.append((full_distortion, full_edge_acc, euc_full_edge_acc))
            os.makedirs(f"{run_name}/cptsparsing/", exist_ok=True)
            torch.save({
            'epoch': n,
            'mapping_state_dict': mapping.state_dict(),
            'encoder_state_dict': encoder.state_dict(),
            'attention_state_dict': attention.state_dict(),
            'scale':scale,
            'map_opt_dict': mapping_optimizer.state_dict(),
            'scale_opt_dict': scaling_optimizer.state_dict(),
            'encoder_opt_dict': encoder_optimizer.state_dict(),
            'attention_opt_dict': attention_optimizer.state_dict(),
            'distortion': full_distortion,
            'edge accuracy': edge_acc
            }, f"{run_name}/cptsparsing/{n}.cpt")

    #Save the metrics in the end for ease of use.
    for i in range(len(epoch_to_metrics)):
        logging.info(f"Epoch: {epoch}, Distortion: {full_distortion}, Edge Accuracy: {full_edge_acc}, Euc Edge Acc: {euc_full_edge_acc}")
        # # do a full pass at the end:
        # if n % full_stats_every == 0 or n == n_epochs-1:
        #     full_distortion = 0

        #     os.makedirs(f"{run_name}/cptsparsing/", exist_ok=True)
        #     for i in range(n_iters):     
        #         training_pair = training_pairs[i]
        #         input_matrix = training_pair[0]
        #         target_matrix = training_pair[1]
        #         size = training_pair[2]
        #         G = training_pair[3]

        #         [fd, ea] = full_stats_pass_point(input_matrix, target_matrix, G, n, encoder, attention, mapping, scale)
        #         full_distortion += fd
        #         edge_acc        += ea

            # full_distortion /= n_iters
            # edge_acc /= n_iters
            # torch.save({
            # 'epoch': n,
            # 'mapping_state_dict': mapping.state_dict(),
            # 'encoder_state_dict': encoder.state_dict(),
            # 'attention_state_dict': attention.state_dict(),
            # 'scale':scale,
            # 'map_opt_dict': mapping_optimizer.state_dict(),
            # 'scale_opt_dict': scaling_optimizer.state_dict(),
            # 'encoder_opt_dict': encoder_optimizer.state_dict(),
            # 'attention_opt_dict': attention_optimizer.state_dict(),
            # 'distortion': full_distortion,
            # 'edge accuracy': edge_acc
            # }, f"{run_name}/cptsparsing/{n}.cpt")

            # logging.info(f"Scale = {scale.data}")
            # print("Scale = ", scale.data)
            # logging.info(f"Full distortion = {full_distortion}, Edge Acc = {edge_acc}")
            # print("\nFull distortion = ", full_distortion, " Edge Acc. = ", edge_acc, "\n")


@argh.arg("run_name", help="Directory to store the run")

def run(run_name, edge_folder="./data/edges/parsing_train/", test_folder="./random_trees/test/"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(run_name, exist_ok=True)
    #Set up logging.
    formatter = logging.Formatter('%(asctime)s %(message)s')
    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        datefmt='%FT%T',)
    logging.info(f"Logging to {run_name}")
    log = logging.getLogger()
    fh  = logging.FileHandler(run_name+"/logparsingfull")
    fh.setFormatter(formatter)
    log.addHandler(fh)
    logging.info(device)
    print("Running Net")
    run_net(run_name, edge_folder, test_folder, device)
    
if __name__ == '__main__':
    _parser = argh.ArghParser()
    _parser.set_default_command(run)
    _parser.dispatch()

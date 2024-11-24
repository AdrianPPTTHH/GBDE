import argparse
import os
import numpy as np
import dataloader
from train_classifier import Model
import criteria
import random
from tqdm import tqdm
import torch.nn.functional as F
import tensorflow as tf
import tensorflow_hub as hub
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset

from TRAIN.tokenization import BertTokenizer
from TRAIN.modeling import BertForSequenceClassification, BertConfig

from Attacker.replace import get_replace_word_dict,BallDict

from transformers import AutoModelForMaskedLM,GPT2TokenizerFast,GPT2LMHeadModel
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import XLNetForSequenceClassification, XLNetTokenizer

from torch.nn import CrossEntropyLoss
from torch.nn import DataParallel

import random
import numpy as np
import torch

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 
tf.random.set_seed(random_seed)
import os


class GPT2LM:
      def __init__(self, gpu_ids=[], max_length = 128 , model_resolution = './GPT2'):
            self.tokenizer = GPT2TokenizerFast.from_pretrained(model_resolution)
            self.tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})

            self.lm = GPT2LMHeadModel.from_pretrained(model_resolution)
            # self.lm = torch.load('gpt2-large.pkl')
            self.cuda = gpu_ids
            self.max_length = max_length

            if len(self.cuda) > 0 :
                self.lm = DataParallel(self.lm, device_ids=gpu_ids)
                self.lm.cuda()

      def calculate_delta_PPL(self, ori_sentences, adv_sentences):

            sentences1 = [s.lower() for s in ori_sentences]
            sentences2 = [s.lower() for s in adv_sentences]

            # if(len(sentences1[0]) > 1024):
            #     max_length = 1024
            # else:
            #     max_length = len(sentences1[0])
            
            max_length = self.max_length

            ipt1 = self.tokenizer(sentences1, return_tensors="pt", padding=True, truncation=True, verbose=False, max_length=max_length)
            ipt2 = self.tokenizer(sentences2, return_tensors="pt", padding=True, truncation=True, verbose=False, max_length=max_length)
        
            if len(self.cuda) > 0:
                  for k in ipt1.keys():
                        ipt1[k] = ipt1[k].cuda()
                  for k in ipt2.keys():
                        ipt2[k] = ipt2[k].cuda()

            logits1 = self.lm(**ipt1).logits
            perplexities1 = torch.exp(self.calculate_loss(logits1, ipt1.input_ids)).cpu().detach().numpy()
            # torch.cuda.empty_cache()

            logits2 = self.lm(**ipt2).logits
            perplexities2 = torch.exp(self.calculate_loss(logits2, ipt2.input_ids)).cpu().detach().numpy()
            # torch.cuda.empty_cache()

            return perplexities2 - perplexities1

      def calculate_loss(self, logits, labels):            
            # Assuming the batch size is the first dimension
            batch_size, max_len, vocab_size = logits.size()
            
            # Initialize an empty list to store individual losses
            losses = []
            
            # Loop over each sentence in the batch
            for i in range(batch_size):
                  # Extract logits and labels for the current sentence
                  sentence_logits = logits[i, :max_len-1, :]
                  sentence_labels = labels[i, 1:max_len].view(-1)
                  
                  # Calculate loss for the current sentence
                  loss_fct = CrossEntropyLoss()
                  sentence_loss = loss_fct(sentence_logits, sentence_labels)
                  
                  # Append the loss to the list
                  losses.append(sentence_loss)
            
            # Convert the list of losses to a tensor
            losses = torch.stack(losses)
                        
            return losses

      def calculate_PPL_loss(self, sentences):            
            sentences = [s.lower() for s in sentences]

            ipt = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, verbose=False, max_length=self.max_length)

            if len(self.cuda) > 0:
                  for k in ipt.keys():
                        ipt[k] = ipt[k].cuda()


            logits = self.lm(**ipt).logits

            return self.calculate_loss(logits, ipt.input_ids).cpu().detach().numpy()


class USE(object):
    def __init__(self, cuda_devices):
        super(USE, self).__init__()

        gpus = tf.config.experimental.list_physical_devices('GPU')

        if gpus:
        # Restrict TensorFlow to only use the victim model's GPU
            try:
                tf.config.experimental.set_visible_devices([gpus[id] for id in cuda_devices], 'GPU')
                for id in cuda_devices:
                    tf.config.experimental.set_memory_growth(gpus[id], True)

            except RuntimeError as e:
                # Visible devices must be set at program startup
                print(e)
        # module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        module_url = "./Tensorflow"
        self.embed = hub.load(module_url)
    
    def semantic_sim(self, sentences1, sentences2):
        sentences1_lower = [sentence.lower() for sentence in sentences1]
        sentences2_lower = [sentence.lower() for sentence in sentences2]

        # Get embeddings for all sentences in each batch
        embeddings1 = self.embed(sentences1_lower)
        embeddings2 = self.embed(sentences2_lower)

        similarity_scores = tf.reduce_sum(tf.multiply(embeddings1, embeddings2), axis=1).numpy()

        return similarity_scores

    def __call__(self, sentence1, sentence2):
        sentence1, sentence2 = sentence1.lower(), sentence2.lower()
        embeddings = self.embed([sentence1, sentence2])

        vector1 = tf.reshape(embeddings[0], [512, 1])
        vector2 = tf.reshape(embeddings[1], [512, 1])

        return tf.matmul(vector1, vector2, transpose_a=True).numpy()[0][0]


def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
    sim_words, sim_values = [], []
    for idx, src_word in enumerate(src_words):
        sim_value = sim_mat[src_word][sim_order[idx]]
        mask = sim_value >= threshold
        sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
        sim_word = [idx2word[id] for id in sim_word]
        sim_words.append(sim_word)
        sim_values.append(sim_value)
    return sim_words, sim_values


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class NLI_infer_XLNET(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        super(NLI_infer_XLNET, self).__init__()
        # self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_dir).cuda()
        self.model = XLNetForSequenceClassification.from_pretrained(pretrained_dir).cuda()

        self.pretrained_dir = pretrained_dir

        # construct dataset loader
        self.dataset = NLIDataset_XLNET(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data, batch_size=32):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)

        probs_all = []

        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()

            with torch.no_grad():
                
                # logits = self.model(input_ids).logits

                if "mnli" in self.pretrained_dir.lower():
                    logits = self.model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids).logits
                else:
                    logits = self.model(input_ids).logits

                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


class NLIDataset_XLNET(Dataset):
    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        self.tokenizer = XLNetTokenizer.from_pretrained(pretrained_dir)
        # self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
        
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, text_a) in enumerate(examples):
            tokens_a = tokenizer.tokenize(' '.join(text_a))

            # # Account for [CLS] and [SEP] with "- 2"
            # if len(tokens_a) > max_seq_length - 2:
            #     tokens_a = tokens_a[:(max_seq_length - 2)]

            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]            

            tokens = ["<s>"] + tokens_a + ["</s>"]

            if "\t" in tokens:
                index = tokens.index("\t")
                tokens[index] = "</s>"


            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(input_ids)
            padding = [0] * (max_seq_length - len(input_ids))

            input_ids += padding
            input_mask += padding
            segment_ids += padding
            
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            
            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
            
        return features

    def transform_text(self, data, batch_size=32):
        # transform data into seq of embeddings
        eval_features = self.convert_examples_to_features(data,
                                                          self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
        
        return eval_dataloader

class NLI_infer_ROBERTA(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        super(NLI_infer_ROBERTA, self).__init__()
        # self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_dir).cuda()
        self.model = RobertaForSequenceClassification.from_pretrained(pretrained_dir).cuda()

        self.pretrained_dir = pretrained_dir

        # construct dataset loader
        self.dataset = NLIDataset_ROBERTA(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data, batch_size=32):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)

        probs_all = []

        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()

            with torch.no_grad():
                
                if "mnli" in self.pretrained_dir.lower():
                    logits = self.model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids).logits
                else:
                    logits = self.model(input_ids).logits

                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)

class NLIDataset_ROBERTA(Dataset):
    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        # self.tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_dir, do_lower_case=True)

        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, text_a) in enumerate(examples):
            tokens_a = tokenizer.tokenize(' '.join(text_a))

            # # Account for [CLS] and [SEP] with "- 2"
            # if len(tokens_a) > max_seq_length - 2:
            #     tokens_a = tokens_a[:(max_seq_length - 2)]

            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]            

            tokens = ["<s>"] + tokens_a + ["</s>"]

            if "\t" in tokens:
                index = tokens.index("\t")
                tokens[index] = "</s>"


            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(input_ids)
            padding = [0] * (max_seq_length - len(input_ids))

            input_ids += padding
            input_mask += padding
            segment_ids += padding
            
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            
            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
            
        return features

    def transform_text(self, data, batch_size=32):
        # transform data into seq of embeddings
        eval_features = self.convert_examples_to_features(data,
                                                          self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
        
        return eval_dataloader

class NLI_infer_BERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 nclasses,
                 max_seq_length=128,
                 batch_size=32):
        super(NLI_infer_BERT, self).__init__()

        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses).cuda()

        # self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_dir).cuda()
        
        # construct dataset loader
        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data, batch_size=32):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)

        probs_all = []
        #         for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)

class NLIDataset_BERT(Dataset):

    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=True)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, text_a) in enumerate(examples):
            tokens_a = tokenizer.tokenize(' '.join(text_a))

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
        return features

    def convert_examples_to_features1(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""
        
        features = []
        for (ex_index, example) in enumerate(examples): 
            if '|<SEP>|' in example:
                text_a, text_b = example[:example.index('|<SEP>|')], example[example.index('|<SEP>|')+1:]
            else:
                text_a, text_b = example, ""

            tokens_a = tokenizer.tokenize(' '.join(text_a))
            tokens_b = tokenizer.tokenize(' '.join(text_b))

            # Account for [CLS], [SEP], [SEP] with "- 3"
            if len(tokens_a) + len(tokens_b) > max_seq_length - 3:
                tokens_a = tokens_a[:max_seq_length - 3 - len(tokens_b)]

            # Combine the tokens and add [CLS], [SEP], [SEP]
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
            segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
        return features
    def transform_text(self, data, batch_size=32):
        # transform data into seq of embeddings
        eval_features = self.convert_examples_to_features1(data,
                                                          self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        return eval_dataloader

# Refer to the TextFooler attack: https://github.com/jind11/TextFooler
def Greedy(text_ls, true_label, predictor, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
           batch_size=32):
    # first check the prediction of the original text
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        print(true_label, orig_label)
        print(text_ls)
        return '', 0, orig_label, orig_label, 0, 0
    else:
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(text_ls)

        # get importance score
        leave_1_texts = [text_ls[:ii] + ['<oov>'] + text_ls[min(ii + 1, len_text):] for ii in range(len_text)]
        leave_1_probs = predictor(leave_1_texts, batch_size=batch_size)
        num_queries += len(leave_1_texts)
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
        import_scores = (orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (
                    leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0,
                                                                      leave_1_probs_argmax))).data.cpu().numpy()

        # get words to perturb ranked by importance scorefor word in words_perturb
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            try:
                if score > import_score_threshold and text_ls[idx] not in stop_words_set:
                    words_perturb.append((idx, text_ls[idx]))
            except:
                print(idx, len(text_ls), import_scores.shape, text_ls, len(leave_1_texts))

        # find synonyms
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)
        synonyms_all = []

        for idx, word in words_perturb:
            if word in word2idx:
                if word == "|<SEP>|":
                    continue
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = text_ls[:]
        text_cache = text_prime[:]
        num_changed = 0
        sim_score = 0

        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            new_probs = predictor(new_texts, batch_size=batch_size)

            # compute semantic similarity 
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            semantic_sims = \
            sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                       list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

            num_queries += len(new_texts)
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= sim_score_threshold)
            # prevent incompatible pos
            synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask
            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_changed += 1

                # sim_score = semantic_sims[(new_probs_mask * semantic_sims).argmax()]

                sim_score =  sim_predictor.semantic_sim([' '.join(text_ls)], [' '.join(text_prime)])[0]
                break
            else:
                new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                        (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
                    num_changed += 1
            text_cache = text_prime[:]
        return ' '.join(text_prime), num_changed, orig_label, torch.argmax(predictor([text_prime])), num_queries , sim_score

def eval(text_ls, true_label, predictor):
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0, 0
    else:
        text_prime = text_ls[:]
        num_changed = 0
        sim_score = 0
        num_queries = 1

        return ' '.join(text_prime), num_changed, orig_label, torch.argmax(predictor([text_prime])), num_queries, sim_score

def GBDE(text_ls, true_label, predictor, PPL_metric, stop_words_set, word2idx, idx2word, replace_dict, sim_predictor=None,
           CR_=-1., max_generations_= -1, batch_size=32, Mutation_algorithm = 1, beta = 10):

    population_size = batch_size 

    if max_generations_ != -1:
        max_generations = max_generations_ 
    else:
        max_generations = 200

    F = 1 

    if CR_ != -1.:
        CR = CR_
    else:
        CR = 0.9

    pos_ls = criteria.get_pos(text_ls)

    # first check the prediction of the original text
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()

    if true_label != orig_label:
        print("predict wrong:",true_label, orig_label)
        return '', 0, orig_label, orig_label, 0 , 0 , 0
    else:                                 
        # get the pos and verb tense info
        pos_ls = criteria.get_pos(text_ls)

        num_queries = 1
        num_changed = 0

        words_perturb = []
        for idx, word in enumerate(text_ls):
            try:
                if text_ls[idx] not in stop_words_set:
                    words_perturb.append((idx, text_ls[idx]))
            except:
                print(idx, len(text_ls))

        # find synonyms
        synonyms_all = []

        for idx, word in words_perturb:
            if word in word2idx:
                word_idx = word2idx[word]
                if word_idx in replace_dict:
                    synonyms = [idx2word[idx] for idx in replace_dict[word_idx]]
                    if synonyms:
                        synonyms_all.append((idx, synonyms))
                else:
                    print("find replace word:{} failed.".format(word))
            else:
                continue

        synonyms_dict = {idx: synonyms for idx, synonyms in synonyms_all}

        for idx, synonyms in synonyms_dict.copy().items():
            text_cache = text_ls[:]
            for synonym in synonyms:
                text_cache = text_cache[:idx] + [synonym] + text_cache[idx + 1:]
                synonyms_pos_ls = [ criteria.get_pos(text_cache[max(idx - 14, 0):idx + 15])[min(14, idx)]
                                        if len(text_cache) > 20 else criteria.get_pos(text_cache)[idx] ]
                
                if not criteria.pos_filter(pos_ls[idx], synonyms_pos_ls)[0]:
                    synonyms_dict[idx].remove(synonym)

            if(len(synonyms_dict[idx]) == 0):
                del synonyms_dict[idx]

        num_replacements = len(synonyms_dict)

        # DE objective function
        # x: The nth word is replaced with the nth synonym
        # x[0], x[2], x[4]... represent the nth index; x[1] represents the xth synonym at the position x[0]; x[3], x[5]... follow the same pattern.
        def objective_function_batch(x_batch, pre_sim, generation):
            nonlocal num_queries

            k = orig_prob.cpu().detach().numpy()

            new_texts = []

            change_rates = np.zeros((len(x_batch),))

            for x in x_batch:
                text_cache = text_ls[:]

                for i in range(num_replacements):
                    word_index = int(x[i * 2])
                    synonym_index = int(x[i * 2 + 1])

                    synonyms = synonyms_dict.get(word_index, [])
                    
                    synonym = synonyms[synonym_index] if (0 <= synonym_index < len(synonyms)) else text_ls[word_index] 

                    text_cache = text_cache[:word_index] + [synonym] + text_cache[word_index + 1:]

                new_texts.append(text_cache)
                num_changed = sum(1 for word1, word2 in zip(text_cache, text_ls) if word1 != word2)
                np.append(change_rates , (num_changed / len(text_ls)) )

            semantic_scores = sim_predictor.semantic_sim([' '.join(text_ls)] * len(new_texts) , 
                                    list(map(lambda x: ' '.join(x), new_texts)) )

            ppl_loss = PPL_metric.calculate_PPL_loss( list(map(lambda x: ' '.join(x), new_texts)) )

            new_probs = predictor(new_texts, batch_size=batch_size)
            num_queries += 1

            #to numpy
            true_label_probs = new_probs[:, orig_label].cpu().detach().numpy()
            label_jud_matrix = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()

            total_diffs = true_label_probs * 1/semantic_scores + np.log(ppl_loss) - ( label_jud_matrix * (1 / np.log(ppl_loss) + semantic_scores) \
                                                                                    + ~label_jud_matrix * (k - true_label_probs)  ) * beta + change_rates * 2

            success = False
            shuffle = False

            if np.any(label_jud_matrix) & np.any(semantic_scores > 0.95) & label_jud_matrix[np.argmax(semantic_scores)]:
                success = True
            #Local optimum
            elif (~np.any(label_jud_matrix)) & (np.all(np.round(semantic_scores, 4) == np.round(pre_sim, 4))):
                shuffle = True

            total_diffs = total_diffs + (semantic_scores < 0.2) * 1
            
            return total_diffs, success, semantic_scores, ppl_loss, shuffle


        def differential_evolution(objective_function_batch, bounds, population_size, max_generations, F, CR):

            num_parameters = len(bounds)
            population = np.empty((population_size, num_parameters), dtype=int)

            # initialize
            for i in range(population_size):
                candidate = [random.randint(bounds[i][0], bounds[i][1]) for i in range(num_parameters)]
                population[i] = np.array(candidate)

            trial_pre_sims = np.zeros(batch_size)
            target_pre_sims = np.zeros(batch_size)

            ppl_values = []
            sim_values = []
            iteration_values = []

            target_values = np.array(objective_function_batch(population, target_pre_sims, 0)[0])
            n = 1
            
            for generation in range(max_generations):

                target_vectors = population.copy()
                mutant_vectors = np.empty((population_size, num_parameters), dtype=int)

                values = np.random.permutation(population_size)
                np.random.shuffle(values)

                a = values
                b = (values + n) % population_size
                c = (values + n + 1) % population_size
                d = (values + n + 2) % population_size

                a, b, c, d = population[a], population[b], population[c], population[d]

                if Mutation_algorithm == 1:
                    # Mutation Method 1: DE/rand/1 (The original DE algorithm)
                    # This mutation method is suitable for general global optimization problems, but the scaling factor may need adjustment for complex problems.
                    mutant_vectors = np.array(a + F * (b - c), dtype=int)

                elif Mutation_algorithm == 2:
                    # Mutation Method 2: DE/rand/1 + parameters
                    # Prevents solution vectors from being all zeros during the ABC iteration process.                    
                    mutant_vectors = np.where(np.random.uniform(0, 1, a.shape) < CR / 2, a + F * (b - c) + 1, a + F * (b - c))
                elif Mutation_algorithm == 3:
                    # Mutation Method 3: DE/current-to-best/1
                    # This mutation method is often used to accelerate convergence, especially when the objective function has a clear global optimum.
                    x_best = population[np.argmin(target_values)]
                    mutant_vectors = a + F * (b - a) + F * (c - x_best)

                elif Mutation_algorithm == 4:
                    # Mutation Method 4: DE/current-to-rand/1
                    # This mutation method is suitable for exploring diversity in the problem space, as it introduces more randomness.
                    mutant_vectors = a + F * (b - a) + F * (c - d)

                bounds_array = np.array(bounds)
                if mutant_vectors.size == 0 or bounds_array.size == 0:
                    break
                mutant_vectors = np.clip(mutant_vectors, bounds_array[:, 0], bounds_array[:, 1])

                mask = np.random.random((population_size, num_parameters)) < CR
                indices = np.arange(num_parameters)
                trial_vectors = np.where(mask | (indices == np.random.randint(0, num_parameters, size=population_size)[:, None]), mutant_vectors, target_vectors)

                trial_total_diffs, trial_success, trial_pre_sims, ppl_loss, trial_shuffle = objective_function_batch(trial_vectors, trial_pre_sims, generation)
                target_total_diffs, target_success, target_pre_sims, ppl_loss, target_shuffle = objective_function_batch(target_vectors , target_pre_sims, generation)

                #get next population
                new_mask = trial_total_diffs < target_total_diffs
                population = np.concatenate([trial_vectors[new_mask], target_vectors[~new_mask]], axis=0)
                target_values = np.concatenate([trial_total_diffs[new_mask], target_total_diffs[~new_mask]], axis=0)

                if trial_shuffle | target_shuffle:
                    n += 1
                    random.seed(random_seed + n)
                    for i in range(population_size):
                        candidate = [random.randint(bounds[i][0], bounds[i][1]) for i in range(num_parameters)]
                        population[i] = np.array(candidate)

                if trial_success | target_success:
                    break 

            best_solution = population[np.argmin(target_values)]

            return best_solution

        bounds = []
        index_list = list(synonyms_dict.keys())


        for i in range(num_replacements):

            word_index = index_list[i]

            if word_index in synonyms_dict:
                num_synonyms = len(synonyms_dict[word_index])
            else:
                num_synonyms = 0

            bounds.append( (word_index, word_index) )

            if num_synonyms == 0:
                bounds.append( (-1, -1) )
            else:
                bounds.append( (0, num_synonyms - 1) )
                          
        best_solution = differential_evolution(objective_function_batch, bounds, population_size, max_generations, F, CR)

        def Adjust_sentence(best_solution):

            text_cache = text_ls[:]

            for i in range(num_replacements):
                word_index = int(best_solution[i * 2])
                synonym_index = int(best_solution[i * 2 + 1])

                synonyms = synonyms_dict.get(word_index, [])

                if 0 <= synonym_index < len(synonyms):
                    synonym = synonyms[synonym_index]
                else:
                    synonym = text_cache[word_index]  

                text_cache = text_cache[:word_index] + [synonym] + text_cache[word_index + 1:]

            return text_cache
        
        adv = Adjust_sentence(best_solution)

        sim_score =  sim_predictor.semantic_sim([' '.join(text_ls)], [' '.join(adv)])[0]

        num_queries += 1
        num_changed = sum(1 for word1, word2 in zip(adv, text_ls) if word1 != word2)
        delta_PPL = PPL_metric.calculate_delta_PPL([' '.join(text_ls)], [' '.join(adv)])[0]

        if(sim_score < 0.6):
            return ' '.join(adv), num_changed, orig_label, orig_label, num_queries, sim_score, delta_PPL
        
        return ' '.join(adv), num_changed, orig_label, torch.argmax(predictor([adv])), num_queries, sim_score, delta_PPL


def SynDE(text_ls, true_label, predictor, PPL_metric, cos_sim, synonym_num, stop_words_set, word2idx, idx2word, replace_dict, sim_predictor=None,
           CR_=-1., max_generations_= -1, batch_size=32, Mutation_algorithm = 1):
    
    population_size = batch_size

    if max_generations_ != -1:
        max_generations = max_generations_ 
    else:
        max_generations = 200

    F = 1 

    if CR_ != -1.:
        CR = CR_
    else:
        CR = 0.9

    pos_ls = criteria.get_pos(text_ls)


    # first check the prediction of the original text
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0 , 0 , 0
    else:                                 
        # get the pos and verb tense info
        pos_ls = criteria.get_pos(text_ls)

        num_queries = 1
        num_changed = 0

        words_perturb = []
        for idx, word in enumerate(text_ls):
            try:
                if text_ls[idx] not in stop_words_set:
                    words_perturb.append((idx, text_ls[idx]))
            except:
                print(idx, len(text_ls))

        # find synonyms
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)
        synonyms_all = []

        for idx, word in words_perturb:
            if word in word2idx:
                if word == "|<SEP>|":
                    continue
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        synonyms_dict = {idx: synonyms for idx, synonyms in synonyms_all}

        for idx, synonyms in synonyms_dict.copy().items():
            text_cache = text_ls[:]
            for synonym in synonyms:
                text_cache = text_cache[:idx] + [synonym] + text_cache[idx + 1:]
                synonyms_pos_ls = [ criteria.get_pos(text_cache[max(idx - 14, 0):idx + 15])[min(14, idx)]
                                        if len(text_cache) > 20 else criteria.get_pos(text_cache)[idx] ]
                
                if not criteria.pos_filter(pos_ls[idx], synonyms_pos_ls)[0]:
                    synonyms_dict[idx].remove(synonym)

            if(len(synonyms_dict[idx]) == 0):
                del synonyms_dict[idx]

        num_replacements = len(synonyms_dict)

        def objective_function_batch(x_batch, pre_sim, generation):
            nonlocal num_queries

            k = orig_prob.cpu().detach().numpy()

            new_texts = []

            change_rates = np.zeros((len(x_batch),))

            for x in x_batch:
                text_cache = text_ls[:]

                for i in range(num_replacements):
                    word_index = int(x[i * 2])
                    synonym_index = int(x[i * 2 + 1])

                    synonyms = synonyms_dict.get(word_index, [])
                    synonym = synonyms[synonym_index] if (0 <= synonym_index < len(synonyms)) else text_ls[word_index] 

                    text_cache = text_cache[:word_index] + [synonym] + text_cache[word_index + 1:]

                new_texts.append(text_cache)
                num_changed = sum(1 for word1, word2 in zip(text_cache, text_ls) if word1 != word2)
                np.append(change_rates , (num_changed / len(text_ls)) )

            semantic_scores = sim_predictor.semantic_sim([' '.join(text_ls)] * len(new_texts) , 
                                    list(map(lambda x: ' '.join(x), new_texts)) )

            ppl_loss = PPL_metric.calculate_PPL_loss( list(map(lambda x: ' '.join(x), new_texts)) )

            new_probs = predictor(new_texts, batch_size=batch_size)
            num_queries += batch_size

            #to numpy
            true_label_probs = new_probs[:, orig_label].cpu().detach().numpy()
            label_jud_matrix = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()

            total_diffs = true_label_probs * 1/semantic_scores + np.log(ppl_loss) - ( label_jud_matrix * (1 / np.log(ppl_loss) + semantic_scores) \
                                                                                    + ~label_jud_matrix * (k - true_label_probs)  ) * 10 + change_rates * 2

            success = False
            shuffle = False

            if np.any(label_jud_matrix) & np.any(semantic_scores > 0.95) & label_jud_matrix[np.argmax(semantic_scores)]:
                success = True
            elif (~np.any(label_jud_matrix)) & (np.all(np.round(semantic_scores, 4) == np.round(pre_sim, 4))):
                shuffle = True

            total_diffs = total_diffs + (semantic_scores < 0.2) * 1
            
            return total_diffs, success, semantic_scores, ppl_loss, shuffle

        def differential_evolution(objective_function_batch, bounds, population_size, max_generations, F, CR):

            num_parameters = len(bounds)
            population = np.empty((population_size, num_parameters), dtype=int)

            for i in range(population_size):
                candidate = [random.randint(bounds[i][0], bounds[i][1]) for i in range(num_parameters)]
                population[i] = np.array(candidate)

            trial_pre_sims = np.zeros(batch_size)
            target_pre_sims = np.zeros(batch_size)

            target_values = np.array(objective_function_batch(population, target_pre_sims, 0)[0])
            n = 1
            
            for generation in range(max_generations):
                target_vectors = population.copy()
                mutant_vectors = np.empty((population_size, num_parameters), dtype=int)

                values = np.random.permutation(population_size)
                np.random.shuffle(values)

                a = values
                b = (values + n) % population_size
                c = (values + n + 1) % population_size
                d = (values + n + 2) % population_size

                a, b, c, d = population[a], population[b], population[c], population[d]

                if Mutation_algorithm == 1:
                    mutant_vectors = np.array(a + F * (b - c), dtype=int)
                elif Mutation_algorithm == 2:
                    mutant_vectors = np.where(np.random.uniform(0, 1, a.shape) < CR / 2, a + F * (b - c) + 1, a + F * (b - c))
                elif Mutation_algorithm == 3:
                    x_best = population[np.argmin(target_values)]
                    mutant_vectors = a + F * (b - a) + F * (c - x_best)
                elif Mutation_algorithm == 4:
                    mutant_vectors = a + F * (b - a) + F * (c - d)

                bounds_array = np.array(bounds)
                mutant_vectors = np.clip(mutant_vectors, bounds_array[:, 0], bounds_array[:, 1])

                mask = np.random.random((population_size, num_parameters)) < CR
                indices = np.arange(num_parameters)
                trial_vectors = np.where(mask | (indices == np.random.randint(0, num_parameters, size=population_size)[:, None]), mutant_vectors, target_vectors)

                trial_total_diffs, trial_success, trial_pre_sims, ppl_loss, trial_shuffle = objective_function_batch(trial_vectors, trial_pre_sims, generation)
                target_total_diffs, target_success, target_pre_sims, ppl_loss, target_shuffle = objective_function_batch(target_vectors , target_pre_sims, generation)

                new_mask = trial_total_diffs < target_total_diffs
                population = np.concatenate([trial_vectors[new_mask], target_vectors[~new_mask]], axis=0)
                target_values = np.concatenate([trial_total_diffs[new_mask], target_total_diffs[~new_mask]], axis=0)

                if trial_shuffle | target_shuffle:
                    n += 1
                    random.seed(random_seed + n)
                    for i in range(population_size):
                        candidate = [random.randint(bounds[i][0], bounds[i][1]) for i in range(num_parameters)]
                        population[i] = np.array(candidate)

                if trial_success | target_success:
                    break 

            best_solution = population[np.argmin(target_values)]

            return best_solution

        bounds = []
        index_list = list(synonyms_dict.keys())

        for i in range(num_replacements):

            word_index = index_list[i]

            if word_index in synonyms_dict:
                num_synonyms = len(synonyms_dict[word_index])
            else:
                num_synonyms = 0

            bounds.append( (word_index, word_index) )

            if num_synonyms == 0:
                bounds.append( (-1, -1) )
            else:
                bounds.append( (0, num_synonyms - 1) )
                          
        best_solution = differential_evolution(objective_function_batch, bounds, population_size, max_generations, F, CR)

        def Adjust_sentence(best_solution):

            text_cache = text_ls[:]

            for i in range(num_replacements):
                word_index = int(best_solution[i * 2])
                synonym_index = int(best_solution[i * 2 + 1])

                synonyms = synonyms_dict.get(word_index, [])

                if 0 <= synonym_index < len(synonyms):
                    synonym = synonyms[synonym_index]
                else:
                    synonym = text_cache[word_index] 

                text_cache = text_cache[:word_index] + [synonym] + text_cache[word_index + 1:]

            return text_cache
        
        adv = Adjust_sentence(best_solution)

        sim_score =  sim_predictor.semantic_sim([' '.join(text_ls)], [' '.join(adv)])[0]

        new_probs = predictor([adv], batch_size=batch_size)
        num_queries += 1
        num_changed = sum(1 for word1, word2 in zip(adv, text_ls) if word1 != word2)
        delta_PPL = PPL_metric.calculate_delta_PPL([' '.join(text_ls)], [' '.join(adv)])[0]

        print((orig_label == torch.argmax(new_probs, dim=-1)).data.cpu().numpy(), sim_score, num_changed / len(text_ls), "PPL:",delta_PPL)
        print(' '.join(adv))

        if(sim_score < 0.6):
            return ' '.join(adv), num_changed, orig_label, orig_label, num_queries, sim_score, delta_PPL
        
        return ' '.join(adv), num_changed, orig_label, torch.argmax(predictor([adv])), num_queries, sim_score, delta_PPL

def SynGA(text_ls, true_label, predictor, PPL_metric, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
         import_score_threshold=-1., synonym_num=50, batch_size=32, generations=10, mutation_rate=0.2):

    population_size = batch_size
    
    class BetaData:
        def __init__(self, text, fitness):
            self.text = text
            self.fitness = fitness

    def fitness_function(text, true_label, predictor):
        # Calculate fitness as per your attack goal
        with torch.no_grad():
            orig_probs = predictor([text]).squeeze()
            orig_label = torch.argmax(orig_probs)
            orig_prob = orig_probs.max()

            fitness = 1 - orig_probs[true_label]
                        
        return fitness.item()

    def crossover(parent1, parent2):
        # Perform crossover between two parents (e.g., single-point crossover)
        crossover_point = np.random.randint(1, len(parent1))
        offspring = parent1[:crossover_point] + parent2[crossover_point:]
        return offspring

    def mutate(text, synonyms_all, mutation_rate=0.1):
        mutated_text = list(text)
        for i in range(len(text)):
            if np.random.uniform(0, 1) < mutation_rate:
                if i in synonyms_all and synonyms_all[i]: 
                    mutated_text[i] = np.random.choice(synonyms_all[i])
        return mutated_text


    # Check original prediction
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()

    if true_label != orig_label:
        print(f"Already misclassified: True label {true_label}, Original label {orig_label}")
        return '', 0, orig_label, orig_label, 0, 0, 0

    len_text = len(text_ls)
    num_queries = 1

    
    words_perturb = []
    for idx, word in enumerate(text_ls):
        try:
            if text_ls[idx] not in stop_words_set:
                words_perturb.append((idx, text_ls[idx]))
        except:
            print(idx, len(text_ls))

    # find synonyms
    words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
    synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)
    synonyms_all = []

    for idx, word in words_perturb:
        if word in word2idx:
            if word == "|<SEP>|":
                continue
            synonyms = synonym_words.pop(0)
            if synonyms:
                synonyms_all.append((idx, synonyms))

    synonyms_dict = {idx: synonyms for idx, synonyms in synonyms_all}    
    
    pos_ls = criteria.get_pos(text_ls)

    for idx, synonyms in synonyms_dict.copy().items():
        text_cache = text_ls[:]
        for synonym in synonyms:
            text_cache = text_cache[:idx] + [synonym] + text_cache[idx + 1:]
            synonyms_pos_ls = [ criteria.get_pos(text_cache[max(idx - 14, 0):idx + 15])[min(14, idx)]
                                    if len(text_cache) > 20 else criteria.get_pos(text_cache)[idx] ]
            
            if not criteria.pos_filter(pos_ls[idx], synonyms_pos_ls)[0]:
                synonyms_dict[idx].remove(synonym)

        if(len(synonyms_dict[idx]) == 0):
            del synonyms_dict[idx]
            
    # Initialize population
    # population = [BetaData(text_ls[:], fitness_function(text_ls, true_label, predictor)) for _ in range(population_size)]
    
    population = []
    for _ in range(population_size):
        mutated_text = mutate(text_ls[:], synonyms_dict, mutation_rate=1.0)  # 初始时使用较高的突变率
        fitness = fitness_function(mutated_text, true_label, predictor)
        population.append(BetaData(mutated_text, fitness))
        # print(mutated_text, fitness)
    
    # Evolutionary loop
    for generation in range(generations):
        # Normalize fitness scores for selection
        total_fitness = sum(ind.fitness for ind in population)
        probabilities = [ind.fitness / total_fitness for ind in population]
        # print(probabilities)
        # Selection, crossover, mutation
        new_population = []
        for _ in range(population_size):
            # Select parents (tournament selection)
            parent1, parent2 = np.random.choice(population, size=2, replace=False, p=probabilities)

            # Perform crossover
            offspring = crossover(parent1.text, parent2.text)

            # Mutate offspring
            mutated_offspring = mutate(offspring, synonyms_dict, mutation_rate)

            # Evaluate fitness of mutated offspring
            fitness = fitness_function(mutated_offspring, true_label, predictor)
            # print(fitness, end = "! ")
            # Create new individual in the population
            new_population.append(BetaData(mutated_offspring[:], fitness))
        # print()
        # Replace old population with new population
        population = new_population

    # Final best individual (highest fitness)
    best_individual = max(population, key=lambda x: x.fitness)
    text_prime = best_individual.text
    num_changed = sum(1 for i in range(len(text_ls)) if text_ls[i] != text_prime[i])
    
    sim_score =  sim_predictor.semantic_sim([' '.join(text_ls)], [' '.join(text_prime)])[0]
    
    if sim_score < 0.5:
        return ' '.join(text_ls), 0, true_label, true_label, 0, 0, 0

    num_queries += 1
    delta_PPL = PPL_metric.calculate_delta_PPL([' '.join(text_ls)], [' '.join(text_prime)])[0]
    
    return ' '.join(text_prime), num_changed, true_label, torch.argmax(predictor([text_prime])), num_queries, sim_score, delta_PPL  # Update with appropriate sim_score calculation

def GBGA(text_ls, true_label, predictor, PPL_metric, replace_dict, stop_words_set, word2idx, idx2word, sim_predictor=None,
         import_score_threshold=-1., synonym_num=50, batch_size=32, generations=10, mutation_rate=0.2):

    population_size = batch_size
    
    class BetaData:
        def __init__(self, text, fitness):
            self.text = text
            self.fitness = fitness

    def fitness_function(text, true_label, predictor):
        # Calculate fitness as per your attack goal
        with torch.no_grad():
            orig_probs = predictor([text]).squeeze()

            fitness = 1 - orig_probs[true_label]
            
        return fitness.item()

    def crossover(parent1, parent2):
        # Perform crossover between two parents (e.g., single-point crossover)
        crossover_point = np.random.randint(1, len(parent1))
        offspring = parent1[:crossover_point] + parent2[crossover_point:]
        return offspring

    def mutate(text, synonyms_all, mutation_rate=0.1):
        mutated_text = list(text)
        for i in range(len(text)):
            if np.random.uniform(0, 1) < mutation_rate:
                if i in synonyms_all and synonyms_all[i]:  # 检查 synonyms_all[i] 是否不为空
                    mutated_text[i] = np.random.choice(synonyms_all[i])
        return mutated_text


    # Check original prediction
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()

    if true_label != orig_label:
        print(f"Already misclassified: True label {true_label}, Original label {orig_label}")
        return '', 0, orig_label, orig_label, 0, 0, 0

    len_text = len(text_ls)
    num_queries = 1

    pos_ls = criteria.get_pos(text_ls)

    words_perturb = []
    for idx, word in enumerate(text_ls):
        try:
            if text_ls[idx] not in stop_words_set:
                words_perturb.append((idx, text_ls[idx]))
        except:
            print(idx, len(text_ls))

    # find synonyms
    synonyms_all = []

    for idx, word in words_perturb:
        if word in word2idx:
            word_idx = word2idx[word]
            if word_idx in replace_dict:
                synonyms = [idx2word[idx] for idx in replace_dict[word_idx]]
                if synonyms:
                    synonyms_all.append((idx, synonyms))
            else:
                print("find replace word:{} failed.".format(word))
        else:
            continue

        synonyms_dict = {idx: synonyms for idx, synonyms in synonyms_all}

        for idx, synonyms in synonyms_dict.copy().items():
            text_cache = text_ls[:]
            for synonym in synonyms:
                text_cache = text_cache[:idx] + [synonym] + text_cache[idx + 1:]
                synonyms_pos_ls = [ criteria.get_pos(text_cache[max(idx - 14, 0):idx + 15])[min(14, idx)]
                                        if len(text_cache) > 20 else criteria.get_pos(text_cache)[idx] ]
                
                if not criteria.pos_filter(pos_ls[idx], synonyms_pos_ls)[0]:
                    synonyms_dict[idx].remove(synonym)

            if(len(synonyms_dict[idx]) == 0):
                del synonyms_dict[idx]
    

    for idx, synonyms in synonyms_dict.copy().items():
        text_cache = text_ls[:]
        for synonym in synonyms:
            text_cache = text_cache[:idx] + [synonym] + text_cache[idx + 1:]
            synonyms_pos_ls = [ criteria.get_pos(text_cache[max(idx - 14, 0):idx + 15])[min(14, idx)]
                                    if len(text_cache) > 20 else criteria.get_pos(text_cache)[idx] ]
            
            if not criteria.pos_filter(pos_ls[idx], synonyms_pos_ls)[0]:
                synonyms_dict[idx].remove(synonym)

        if(len(synonyms_dict[idx]) == 0):
            del synonyms_dict[idx]
            
    # Initialize population

    population = []
    for _ in range(population_size):
        mutated_text = mutate(text_ls[:], synonyms_dict, mutation_rate=1.0) 
        fitness = fitness_function(mutated_text, true_label, predictor)
        population.append(BetaData(mutated_text, fitness))
        # print(mutated_text, fitness)
    
    # Evolutionary loop
    for generation in range(generations):
        # Normalize fitness scores for selection
        total_fitness = sum(ind.fitness for ind in population)
        probabilities = [ind.fitness / total_fitness for ind in population]
        # Selection, crossover, mutation
        new_population = []
        for _ in range(population_size):
            # Select parents (tournament selection)
            parent1, parent2 = np.random.choice(population, size=2, replace=False, p=probabilities)

            # Perform crossover
            offspring = crossover(parent1.text, parent2.text)

            # Mutate offspring
            mutated_offspring = mutate(offspring, synonyms_dict, mutation_rate)

            # Evaluate fitness of mutated offspring
            fitness = fitness_function(mutated_offspring, true_label, predictor)
            # Create new individual in the population
            new_population.append(BetaData(mutated_offspring[:], fitness))
        # Replace old population with new population
        population = new_population

    # Final best individual (highest fitness)
    best_individual = max(population, key=lambda x: x.fitness)
    text_prime = best_individual.text
    num_changed = sum(1 for i in range(len(text_ls)) if text_ls[i] != text_prime[i])
    
    sim_score =  sim_predictor.semantic_sim([' '.join(text_ls)], [' '.join(text_prime)])[0]
    
    if sim_score < 0.5:
        return ' '.join(text_ls), 0, true_label, true_label, 0, 0, 0

    num_queries += 1
    delta_PPL = PPL_metric.calculate_delta_PPL([' '.join(text_ls)], [' '.join(text_prime)])[0]
    
    return ' '.join(text_prime), num_changed, true_label, torch.argmax(predictor([text_prime])), num_queries, sim_score, delta_PPL  # Update with appropriate sim_score calculation


def GBGreedy(text_ls, true_label, predictor, stop_words_set, word2idx, idx2word, replace_dict, PPL_metric, sim_predictor=None,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
           batch_size=32, ):
    # first check the prediction of the original text
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        print(true_label, orig_label)
        print(text_ls)
        return '', 0, orig_label, orig_label, 0, 0, 0
    else:
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(text_ls)

        # get importance score
        leave_1_texts = [text_ls[:ii] + ['<oov>'] + text_ls[min(ii + 1, len_text):] for ii in range(len_text)]
        leave_1_probs = predictor(leave_1_texts, batch_size=batch_size)
        num_queries += len(leave_1_texts)
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
        import_scores = (orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (
                    leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0,
                                                                      leave_1_probs_argmax))).data.cpu().numpy()

        # get words to perturb ranked by importance scorefor word in words_perturb
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            try:
                if score > import_score_threshold and text_ls[idx] not in stop_words_set:
                    words_perturb.append((idx, text_ls[idx]))
            except:
                print(idx, len(text_ls), import_scores.shape, text_ls, len(leave_1_texts))

        # find synonyms
        synonyms_all = []

        for idx, word in words_perturb:
            if word in word2idx:
                word_idx = word2idx[word]
                if word_idx in replace_dict:
                    synonyms = [idx2word[idx] for idx in replace_dict[word_idx]][:synonym_num]
                    if synonyms:
                        synonyms_all.append((idx, synonyms))
                else:
                    print("find replace word:{} failed.".format(word))
            else:
                continue

        # start replacing and attacking
        text_prime = text_ls[:]
        text_cache = text_prime[:]
        num_changed = 0
        sim_score = 0

        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            new_probs = predictor(new_texts, batch_size=batch_size)

            # compute semantic similarity 
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            semantic_sims = \
            sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                       list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

            num_queries += len(new_texts)
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= sim_score_threshold)
            # prevent incompatible pos
            synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask
            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_changed += 1

                sim_score =  sim_predictor.semantic_sim([' '.join(text_ls)], [' '.join(text_prime)])[0]
                break
            else:
                new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                        (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
                    num_changed += 1
            text_cache = text_prime[:]
        
        delta_PPL = PPL_metric.calculate_delta_PPL([' '.join(text_ls)], [' '.join(text_prime)])[0]

        return ' '.join(text_prime), num_changed, orig_label, torch.argmax(predictor([text_prime])), num_queries , sim_score, delta_PPL

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--dataset_path",
                        type=str,
                        required=True,
                        help="Which dataset to attack.")
    parser.add_argument("--nclasses",
                        type=int,
                        required=True,
                        default=2,
                        help="How many classes for classification.")
    parser.add_argument("--target_model",
                        type=str,
                        required=True,
                        choices=['wordLSTM', 'bert', 'wordCNN', 'roberta', 'xlnet'],
                        help="Target models for text classification: fasttext, charcnn, word level lstm "
                             "For NLI: InferSent, ESIM, bert-base-uncased")
    parser.add_argument("--target_model_path",
                        type=str,
                        required=True,
                        help="pre-trained target model path")
    parser.add_argument("--word_embeddings_path",
                        type=str,
                        default='',
                        help="path to the word embeddings for the target model")
    parser.add_argument("--counter_fitting_embeddings_path",
                        type=str,
                        required=True,
                        help="path to the counter-fitting embeddings we used to find synonyms")
    parser.add_argument("--counter_fitting_cos_sim_path",
                        type=str,
                        default='',
                        help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")
    parser.add_argument("--USE_cache_path",
                        type=str,
                        default='',
                        required=False,
                        help="Path to the USE encoder cache.")
    parser.add_argument("--output_dir",
                        type=str,
                        default='adv_results',
                        help="The output directory where the attack results will be written.")
    parser.add_argument("--device_ids",
                        type=str,
                        default="0,1",
                        required=True,
                        help="the gpu id")
    parser.add_argument("--Mutation_algorithm",
                        type=int,
                        default=1,
                        required=True,
                        help="Mutation algorithm")


    ## DE hyperparameters
    parser.add_argument("--CR",
                        default=0.9,
                        required=True,
                        type=float,
                        help="DE crossover probability")
    parser.add_argument("--max_generations",
                        default=200,
                        required=True,
                        type=int,
                        help="DE iterations")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="model train batch size and Number of solutions per generation")
    parser.add_argument("--beta",
                        default=10,
                        type=float,
                        help="model train batch size and Number of solutions per generation")

    ## Model hyperparameters
    parser.add_argument("--sim_score_window",
                        default=15,
                        type=int,
                        help="Text length or token number to compute the semantic similarity score")
    parser.add_argument("--import_score_threshold",
                        default=-1.,
                        type=float,
                        help="Required mininum importance score.")
    parser.add_argument("--sim_score_threshold",
                        default=0.7,
                        type=float,
                        help="Required minimum semantic similarity score.")
    parser.add_argument("--synonym_num",
                        default=50,
                        type=int,
                        help="Number of synonyms to extract")
    parser.add_argument("--data_size",
                        default=1000,
                        type=int,
                        help="Data size to create adversaries")
    parser.add_argument("--perturb_ratio",
                        default=0.,
                        required=True,
                        type=float,
                        help="Whether use random perturbation for ablation study")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="max sequence length for BERT target model")

    args = parser.parse_args()

    device_ids = [int(id) for id in args.device_ids.split(",")]

    gpu_ids = device_ids
    gpus = tf.config.experimental.list_physical_devices('GPU')    
    tf.config.set_visible_devices([gpus[id] for id in gpu_ids], 'GPU')
    torch.cuda.set_device(gpu_ids[0])

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # get data to attack
    texts, labels = dataloader.read_corpus(args.dataset_path, is_bert=True if args.target_model == 'bert' else False)

    data = list(zip(texts, labels))
    data = data[:args.data_size] # choose how many samples for adversary
    print("Data import finished!")

    # construct the model
    print("Building Model...")
    if args.target_model == 'wordLSTM':
        if "mnli" in args.target_model_path:
            model = Model(args.word_embeddings_path, hidden_size = 300, depth=3, nclasses=args.nclasses).cuda()
            checkpoint = torch.load(args.target_model_path, map_location='cuda:' + str(gpu_ids[0]))
        else:
            model = Model(args.word_embeddings_path, hidden_size = 150, depth=1, nclasses=args.nclasses).cuda()
            checkpoint = torch.load(args.target_model_path, map_location='cuda:' + str(gpu_ids[0]))
        model.load_state_dict(checkpoint)
    elif args.target_model == 'wordCNN':
        model = Model(args.word_embeddings_path, nclasses=args.nclasses, hidden_size=100, cnn=True).cuda()
        checkpoint = torch.load(args.target_model_path, map_location='cuda:' + str(gpu_ids[0]))
        model.load_state_dict(checkpoint)
    elif args.target_model == 'bert':
        model = NLI_infer_BERT(args.target_model_path, nclasses=args.nclasses, max_seq_length=args.max_seq_length)
    elif args.target_model == 'roberta':
        model = NLI_infer_ROBERTA(args.target_model_path, max_seq_length=args.max_seq_length)
    elif args.target_model == 'xlnet':
        model = NLI_infer_XLNET(args.target_model_path, max_seq_length=args.max_seq_length)
    predictor = model.text_pred
    print("Model built!")

    # prepare synonym extractor
    # build dictionary via the embedding file
    idx2word = {}
    word2idx = {}

    print("Building vocab...")
    with open(args.counter_fitting_embeddings_path, 'r') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1

    if args.perturb_ratio == 5. or args.perturb_ratio == 25. or args.perturb_ratio == 30.:
        print("Building cos sim matrix...")
        if args.counter_fitting_cos_sim_path:
            # load pre-computed cosine similarity matrix if provided
            print('Load pre-computed cosine similarity matrix from {}'.format(args.counter_fitting_cos_sim_path))
            cos_sim = np.load(args.counter_fitting_cos_sim_path)
        else:
            # calculate the cosine similarity matrix
            print('Start computing the cosine similarity matrix!')
            embeddings = []
            with open(args.counter_fitting_embeddings_path, 'r') as ifile:
                for line in ifile:
                    embedding = [float(num) for num in line.strip().split()[1:]]
                    embeddings.append(embedding)
            embeddings = np.array(embeddings)
            product = np.dot(embeddings, embeddings.T)
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            cos_sim = product / np.dot(norm, norm.T)
        print("Cos sim import finished!")

    # build the semantic similarity module
    use = USE(gpu_ids)

    # start attacking
    orig_failures = 0.
    adv_failures = 0.
    delta_PPL = 0.
    changed_rates = []
    nums_queries = []
    orig_texts = []
    adv_texts = []
    true_labels = []
    new_labels = []
    sim_scores = []
    delta_PPLs = []

    log_file = open(os.path.join(args.output_dir, 'results_log'), 'a')

    # 粒球攻击类
    if(os.path.exists("./Attacker/replace.npy")):
        replace_dict = np.load('./Attacker/replace.npy', allow_pickle=True).item()
    else:
        centers = np.load("./Attacker/cluster_center_result.npy", allow_pickle=True)
        word_balls = np.load("./Attacker/cluster_result.npy", allow_pickle=True)
        Ba = BallDict(centers=centers[0], balls=word_balls, path="./Attacker/euclidean_distances.npy", distance=0.5)
        replace_dict = Ba.get_replace_word_dict()

    stop_words_set = criteria.get_stopwords()
    print('Start attacking!')

    #GPT2
    PPL_metric = GPT2LM(gpu_ids, max_length = args.max_seq_length ,model_resolution = './GPT2')

    for idx, (text, true_label) in enumerate(tqdm(data, total=len(data), desc="Attacking...")):

        if idx % 100 == 0:
            print('{} samples out of {} have been finished!'.format(idx, args.data_size))
        
        if args.perturb_ratio == 5.:
            # Greedy
            new_text, num_changed, orig_label, \
            new_label, num_queries, sim_score = Greedy(text, true_label, predictor, stop_words_set,
                                            word2idx, idx2word, cos_sim, sim_predictor=use,
                                            sim_score_threshold=args.sim_score_threshold,
                                            import_score_threshold=args.import_score_threshold,
                                            sim_score_window=args.sim_score_window,
                                            synonym_num=args.synonym_num,
                                            batch_size=args.batch_size)
        elif args.perturb_ratio == 15.:
            # DE 粒球攻击
            new_text, num_changed, orig_label, \
            new_label, num_queries, sim_score, delta_PPL = GBDE(text, true_label, predictor, PPL_metric, stop_words_set, 
                                               word2idx, idx2word, replace_dict, sim_predictor=use,
                                               CR_= args.CR, max_generations_=args.max_generations, batch_size=args.batch_size, 
                                               Mutation_algorithm=args.Mutation_algorithm, beta = args.beta)
        elif args.perturb_ratio == 20.:
            # eval
            new_text, num_changed, orig_label, \
            new_label, num_queries, sim_score = eval(text, true_label, predictor)

        elif args.perturb_ratio == 25.:
            # Syn+DE
            new_text, num_changed, orig_label, \
            new_label, num_queries, sim_score, delta_PPL = SynDE(text, true_label, predictor, PPL_metric, cos_sim, args.synonym_num, stop_words_set, 
                                               word2idx, idx2word, replace_dict, sim_predictor=use,
                                               CR_=args.CR, max_generations_=args.max_generations, batch_size=args.batch_size, Mutation_algorithm=args.Mutation_algorithm)
        elif args.perturb_ratio == 30.:
            # Syn+GA
            new_text, num_changed, orig_label, \
            new_label, num_queries, sim_score, delta_PPL = SynGA(text, true_label, predictor, PPL_metric, stop_words_set,
                                            word2idx, idx2word, cos_sim, sim_predictor=use,
                                            import_score_threshold=args.import_score_threshold,
                                            synonym_num=args.synonym_num,
                                            batch_size=args.batch_size,
                                            generations=args.max_generations,
                                            mutation_rate = args.CR)
            
        elif args.perturb_ratio == 35.:
            # Syn+GA
            new_text, num_changed, orig_label, \
            new_label, num_queries, sim_score, delta_PPL = GBGA(text, true_label, predictor, PPL_metric, replace_dict, stop_words_set,
                                            word2idx, idx2word, sim_predictor=use,
                                            import_score_threshold=args.import_score_threshold,
                                            synonym_num=args.synonym_num,
                                            batch_size=args.batch_size,
                                            generations=args.max_generations,
                                            mutation_rate = args.CR)
            
        elif args.perturb_ratio == 40.:
            # Greedy
            new_text, num_changed, orig_label, \
            new_label, num_queries, sim_score, delta_PPL = GBGreedy(text, true_label, predictor, stop_words_set,
                                            word2idx, idx2word, replace_dict, PPL_metric, sim_predictor=use,
                                            sim_score_threshold=args.sim_score_threshold,
                                            import_score_threshold=args.import_score_threshold,
                                            sim_score_window=args.sim_score_window,
                                            synonym_num=args.synonym_num,
                                            batch_size=args.batch_size)

            
        if true_label != orig_label:
            orig_failures += 1


        if true_label != new_label:
            adv_failures += 1

        changed_rate = 1.0 * num_changed / len(text)

        if true_label == orig_label and true_label != new_label:
            changed_rates.append(changed_rate)
            orig_texts.append(' '.join(text))
            adv_texts.append(new_text)
            true_labels.append(true_label)
            new_labels.append(new_label)
            if sim_score != 0:
                sim_scores.append(sim_score)
                delta_PPLs.append(delta_PPL)


    message = 'Target model {} on {} Al-{} CR-{}: original accuracy: {:.3f}%, adv accuracy: {:.3f}%, ASR: {:.3f}%, ' \
              'avg changed rate: {:.3f}%, sim_score: {:.3f}\n, delta_PPL: {:.3f}\n'.format(
                                                                    args.target_model,
                                                                    args.dataset_path,
                                                                    args.Mutation_algorithm,
                                                                    args.CR,
                                                                    (1-orig_failures/args.data_size)*100,
                                                                    (1-adv_failures/args.data_size)*100,
                                                                    ((1 - orig_failures/args.data_size) - (1 - adv_failures/args.data_size)) / (1 - orig_failures/args.data_size) * 100,
                                                                        np.mean(changed_rates)*100,
                                                                    np.mean(sim_scores),
                                                                    np.mean(delta_PPLs))
    print(message)
    log_file.write(message)
    adversaries_filename = 'adversaries_{}_{}_{}_Al{}_CR{}.txt'.format(
                            args.target_model,
                            args.batch_size,
                            args.dataset_path[7:],
                            args.Mutation_algorithm,
                            args.CR)
    
    with open(os.path.join(args.output_dir, adversaries_filename), 'w') as ofile:
        for orig_text, adv_text, true_label, new_label in zip(orig_texts, adv_texts, true_labels, new_labels):
            ofile.write('orig sent ({}):\t{}\nadv sent ({}):\t{}\n\n'.format(true_label, orig_text, new_label, adv_text))

if __name__ == "__main__":
    main()
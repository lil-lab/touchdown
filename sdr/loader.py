import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

import json
import os
from collections import defaultdict

gpu = torch.device('cuda')
cpu = torch.device('cpu')

class Loader:
    def __init__(self, data_dir, image_dir, target_dir):
        self.data_dir = data_dir
        self.vocab = Vocabulary()
        self.max_length = 0
        self.datasets = {}
        self.image_dir = image_dir
        self.target_dir = target_dir

    def load_json(self, filename):
        path = os.path.join(self.data_dir, filename)
        data = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                for prefix in ['main', 'pre', 'post']:
                    pano_type = prefix + '_pano'
                    center_type = prefix + '_static_center'
                    center = json.loads(obj[center_type])
                    heading = prefix + '_heading'
                    if center == {'x': -1,'y': -1}:
                        continue
                    data.append({
                        'route_id': obj['route_id'],
                        'panoid': obj[pano_type], 
                        'center': center,
                        'text': obj['td_location_text'],
                    })
        return data

    def load_image_paths(self, data):
        image_paths = []
        for data_obj in data:
            panoid = data_obj['panoid']
            image_paths.append('{}{}.npy'.format(self.image_dir, panoid))
        return image_paths

    def load_target_paths(self, data):
        return ['{}{}.{}.npy'.format(self.target_dir, data_obj['route_id'], data_obj['panoid']) for data_obj in data]

    def load_texts(self, data):
        return [data_obj['text'] for data_obj in data]

    def load_centers(self, data):
        return [data_obj['center'] for data_obj in data]

    def load_route_ids(self, data):
        return [data_obj['route_id'] for data_obj in data]

    def build_vocab(self, texts, mode):
        '''Add words to the vocabulary'''
        ids = []
        seq_lengths = []
        for text in texts:
            line_ids = []
            words = text.lower().split()
            self.max_length = max(self.max_length, len(words))
            for word in words:
                word = self.vocab.add_word(word, mode)
                line_ids.append(self.vocab.word2idx[word])
            ids.append(line_ids)
            seq_lengths.append(len(words))
        text_ids = np.array([row + [0] * (self.max_length - len(row)) for row in ids])
        return text_ids, seq_lengths

    def build_dataset(self, file, gaussian_target, sample_used):
        mode, ext = file.split('.')
        print('[{}]: Start loading JSON file...'.format(mode))
        data = self.load_json(file)
        data_size = len(data)

        if isinstance(sample_used, tuple):
            start, end = sample_used
            data = data[start:end]
            num_samples = end - start
        elif isinstance(sample_used, int) or isinstance(sample_used, float):
            if sample_used <= 1:
                num_samples = int(len(data) * sample_used)
            elif 1 < sample_used < len(data):
                num_samples = int(sample_used)
            else:
                raise ValueError
            data = data[:num_samples]
        else:
            raise ValueError
        print('[{}]: Using {} ({}%) samples'.format(mode, num_samples, num_samples / data_size * 100))

        centers = self.load_centers(data)
        image_paths = self.load_image_paths(data)
        target_paths = self.load_target_paths(data)
        route_ids = self.load_route_ids(data)

        print('[{}]: Building vocab from text data...'.format(mode))
        texts = self.load_texts(data)
        texts, seq_lengths = self.build_vocab(texts, mode)

        print('[{}]: Building dataset...'.format(mode))
        dataset = TDLocationDataset(image_paths, texts, seq_lengths, target_paths, centers, gaussian_target, route_ids)
        self.datasets[mode] = dataset
        print('[{}]: Finish building dataset...'.format(mode))

class Vocabulary:
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}

    def add_word(self, word, mode):
        if word not in self.word2idx and mode in ('train', 'dev'):
            idx = len(self.idx2word)
            self.idx2word[idx] = word
            self.word2idx[word] = idx
            return word
        elif word not in self.word2idx and mode == 'test':
            return '<unk>'
        else:
            return word

    def __len__(self):
        return len(self.idx2word)


class TDLocationDataset(Dataset):
    def __init__(self, image_paths, texts, seq_lengths, target_paths, centers, gaussian_target, route_ids):
        self.image_paths = image_paths
        self.texts = texts
        self.seq_lengths = seq_lengths
        self.target_paths = target_paths
        self.gaussian_target = gaussian_target
        self.centers = centers
        self.route_ids = route_ids

    def __getitem__(self, index):
        route_id = self.route_ids[index]
        center = self.centers[index]
        text = torch.cuda.LongTensor(self.texts[index])
        seq_length = np.array(self.seq_lengths[index])
        target = torch.FloatTensor(np.load(self.target_paths[index]))
        image = np.load(self.image_paths[index]).transpose(2, 0, 1)
        image = torch.cuda.FloatTensor(image)

        if not self.gaussian_target:
            # concentrate the prob mass to the peak of the gaussian
            target_shape = target.size()
            flat_target = target.view(1, -1)
            target_pixel_val, target_pixel_idx = torch.max(flat_target, 1)
            target = torch.zeros(flat_target.size())
            target[:, target_pixel_idx] = 1
            target = target.view(target_shape)
        target = target.to(gpu)

        return image, text, seq_length, target, center, route_id

    def __len__(self):
        return len(self.image_paths)

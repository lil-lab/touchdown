import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

import copy

from loader import Loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, 
                 dropout, bidirectional, reduce):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.reduce = reduce # ['last', 'mean']

        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, seq_lengths):
        # transpose so the text data has shape=(seq_length, batch_size)
        x = x.t().contiguous()

        # prepare all the indices for sorting
        sorted_indices = np.argsort(-seq_lengths) # descending order
        unsort_indices = np.argsort(sorted_indices)

        embed = self.embedding(x)
        embed = self.dropout(embed)

        sorted_seq_lengths = seq_lengths[sorted_indices]
        sorted_embed = embed[:, sorted_indices] # sort by sequence lengths 
        embed_packed = pack_padded_sequence(sorted_embed, sorted_seq_lengths)

        outputs = []
        out_packed = embed_packed
        for i in range(self.num_layers):
            self.lstm.flatten_parameters()
            out_packed, _ = self.lstm(out_packed)

            # unpack and unsort the sequence
            out, _ = pad_packed_sequence(out_packed)
            out = out[:, unsort_indices]

            # reduce the dimension
            if self.reduce == 'last':
                out = out[seq_lengths - 1, np.arange(len(seq_lengths)), :]
            elif self.reduce == 'mean':
                seq_lengths_ = torch.cuda.FloatTensor(seq_lengths.numpy()).unsqueeze(-1)
                out = torch.sum(out[:, np.arange(len(seq_lengths_)), :], 0) / seq_lengths_
            outputs.append(out)

        return outputs


class LinearProjectionLayers(nn.Module):
    def __init__(self, image_channels, linear_hidden_size, rnn_hidden_size, num_hidden_layers):
        super(LinearProjectionLayers, self).__init__()

        if num_hidden_layers == 0:
            # map pixel feature vector directly to score without activation
            self.out_layers = nn.Linear(image_channels + rnn_hidden_size, 1, bias=False)
        else:
            linear_hidden_layers = []
            for i in range(num_hidden_layers):
                linear_hidden_layers += [nn.Linear(linear_hidden_size, linear_hidden_size), nn.ReLU()]
    
            self.out_layers = nn.Sequential(
                nn.Linear(image_channels + rnn_hidden_size, linear_hidden_size),
                nn.ReLU(),
                *linear_hidden_layers,
                nn.Linear(linear_hidden_size, 1, bias=False)
            )

    def forward(self, x):
        return self.out_layers(x)


def clones(module, N):
    '''Produce N identical layers'''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Concat(nn.Module):
    def __init__(self, rnn_args, out_layer_args, image_channels=128):
        super(Concat, self).__init__()

        if not rnn_args['bidirectional']:
            rnn_hidden_size = rnn_args['rnn_hidden_size']
        else:
            rnn_hidden_size = rnn_args['rnn_hidden_size'] * 2

        self.rnn = RNN(
            rnn_args['input_size'], 
            rnn_args['embed_size'], 
            rnn_args['rnn_hidden_size'], 
            rnn_args['num_rnn_layers'], 
            rnn_args['embed_dropout'],
            rnn_args['bidirectional'],
            rnn_args['reduce']
        ).to(device)

        self.out_layers = LinearProjectionLayers(
            image_channels=image_channels, 
            linear_hidden_size=out_layer_args['linear_hidden_size'], 
            rnn_hidden_size=rnn_hidden_size, 
            num_hidden_layers=out_layer_args['num_hidden_layers']
        )

    def forward(self, images, texts, seq_lengths):
        text_embed = self.rnn(texts, seq_lengths)[-1]
        image_embed = images
        image_embed = image_embed.permute([0, 2, 3, 1])
        batch_size, H, W, d = image_embed.size()
        
        text_embed = text_embed.repeat(1, H * W).view(batch_size, H, W, -1)
        concat_embed = torch.cat((image_embed, text_embed), -1)
        out = self.out_layers(concat_embed).squeeze(-1)
        out = F.log_softmax(out.view(batch_size, -1), 1).view(batch_size, H, W)
        return out


class ConcatConv(nn.Module):
    def __init__(self, rnn_args, cnn_args, out_layer_args, image_channels=128):
        super(ConcatConv, self).__init__()
        if not rnn_args['bidirectional']:
            rnn_hidden_size = rnn_args['rnn_hidden_size']
        else:
            rnn_hidden_size = rnn_args['rnn_hidden_size'] * 2

        self.rnn = RNN(
            rnn_args['input_size'], 
            rnn_args['embed_size'], 
            rnn_args['rnn_hidden_size'], 
            rnn_args['num_rnn_layers'], 
            rnn_args['embed_dropout'],
            rnn_args['bidirectional'],
            rnn_args['reduce']
        ).to(device)

        conv = nn.Conv2d(
            in_channels=image_channels + rnn_hidden_size, 
            out_channels=image_channels + rnn_hidden_size, 
            kernel_size=cnn_args['kernel_size'],
            padding=cnn_args['padding']
        )
        self.conv_layers = clones(conv, cnn_args['num_conv_layers'])

        self.out_layers = LinearProjectionLayers(
            image_channels=image_channels, 
            linear_hidden_size=out_layer_args['linear_hidden_size'], 
            rnn_hidden_size=rnn_hidden_size, 
            num_hidden_layers=out_layer_args['num_hidden_layers']
        )

    def forward(self, images, texts, seq_lengths):
        text_embed = self.rnn(texts, seq_lengths)[-1]
        image_embed = images
        image_embed = image_embed.permute([0, 2, 3, 1])
        batch_size, H, W, d = image_embed.size()
        
        text_embed = text_embed.repeat(1, H * W).view(batch_size, H, W, -1)
        concat_embed = torch.cat((image_embed, text_embed), -1)
        conv_embed = concat_embed.permute([0, 3, 1, 2])
        for conv_layer in self.conv_layers:
            conv_embed = conv_layer(conv_embed)
        conv_embed = conv_embed.permute([0, 2, 3, 1])
        out = self.out_layers(concat_embed).squeeze(-1)
        out = F.log_softmax(out.view(batch_size, -1), 1).view(batch_size, H, W)
        return out


class RNN2Conv(nn.Module):
    def __init__(self, rnn_args, cnn_args, out_layer_args, num_layers, image_channels=128):
        super(RNN2Conv, self).__init__()
        self.cnn_args = cnn_args
        self.rnn_args = rnn_args
        self.num_layers = num_layers
        
        self.image_channels = image_channels
        self.out_channels = self.image_channels // 2
        linear_hidden_size = out_layer_args['linear_hidden_size']
        if not rnn_args['bidirectional']:
            rnn_hidden_size = rnn_args['rnn_hidden_size']
        else:
            rnn_hidden_size = rnn_args['rnn_hidden_size'] * 2
        kernel_size = cnn_args['kernel_size']

        self.rnn = RNN(
            rnn_args['input_size'], 
            rnn_args['embed_size'], 
            rnn_args['rnn_hidden_size'], 
            rnn_args['num_rnn_layers'], 
            rnn_args['embed_dropout'],
            rnn_args['bidirectional'],
            rnn_args['reduce']
        ).to(device)

        self.conv_dropout = nn.Dropout(p=cnn_args['conv_dropout'])

        self.text2convs = nn.ModuleList([])
        for i in range(num_layers):
            in_channels = self.image_channels if i == 0 else self.out_channels
            self.text2convs.append(nn.Linear(
                rnn_hidden_size, 
                kernel_size * kernel_size * in_channels * self.out_channels
            ))

        self.out_layers = LinearProjectionLayers(
            image_channels=self.out_channels, 
            linear_hidden_size=linear_hidden_size, 
            rnn_hidden_size=0, 
            num_hidden_layers=out_layer_args['num_hidden_layers']
        )

    def forward(self, images, texts, seq_lengths):
        text_embeds = self.rnn(texts, seq_lengths)
        batch_size, image_channels, H, W = images.size()

        for i, (text_embed, text2conv) in enumerate(zip(text_embeds, self.text2convs)):
            in_channels = image_channels if i == 0 else self.out_channels

            conv_filters = text2conv(text_embed).view(
                batch_size, 
                self.out_channels, 
                in_channels, 
                self.cnn_args['kernel_size'], 
                self.cnn_args['kernel_size']
            )
            conv_filters = self.conv_dropout(conv_filters)

            conv_outs = []
            for conv_filter, image in zip(conv_filters, images):
                conv_out = F.conv2d(image.unsqueeze(0), conv_filter, padding=self.cnn_args['padding'])
                conv_outs.append(conv_out)
            images = torch.cat(conv_outs, 0)

        convolved_images = images
        convolved_images = convolved_images.permute([0, 2, 3, 1])
        out = self.out_layers(convolved_images).squeeze(-1)
        out = F.log_softmax(out.view(batch_size, -1), 1).view(batch_size, H, W)
        return out


class LingUNet(nn.Module):
    def __init__(self, rnn_args, cnn_args, out_layer_args, image_channels=128, m=None):
        super(LingUNet, self).__init__()
        self.cnn_args = cnn_args
        self.rnn_args = rnn_args
        self.m = m
        self.image_channels = image_channels

        if not rnn_args['bidirectional']:
            self.rnn_hidden_size = rnn_args['rnn_hidden_size']
        else:
            self.rnn_hidden_size = rnn_args['rnn_hidden_size'] * 2
        assert self.rnn_hidden_size % self.m == 0

        self.rnn = RNN(rnn_args['input_size'], 
                       rnn_args['embed_size'], 
                       rnn_args['rnn_hidden_size'], 
                       rnn_args['num_rnn_layers'], 
                       rnn_args['embed_dropout'],
                       rnn_args['bidirectional'],
                       rnn_args['reduce']).to(device)

        sliced_text_vector_size = self.rnn_hidden_size // self.m
        flattened_conv_filter_size = 1 * 1 * self.image_channels * self.image_channels
        self.text2convs = clones(nn.Linear(sliced_text_vector_size, flattened_conv_filter_size), self.m)

        conv = nn.Conv2d(
            in_channels=self.image_channels, 
            out_channels=self.image_channels, 
            kernel_size=cnn_args['kernel_size'], 
            padding=cnn_args['padding'],
            stride=1
        )
        self.conv_layers = clones(conv, self.m)

        # create deconv layers with appropriate paddings
        self.deconv_layers = nn.ModuleList([])

        for i in range(self.m):
            in_channels = self.image_channels if i == 0 else self.image_channels * 2
            out_channels = self.image_channels
            self.deconv_layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=cnn_args['kernel_size'], 
                    padding=cnn_args['padding'],
                    stride=1,
                )
            )
        self.deconv_dropout = nn.Dropout(p=cnn_args['deconv_dropout'])

        self.out_layers = LinearProjectionLayers(
            image_channels=self.image_channels, 
            linear_hidden_size=out_layer_args['linear_hidden_size'], 
            rnn_hidden_size=0, 
            num_hidden_layers=out_layer_args['num_hidden_layers']
        )

    def forward(self, images, texts, seq_lengths):
        batch_size, image_channels, height, width = images.size()

        text_embed = self.rnn(texts, seq_lengths)[-1]
        sliced_size = self.rnn_hidden_size // self.m
        
        Gs = []
        image_embeds = images
        for i in range(self.m): 
            image_embeds = self.conv_layers[i](image_embeds)
            text_slice = text_embed[:, i * sliced_size:(i + 1) * sliced_size]

            conv_kernel_shape = (batch_size, self.image_channels, self.image_channels, 1, 1)
            text_conv_filters = self.text2convs[i](text_slice).view(conv_kernel_shape)

            # looping over batch TODO: this is very inefficient, need to optimize
            outputs = []
            for image_embed, text_conv_filter in zip(image_embeds, text_conv_filters):
                output = F.conv2d(image_embed.unsqueeze(0), text_conv_filter)
                outputs.append(output)
            G = torch.cat(outputs, 0)
            Gs.append(G)

        # deconvolution operations, from the bottom up
        H = Gs.pop()
        for i in range(self.m):
            if i == 0:
                H = self.deconv_dropout(H)
                H = self.deconv_layers[i](H)
            else:
                G = Gs.pop()
                concated = torch.cat((H, G), 1)
                H = self.deconv_layers[i](concated)

        H = H.permute([0, 2, 3, 1])
        out = self.out_layers(H).squeeze(-1)
        out = F.log_softmax(out.view(batch_size, -1), 1).view(batch_size, height, width)
        return out


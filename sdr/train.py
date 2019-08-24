import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from tensorboardX import SummaryWriter

import logging
import argparse
import datetime
import os
import copy

from loader import Loader
from model import Concat
from model import ConcatConv
from model import RNN2Conv
from model import LingUNet


parser = argparse.ArgumentParser(description='SDR task')

# set path to data folders
parser.add_argument('--data_dir', type=str, default=None,
                    help='path to data folder where train.json, dev.json, and test.json files')
parser.add_argument('--image_dir', type=str, default=None,
                    help='path to `image_features`')
parser.add_argument('--target_dir', type=str, default=None,
                    help='path to sdr_targets')

parser.add_argument('--name', type=str, default='run',
                    help='name of the run')
parser.add_argument('--model', type=str, default='concat',
                    help='model used')

# CNN
parser.add_argument('--num_conv_layers', type=int, default=1,
                    help='number of conv layers')
parser.add_argument('--conv_dropout', type=float, default=0.0,
                    help='dropout applied to the conv_filters (0 = no dropout)')
parser.add_argument('--deconv_dropout', type=float, default=0.0,
                    help='dropout applied to the deconv_filters (0 = no dropout)')
# RNN
parser.add_argument('--embed_size', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--rnn_hidden_size', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--num_rnn_layers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--bidirectional', type=bool, default=False,
                    help='use bidirectional rnn')
parser.add_argument('--embed_dropout', type=float, default=0.1,
                    help='dropout applied to the embedding layer (0 means no dropout)')

# final linear layers
parser.add_argument('--num_linear_hidden_layers', type=int, default=1,
                    help='number of final linear hidden layers')
parser.add_argument('--linear_hidden_size', type=int, default=128,
                    help='final linear hidden layer size')

# architecture specific arguments
parser.add_argument('--num_rnn2conv_layers', type=int, default=None,
                    help='number of rnn2conv layers')
parser.add_argument('--num_lingunet_layers', type=int, default=None,
                    help='number of LingUNet layers')
parser.add_argument('--num_unet_layers', type=int, default=None,
                    help='number of UNet layers')
parser.add_argument('--num_reslingunet_layers', type=int, default=None,
                    help='number of ResLingUNet layers')


parser.add_argument('--gaussian_target', type=bool, default=True,
                    help='use Gaussian target')
parser.add_argument('--sample_used', type=float, default=1.0,
                    help='portion of sample used for training')
parser.add_argument('--tuneset_ratio', type=float, default=0.07,
                    help='portion of tune set')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--grad_clip', type=float, default=0.5,
                    help='gradient clipping')
parser.add_argument('--num_epoch', type=int, default=30,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                    help='batch size')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--print_every', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--log', action='store_true',
                    help='log losses')
parser.add_argument('--summary', action='store_true',
                    help='write summary to tensorboard')
parser.add_argument('--no_date', action='store_true', default=True,
                    help='do not append date to the run name')

args = parser.parse_args()

if not args.no_date:
    now = datetime.datetime.now()
    t = (now.year, now.month, now.day, now.hour, now.minute)
    run_name = '{}_{}_{:02d}{:02d}_{:02d}{:02d}/'.format(args.name, *t)
else:
    run_name = args.name

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# set up summary writer for tensorboard logging
if args.summary:
    writer = SummaryWriter(os.path.join('/home/hc839/street-view-navigation/touchdown_location/runs/', run_name))
    counters = {'train': 0, 'tune': 0, 'dev': 0}

if args.log:
    out_dir = os.path.join('/home/hc839/street-view-navigation/touchdown_location/logs/', run_name)
    os.system('mkdir {}'.format(out_dir))
    print('Log directory created under {}'.format(out_dir))
    
    # log file
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join(out_dir, 'loss.log')))
    logger.setLevel(logging.INFO)
    logger.disabled = False


def write_summary(mode, log_dict):
    global counters
    for name, value in log_dict.items():
        writer.add_scalar(name, value, counters[mode])
    counters[mode] += 1


def log(mode, log_info):
    if mode == 'train':
        log_string = '[Train]: Epoch {:3d} | {:5d}/{:5d} batches | lr {:04.4f} | Loss {:5.6f} | Mean Dist {:5.4f} | Accuracy {:5.4f}'
    elif mode == 'dev':
        log_string = '[Dev]:   Epoch {:3d} | Loss {:5.6f} | Mean Dist {:5.4f} | Accuracy {:5.4f}'
    elif mode == 'tune':
        log_string = '[Tune]:   Epoch {:3d} | Loss {:5.6f} | Mean Dist {:5.4f} | Accuracy {:5.4f}'
    if args.log:
        logger.info(log_string.format(*log_info))
    print(log_string.format(*log_info))


def distance_metric(preds, targets):
    """Calculate distances between model predictions and targets within a batch."""
    preds = preds.cpu()
    targets = targets.cpu()
    distances = []
    for pred, target in zip(preds, targets):
        pred_coord = np.unravel_index(pred.argmax(), pred.size())
        target_coord = np.unravel_index(target.argmax(), target.size())
        dist = np.sqrt((target_coord[0] - pred_coord[0]) ** 2 + (target_coord[1] - pred_coord[1]) ** 2)
        distances.append(dist)
    return distances

def accuracy(distances, margin=10):
    """Calculating accuracy at 80 pixel by default"""
    num_correct = 0
    for distance in distances:
        num_correct = num_correct + 1 if distance < margin else num_correct
    return num_correct / len(distances)

def convert_model_to_state(model, args, rnn_args, cnn_args, out_layer_args):
    state = {
        'args': vars(args),
        'rnn_args': rnn_args,
        'cnn_args': cnn_args,
        'out_layer_args': out_layer_args,
        'state_dict': {}
    }
    # use copies instead of references
    for k, v in model.state_dict().items():
        state['state_dict'][k] = v.clone().to(torch.device('cpu'))
    return state

def evaluate(model, data_iterator, mode, epoch):
    model.eval()
    total_loss = 0
    distances = []
    num_batches = 0

    with torch.no_grad():
        for batch_images, batch_texts, batch_seq_lengths, batch_targets, _, _ in data_iterator:
            batch_size, C, H, W = batch_images.size()

            batch_size = batch_images.size(0)
            preds = model(batch_images, batch_texts, batch_seq_lengths)
            loss = loss_func(preds, batch_targets) / batch_size

            total_loss += loss.item()
            num_batches += 1

            distances += distance_metric(preds, batch_targets)
            
        avg_loss = total_loss / num_batches
        mean_dist = np.mean(distances)
        acc = accuracy(distances)
    if args.summary:
        log_dict = {'{}_loss'.format(mode): avg_loss, 'accuracy': acc}
        write_summary(mode, log_dict)
    log(mode, (epoch, avg_loss, mean_dist, acc))
    return acc


def train(model, data_iterator, epoch):
    model.train()
    total_loss = 0
    batch_idx = 0
    num_batches = len(data_iterator) 

    for batch_images, batch_texts, batch_seq_lengths, batch_targets, _, _ in data_iterator:
        batch_size, C, H, W = batch_images.size()

        optimizer.zero_grad()
        preds = model(batch_images, batch_texts, batch_seq_lengths)
        loss = loss_func(preds, batch_targets) / batch_size

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % args.print_every == 0 and batch_idx > 0:
            avg_loss = total_loss / args.print_every
            total_loss = 0
            
            distances = distance_metric(preds, batch_targets)
            mean_dist = np.mean(distances)
            acc = accuracy(distances)
            log('train', (epoch, batch_idx, num_batches, optimizer.param_groups[-1]['lr'], avg_loss, mean_dist, acc))

            if args.summary:
                log_dict = {'train_loss': avg_loss, 'train_acc': acc, 'train_mean_dist': mean_dist}
                write_summary('train', log_dict)
        batch_idx += 1


def split_dataset(dataset, split_ratio, batch_size, shuffle_split=False):
    # creating data indices for training and tuning splits
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(dataset_size * split_ratio)

    if shuffle_split:
        np.random.seed(args.seed)
        np.random.shuffle(indices)

    train_indices = indices[split:]
    tune_indices = indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    tune_sampler = SubsetRandomSampler(tune_indices)

    train_iterator = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    tune_iterator = DataLoader(dataset, batch_size=batch_size, sampler=tune_sampler)
    return train_iterator, tune_iterator


if __name__ == '__main__':
    # load data
    loader = Loader(data_dir=args.data_dir, image_dir=args.image_dir, target_dir=args.target_dir)

    loader.build_dataset(
        file='train.json', 
        gaussian_target=args.gaussian_target, 
        sample_used=args.sample_used
    )

    loader.build_dataset(
        file='dev.json', 
        gaussian_target=args.gaussian_target, 
        sample_used=args.sample_used
    )

    loader.build_dataset(
        file='test.json', 
        gaussian_target=args.gaussian_target, 
        sample_used=args.sample_used
    )


    rnn_args = {
        'input_size': len(loader.vocab),
        'embed_size': args.embed_size,
        'rnn_hidden_size': args.rnn_hidden_size,
        'num_rnn_layers': args.num_rnn_layers,
        'embed_dropout': args.embed_dropout,
        'bidirectional': args.bidirectional,
        'reduce': 'last' if not args.bidirectional else 'mean'
    }
    cnn_args = {}
    out_layer_args = {'linear_hidden_size': args.linear_hidden_size, 'num_hidden_layers': args.num_linear_hidden_layers}

    if args.model == 'concat':
        model = Concat(rnn_args, out_layer_args)

    elif args.model == 'concat_conv':
        cnn_args = {'kernel_size': 5, 'padding': 2, 'num_conv_layers': args.num_conv_layers, 'conv_dropout': args.conv_dropout}
        model = ConcatConv(rnn_args, cnn_args, out_layer_args)

    elif args.model == 'rnn2conv':
        assert args.num_rnn2conv_layers is not None
        assert args.num_rnn2conv_layers <= args.num_rnn_layers
        cnn_args = {'kernel_size': 5, 'padding': 2, 'conv_dropout': args.conv_dropout}
        model = RNN2Conv(rnn_args, cnn_args, out_layer_args, args.num_rnn2conv_layers)

    elif args.model == 'lingunet':
        assert args.num_lingunet_layers is not None
        cnn_args = {'kernel_size': 5, 'padding': 2, 'deconv_dropout': args.deconv_dropout}
        model = LingUNet(rnn_args, cnn_args, out_layer_args, m=args.num_lingunet_layers)

    else:
        raise ValueError('Please specify model.')

    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print('Number of parameters:', num_params)

    if args.log:
        logger.info(args)
        logger.info(rnn_args)
        logger.info(cnn_args)
        logger.info(out_layer_args)
        logger.info('Number of parameters: {}'.format(num_params))

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.KLDivLoss(reduction='sum')

    # start training
    best_tune_acc = float('-inf')
    best_model = None
    patience = 0

    for epoch in range(args.num_epoch):
        dev_iterator = DataLoader(
            dataset=loader.datasets['dev'], 
            batch_size=args.batch_size, 
            shuffle=False
        )
        train_iterator, tune_iterator = split_dataset(loader.datasets['train'], args.tuneset_ratio, args.batch_size)

        train(model, train_iterator, epoch)
        tune_acc = evaluate(model, tune_iterator, mode='tune', epoch=epoch)

        if tune_acc > best_tune_acc:
            best_model = copy.deepcopy(model)
            best_tune_acc = tune_acc
            patience = 0
            if args.log:
                save_path = os.path.join(out_dir, '{}_acc{:.2f}_epoch{}.pt'.format(args.name, tune_acc, epoch))
                state = convert_model_to_state(best_model, args, rnn_args, cnn_args, out_layer_args)
                torch.save(state, save_path)
                logger.info('[Sys]:   Model saved')
            print('[Tune]: Best tune accuracy:', best_tune_acc)
        else:
            # acc not better, update patience
            patience += 1
            # learning rate scheduling
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 1.0
        print('Patience:', patience)
        if patience > 3:
            break

    dev_iterator = DataLoader(
        dataset=loader.datasets['dev'], 
        batch_size=args.batch_size, 
        shuffle=False
    )
    dev_acc = evaluate(best_model, dev_iterator, mode='dev', epoch=0)

    if args.log:
        print('Dev accuracy:', dev_acc)
        logger.info('Dev accuracy: {}'.format(dev_acc))


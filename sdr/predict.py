import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from collections import defaultdict

from loader import Loader
from train import accuracy
from train import distance_metric
from model import VisualLinearNetwork
from model import VisualConvNetwork
from model import RNN2Conv
from model import LingUNet
from model import UNet
from model import ResLingUNet

cpu = torch.device('cpu')
gpu = torch.device('cuda')
loss_func = nn.KLDivLoss(size_average=False)

def remove_padding(seq):
    while seq[-1] == '<pad>':
        seq.pop()
    return seq

def load_model(path, model_type):
    state = torch.load(path)
    rnn_args = state['rnn_args']
    cnn_args = state['cnn_args']
    out_layer_args = state['out_layer_args']

    if model_type == 'vis_linear':
        model = VisualLinearNetwork(rnn_args, out_layer_args)

    elif model_type == 'vis_conv':
        model = VisualConvNetwork(rnn_args, cnn_args, out_layer_args)

    elif model_type == 'rnn2conv':
        num_rnn2conv_layers = state['args']['num_rnn2conv_layers']
        model = RNN2Conv(rnn_args, cnn_args, out_layer_args, num_rnn2conv_layers)

    elif model_type == 'lingunet':
        num_lingunet_layers = state['args']['num_lingunet_layers']
        model = LingUNet(rnn_args, cnn_args, out_layer_args, m=num_lingunet_layers)

    elif model_type == 'unet':
        num_unet_layers = state['args']['num_unet_layers']
        model = UNet(cnn_args, out_layer_args, m=num_unet_layers)

    elif model_type == 'reslingunet':
        num_reslingunet_layers = state['args']['num_reslingunet_layers']
        model = ResLingUNet(rnn_args, cnn_args, out_layer_args, m=num_reslingunet_layers)
    model = model.to(gpu)
    model.load_state_dict(state['state_dict'])
    model.eval()
    return model

def retrieve_completed_route_ids(route_ids, distances, margin, successful_routes):
    """Retrieve route ids where their distances < margin"""
    
    for route_id, distance in zip(route_ids, distances):
        if distance < margin:
            successful_routes.add(route_id)

def save_successful_routes(successful_routes, path):
    with open(path, 'w') as f:
        for line in list(successful_routes):
            f.write(str(line) + '\n')

def predict(model, data_iterator, loader, mode):
    per_text_distances = defaultdict(list)
    total_loss = 0
    distances = []
    route_ids = []
    num_batches = 0
    successful_routes40 = set()
    successful_routes80 = set()
    successful_routes120 = set()

    with torch.no_grad():
        for i, (batch_images, batch_texts, batch_seq_lengths, batch_targets, batch_centers, batch_route_ids) in enumerate(data_iterator):
            batch_size, C, H, W = batch_images.size()
            if i % 20 == 0:
                print('{} batches evaled, batch_size = {}'.format(i, batch_size))
            preds = model(batch_images, batch_texts, batch_seq_lengths)

            loss = loss_func(preds, batch_targets) / batch_size
            total_loss += loss.item()
            num_batches += 1
            batch_distances = distance_metric(preds, batch_targets)
            distances += batch_distances
            route_ids += batch_route_ids.tolist()

            for text, dist in zip(batch_texts, batch_distances):
                text = [loader.vocab.idx2word[w.item()] for w in text]
                per_text_distances[' '.join(text)].append(dist)

        retrieve_completed_route_ids(route_ids, distances, 5, successful_routes40)
        retrieve_completed_route_ids(route_ids, distances, 10, successful_routes80)
        retrieve_completed_route_ids(route_ids, distances, 15, successful_routes120)
        save_successful_routes(successful_routes40, path='{}_successful_routes40.txt'.format(mode))
        save_successful_routes(successful_routes80, path='{}_successful_routes80.txt'.format(mode))
        save_successful_routes(successful_routes120, path='{}_successful_routes120.txt'.format(mode))
            
        avg_loss = total_loss / num_batches
        mean_dist = np.mean(distances) * 8
        acc40 = accuracy(distances, margin=5)
        acc80 = accuracy(distances, margin=10)
        acc120 = accuracy(distances, margin=15)
        acc = (acc40, acc80, acc120)

        acc40_per_text = 0
        acc80_per_text = 0
        acc120_per_text = 0
        total_count = 0
        for text, dists in per_text_distances.items():
           
            acc40_per_text = acc40_per_text + 1 if sum([d < 5 for d in dists]) == len(dists) else acc40_per_text
            acc80_per_text = acc80_per_text + 1 if sum([d < 10 for d in dists]) == len(dists) else acc80_per_text
            acc120_per_text = acc120_per_text + 1 if sum([d < 15 for d in dists]) == len(dists) else acc120_per_text
            total_count += 1
        acc_per_text = (acc40_per_text / total_count, acc80_per_text / total_count, acc120_per_text / total_count)
        print(acc_per_text)
    return acc, mean_dist, avg_loss

def predict_baselines(data_iterator, loader, opt):
    per_text_distances = defaultdict(list)
    distances = []

    with torch.no_grad():
        for i, (_, batch_texts, batch_seq_lengths, batch_targets, batch_centers, batch_route_ids) in enumerate(data_iterator):
            H = 800
            W = 460 * 8
            for x, y, text in zip(batch_centers['x'], batch_centers['y'], batch_texts):
                x, y = x.item(), y.item()
                w = W * x
                h = H * y
                if opt == 'CENTER':
                    w_pred, h_pred = W * 0.5, H * 0.5
                elif opt == 'RANDOM':
                    w_pred, h_pred = W * np.random.uniform(0, 1), H * np.random.uniform(0, 1)
                elif opt == 'AVERAGE':
                    w_pred, h_pred = W * 0.5299110677027002, H * 0.5137217333987077
                dist = np.sqrt((w - w_pred) ** 2 + (h - h_pred) ** 2)
                distances.append(dist)

                # track consistency
                text = [loader.vocab.idx2word[w.item()] for w in text]
                per_text_distances[' '.join(text)].append(dist)

        mean_dist = np.mean(distances) 
        acc40 = accuracy(distances, margin=40)
        acc80 = accuracy(distances, margin=80)
        acc120 = accuracy(distances, margin=120)
        acc = (acc40, acc80, acc120)

        acc40_per_text = 0
        acc80_per_text = 0
        acc120_per_text = 0
        total_count = 0
        for text, dists in per_text_distances.items():
            acc40_per_text = acc40_per_text + 1 if sum([d < 40 for d in dists]) == len(dists) else acc40_per_text
            acc80_per_text = acc80_per_text + 1 if sum([d < 80 for d in dists]) == len(dists) else acc80_per_text
            acc120_per_text = acc120_per_text + 1 if sum([d < 120 for d in dists]) == len(dists) else acc120_per_text
            total_count += 1
        acc_per_text = (acc40_per_text / total_count, acc80_per_text / total_count, acc120_per_text / total_count)
        print('Consistency:', acc_per_text)
    return acc, mean_dist

def export_predictions(model, data_iterator, loader, experiment):
    with torch.no_grad():
        for i, (batch_images, batch_texts, batch_seq_lengths, batch_targets, batch_centers, batch_route_ids) in enumerate(data_iterator):
            batch_size, C, H, W = batch_images.size()
            out = model(batch_images, batch_texts, batch_seq_lengths)

            text = [loader.vocab.idx2word[id] for id in batch_texts.squeeze(0).cpu().numpy()]
            text = remove_padding(text)
            text = ' '.join(text)

            with open('example/{}/text/dev_text_{}.txt'.format(experiment, i), 'w') as f:
                f.write(text)
            np.save('example/{}/pred/dev_pred_{}.npy'.format(experiment, i), out.squeeze(0).cpu().numpy())
            np.save('example/{}/target/dev_target_{}.npy'.format(experiment, i), batch_targets.squeeze(0).cpu().numpy())

def run_export():
#################
    # fixed LingUNet
    experiment = 'lingunet_layer4_embeddp05_lr5e-4_2019_0527_0718'
    model_name = 'lingunet_layer4_embeddp05_lr5e-4_acc0.33_epoch3.pt'
    model_type = 'lingunet'
#################
#    experiment = 'lingunet_layer2_embeddp05_lr5e-4_2018_1011_0458/'
#    model_name = 'lingunet_layer2_embeddp05_lr5e-4_acc0.37_epoch3.pt'
#    model_type = 'lingunet'
#################
#    experiment = 'birnn2conv_embeddp05_2018_1009_0039/'
#    model_name = 'birnn2conv_embeddp05_acc0.33_epoch5.pt'
#    model_type = 'rnn2conv'
#################
#    experiment = 'vis_linear_birnn_embeddp05_2018_1009_0043/'
#    model_name = 'vis_linear_birnn_embeddp05_acc0.21_epoch9.pt'
#    model_type = 'vis_linear'
#################
#    experiment = 'vis_conv_birnn_embeddp05_2018_1013_1630/'
#    model_name = 'vis_conv_birnn_embeddp05_acc0.20_epoch7.pt'
#    model_type = 'vis_conv'
    train_sample_used = 1.0
    dev_sample_used = 0.0020 # exporting 7 examples
    dev_sample_used = 0.005 # exporting 19 examples

    os.system('mkdir example/{}'.format(experiment))
    os.system('mkdir example/{}/target/'.format(experiment))
    os.system('mkdir example/{}/pred/'.format(experiment))
    os.system('mkdir example/{}/text/'.format(experiment))
    os.system('mkdir example/{}/image/'.format(experiment))
    os.system('mkdir example/{}/all/'.format(experiment))

    loader = Loader('../data/')
    loader.build_dataset('train.json', gaussian_target=True, sample_used=train_sample_used)
    loader.build_dataset('dev.json', gaussian_target=True, sample_used=dev_sample_used)
#    data_iterator = DataLoader(dataset=loader.datasets['train'], batch_size=1, shuffle=False)
    data_iterator = DataLoader(dataset=loader.datasets['dev'], batch_size=1, shuffle=False)

    model = load_model(os.path.join('logs/', experiment, model_name), model_type)
    export_predictions(model, data_iterator, loader, experiment)

def run_eval(base=False):
#    experiment = 'lingunet_layer4_embeddp05_lr5e-4_2019_0414_2111/'
#    model_name = 'lingunet_layer4_embeddp05_lr5e-4_acc0.33_epoch3.pt'

#    experiment = 'lingunet_layer2_embeddp05_lr5e-4_2019_0527_0717'
#    model_name = 'lingunet_layer2_embeddp05_lr5e-4_acc0.33_epoch3.pt'

#    experiment = 'lingunet_layer3_embeddp05_lr5e-4_2019_0527_0717'
#    model_name = 'lingunet_layer3_embeddp05_lr5e-4_acc0.31_epoch2.pt'

    experiment = 'lingunet_layer4_embeddp05_lr5e-4_2019_0527_0718'
    model_name = 'lingunet_layer4_embeddp05_lr5e-4_acc0.33_epoch3.pt'

#    experiment = 'lingunet_layer2_embeddp05_lr5e-4_2018_1011_0458/'
#    model_name = 'lingunet_layer2_embeddp05_lr5e-4_acc0.37_epoch3.pt'
#    experiment = 'lingunet_layer2_embeddp05_lr5e-4_2018_1011_1616'
#    model_name = 'lingunet_layer2_embeddp05_lr5e-4_acc0.36_epoch2.pt'
#    experiment = 'lingunet_layer2_embeddp05_lr5e-4_2018_1011_1823/'
#    model_name = 'lingunet_layer2_embeddp05_lr5e-4_acc0.36_epoch4.pt'
    model_type = 'lingunet'
#################
#    experiment = 'birnn2conv_embeddp05_2018_1009_0039/'
#    model_name = 'birnn2conv_embeddp05_acc0.33_epoch5.pt'
#    experiment = 'birnn2conv_embeddp05_2018_1010_0606/'
#    model_name = 'birnn2conv_embeddp05_acc0.33_epoch5.pt'
#    experiment = 'birnn2conv_embeddp05_2018_1010_0610/'
#    model_name = 'birnn2conv_embeddp05_acc0.32_epoch4.pt'
#    model_type = 'rnn2conv'
#################
#    experiment = 'vis_linear_birnn_embeddp05_2018_1009_0043/'
#    model_name = 'vis_linear_birnn_embeddp05_acc0.21_epoch9.pt'
#    experiment = 'vis_linear_birnn_embeddp05_2018_1012_0625'
#    model_name = 'vis_linear_birnn_embeddp05_acc0.20_epoch8.pt'
#    experiment = 'vis_linear_birnn_embeddp05_2018_1012_0627'
#    model_name = 'vis_linear_birnn_embeddp05_acc0.20_epoch5.pt'
#    model_type = 'vis_linear'
#################
#    experiment = 'vis_conv_birnn_embeddp05_2018_1013_1630/'
#    model_name = 'vis_conv_birnn_embeddp05_acc0.20_epoch7.pt'
#    experiment = 'vis_conv_birnn_embeddp05_2018_1013_1633/'
#    model_name = 'vis_conv_birnn_embeddp05_acc0.20_epoch9.pt'
#    experiment = 'vis_conv_birnn_embeddp05_2018_1013_1639/'
#    model_name = 'vis_conv_birnn_embeddp05_acc0.20_epoch10.pt'
#    model_type = 'vis_conv'
#################
#    experiment = 'unet_layer2_lr5e-4_2018_1109_1625/'
#    model_name = 'unet_layer2_lr5e-4_acc0.13_epoch5.pt'
#    experiment = 'unet_layer2_lr5e-4_2018_1109_1650/'
#    model_name = 'unet_layer2_lr5e-4_acc0.15_epoch9.pt'
#    experiment = 'unet_layer2_lr5e-4_2018_1109_1827/'
#    model_name = 'unet_layer2_lr5e-4_acc0.15_epoch9.pt'
#    model_type = 'unet'
#################
#    experiment = 'lingunet_layer2_embeddp05_lr5e-4_noslice_2018_1108_1408/'
#    model_name = 'lingunet_layer2_embeddp05_lr5e-4_noslice_acc0.34_epoch2.pt'
#    experiment = 'lingunet_layer2_embeddp05_lr5e-4_noslice_2018_1108_1412/'
#    model_name = 'lingunet_layer2_embeddp05_lr5e-4_noslice_acc0.33_epoch1.pt'
#    experiment = 'lingunet_layer2_embeddp05_lr5e-4_noslice_2018_1108_1415/'
#    model_name = 'lingunet_layer2_embeddp05_lr5e-4_noslice_acc0.34_epoch2.pt'
#    model_type = 'lingunet'
#################
#    experiment = 'reslingunet_layer4_embeddp05_r5e-4_2018_1110_1613/'
#    model_name = 'reslingunet_layer4_embeddp05_r5e-4_acc0.34_epoch8.pt'
#    model_type = 'reslingunet'
#################
    mode = 'test'
    train_sample_used = 1.0
    dev_sample_used = 1.0
    test_sample_used = 1.0

    loader = Loader('../data/')
    loader.build_dataset('train.json', gaussian_target=True, sample_used=train_sample_used)
    loader.build_dataset('{}.json'.format(mode), gaussian_target=True, sample_used=dev_sample_used)

    data_iterator = DataLoader(dataset=loader.datasets[mode], batch_size=20, shuffle=False)

    if base:
        opts = ['RANDOM', 'CENTER', 'AVERAGE']
        for opt in opts:
            acc, mean_dist = predict_baselines(data_iterator, loader, opt)
            print(f'Results for {opt}')
            print('Accuracy:', acc)
            print('Mean distance:', mean_dist)
    else:
        model = load_model(os.path.join('logs/', experiment, model_name), model_type)
        acc, mean_dist, avg_loss = predict(model, data_iterator, loader, mode)
        print(f'Results for {experiment} on checkpoint {model_name}')
        print('Accuracy:', acc)
        print('Mean distance:', mean_dist)
        print('Average loss:', avg_loss)


if __name__ == '__main__':
    run_export()
#    run_eval(base=False)
#    run_eval(base=True)


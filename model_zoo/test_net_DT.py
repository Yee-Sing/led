import torch
import argparse
import numpy as np
import torch.nn as nn
from .neurons.spiking_neuron import *
from torch.cuda.amp import autocast as autocast
from .model.snn_network import EDSNN_LIF_ADAPTIVE_final, PAEVSNN_LIF_AMPLIF_final

def parse_arguments():
    # basic parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('-network', type=str, default='EDSNN_LIF_ADAPTIVE_final')  # EDSNN_LIF_ADAPTIVE_final or PAEVSNN_LIF_AMPLIF_final
    parser.add_argument('-data_dir', type=str, default='./data/poster_6dof_cut.txt', help='root dir of events stream dataset')
    parser.add_argument('-out_dir', type=str, default='./logs_SW', help='root dir for saving logs and checkpoint')
    parser.add_argument('-save_path', type=str, default='./results')
    parser.add_argument('-num_events_per_pixel', type=float, default=0.5)

    # training parameters
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=8, type=int, help='batch size')
    parser.add_argument('-epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam', help='use which optimizer. SDG or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=2e-3, type=float, help='learning rate')
    
    return parser.parse_args()
   
def create_model(args):
    EDSNN_LIF_denoise = EDSNN_LIF_ADAPTIVE_final(kwargs=args)
    return EDSNN_LIF_denoise

def load_model_DT():
    args = parse_arguments()
    device = args.device
    gpus = [0]
    network_kwargs = {'activation_type': 'lif',
                      'mp_activation_type': 'amp_lif',
                      'spike_connection': 'concat',
                      'num_encoders': 3,
                      'num_resblocks': 1,
                      'v_threshold': 1.0,
                      'v_reset': 0.0,  # ,None
                      'tau': 2.0
                      }
    net = create_model(network_kwargs).to(device)
    net = nn.DataParallel(net.to(device), device_ids=gpus, output_device=gpus[0])
    checkpoint = torch.load("model_zoo/pretrain/SW/checkpoint_epoch_148.pth")
    net.load_state_dict(checkpoint['net'])
    net.eval()
    return device, net

def infer_DTSNN(states, device, net, data):
    with torch.no_grad():
        events_frame = data[0].copy()
        events_frame[events_frame != 0] = 1
        data_exp = np.expand_dims(np.expand_dims(events_frame, axis=0), axis=0)
        inputs = torch.from_numpy(data_exp.astype(np.float32)).to(device)
        outputs, thres_map = net(inputs, states)
        if states is None:
            states = torch.zeros_like(outputs)
        states.detach_()
        states = outputs
        results = outputs.cpu().numpy()[0][0] * events_frame
        thres_map_np = thres_map.cpu().numpy()[0][0]
        results[results >= thres_map_np] = 1
        results[results < thres_map_np] = 0
        return states, np.concatenate([np.expand_dims(results, axis=0), np.expand_dims(data[0], axis=0)], axis=0)
import os
import re
import cv2
import glob
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model_zoo.neurons.spiking_neuron import *
from torch.nn.utils.rnn import pad_sequence
from matplotlib.colors import ListedColormap
from torch.cuda.amp import autocast as autocast
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model_zoo.model.snn_network import EDSNN_LIF_ADAPTIVE_final, PAEVSNN_LIF_AMPLIF_final
from spikingjelly.activation_based import neuron, functional, surrogate, layer

torch.backends.cudnn.benchmark = True
_seed_ = 2023
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

def parse_arguments():
    # basic parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('-network', type=str, default='EDSNN_LIF_ADAPTIVE_final')  # EDSNN_LIF_ADAPTIVE_final or PAEVSNN_LIF_AMPLIF_final
    parser.add_argument('-data_dir', type=str, default='./data/poster_6dof_cut.txt', help='root dir of events stream dataset')
    parser.add_argument('-out_dir', type=str, default='./logs_SW', help='root dir for saving logs and checkpoint')
    parser.add_argument('-save_path', type=str, default='./results')
    parser.add_argument('-height', type=int, default=180)
    parser.add_argument('-width', type=int, default=240)
    parser.add_argument('-num_events_per_pixel', type=float, default=0.5)

    # training parameters
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=8, type=int, help='batch size')
    parser.add_argument('-epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam', help='use which optimizer. SDG or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=2e-3, type=float, help='learning rate')
    
    return parser.parse_args()

def preprocess_data(data, If_label_ts=False):
    # 获取中心区域
    center_h = data.shape[0] // 2
    center_w = data.shape[1] // 2
    half_size = 120

    # 计算中心区域的范围
    start_h = center_h - half_size
    end_h = center_h + half_size
    start_w = center_w - half_size
    end_w = center_w + half_size

    # 截取中心区域
    data = data[start_h:end_h, start_w:end_w]

    if If_label_ts:
        pass
    else:
        # 仍用无极性0、1脉冲量表征事件
        data[data != 0] = 1
    
    image_tensor = torch.from_numpy(data.astype(np.float32))
    return image_tensor

def preprocess_data_2(data):
    original_input = data
    original_tensor = torch.from_numpy(original_input.astype(np.float32))

    data[data != 0] = 1
    image_tensor = torch.from_numpy(data.astype(np.float32))
    
    return image_tensor, original_tensor

def get_events_frame(xs, ys, ps):
    events_frame = np.zeros((720, 1280))
    ps[ps == 0] = -1
    events_frame[ys, xs] = ps
    return events_frame

class Training_Dataset(Dataset):
    def __init__(self, data_dir, label_dir, label_ts_dir, time_sequence, start_index):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.label_ts_dir = label_ts_dir
        self.time_steps = time_sequence
        self.start_index = start_index

        self.train_data_files = self.get_train_data_files(data_dir)
        self.train_label_files = self.get_train_data_files(label_dir)
        self.train_label_ts_files = self.get_train_data_files(label_ts_dir)

    def get_train_data_files(self, data_dir):
        scene_dirs = os.listdir(data_dir)
        scene_dirs = sorted(scene_dirs, key=self.sort_by_number)

        random.seed(_seed_)
        random.shuffle(scene_dirs)

        train_data_files = []
        for scene_dir in scene_dirs:
            scene_path = os.path.join(data_dir, scene_dir)
            if os.path.isdir(scene_path):
                files = os.listdir(scene_path)
                files = sorted(files, key=self.sort_by_number)
                files = files[self.start_index: self.start_index + self.time_steps]  # 选取指定数量的数据
                files = [os.path.join(scene_path, file) for file in files]
                train_data_files.append(files)
        return train_data_files

    def sort_by_number(self, filename):
        # 从文件名中提取数字
        number = re.findall(r'\d+', filename)
        if number:
            return int(number[0])
        else:
            return 0

    def __getitem__(self, index):
        scene_data_files = self.train_data_files[index]
        scene_label_files = self.train_label_files[index]
        scene_label_ts_files = self.train_label_ts_files[index]

        image_sequence = []
        label_sequence = []
        label_ts_sequence = []

        for t in range(self.time_steps):
            image_file = scene_data_files[t % len(scene_data_files)]
            label_file = scene_label_files[t % len(scene_label_files)]
            label_ts_file = scene_label_ts_files[t % len(scene_label_ts_files)]

            event_data = np.load(image_file)
            image_data = get_events_frame(event_data[0], event_data[1], event_data[2])
            image = preprocess_data(image_data)

            label_data = np.load(label_file)
            label_data_1C = label_data[:, :, 0] + label_data[:, :, 1]
            label = preprocess_data(label_data_1C)

            label_ts_data = np.load(label_ts_file)
            label_ts = preprocess_data(label_ts_data, If_label_ts=True)

            image_sequence.append(image)
            label_sequence.append(label)
            label_ts_sequence.append(label_ts)

        image_sequence = torch.stack(image_sequence)
        label_sequence = torch.stack(label_sequence)
        label_ts_sequence = torch.stack(label_ts_sequence)

        return image_sequence, label_sequence, label_ts_sequence, scene_data_files, scene_label_files

    def __len__(self):
        return len(self.train_data_files)

class Validation_Dataset(Dataset):
    def __init__(self, data_dir, label_dir, label_ts_dir, time_sequence, start_index):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.label_ts_dir = label_ts_dir
        self.time_steps = time_sequence
        self.start_index = start_index

        self.train_data_files = self.get_train_data_files(data_dir)
        self.train_label_files = self.get_train_data_files(label_dir)
        self.train_label_ts_files = self.get_train_data_files(label_ts_dir)

    def get_train_data_files(self, data_dir):
        scene_dirs = os.listdir(data_dir)
        scene_dirs = sorted(scene_dirs, key=self.sort_by_number)

        random.seed(_seed_)
        random.shuffle(scene_dirs)

        train_data_files = []
        for scene_dir in scene_dirs:
            scene_path = os.path.join(data_dir, scene_dir)
            if os.path.isdir(scene_path):
                files = os.listdir(scene_path)
                files = sorted(files, key=self.sort_by_number)
                files = files[self.start_index: self.start_index + self.time_steps]  # 选取指定数量的数据
                files = [os.path.join(scene_path, file) for file in files]
                train_data_files.append(files)
        return train_data_files

    def sort_by_number(self, filename):
        # 从文件名中提取数字
        number = re.findall(r'\d+', filename)
        if number:
            return int(number[0])
        else:
            return 0

    def __getitem__(self, index):
        scene_data_files = self.train_data_files[index]
        scene_label_files = self.train_label_files[index]
        scene_label_ts_files = self.train_label_ts_files[index]

        image_sequence = []
        label_sequence = []
        label_ts_sequence = []
        origin_input_sequence = []

        for t in range(self.time_steps):
            image_file = scene_data_files[t % len(scene_data_files)]
            label_file = scene_label_files[t % len(scene_label_files)]
            label_ts_file = scene_label_ts_files[t % len(scene_label_ts_files)]

            event_data = np.load(image_file)
            image_data = get_events_frame(event_data[0], event_data[1], event_data[2])
            image, origin_input = preprocess_data_2(image_data)

            label_data = np.load(label_file)
            label_data_1C = label_data[:, :, 0] + label_data[:, :, 1]
            label_data_1C[label_data_1C != 0] = 1
            label = torch.from_numpy(label_data_1C.astype(np.float32))

            label_ts_data = np.load(label_ts_file)
            label_ts = torch.from_numpy(label_ts_data.astype(np.float32))

            image_sequence.append(image)
            label_sequence.append(label)
            label_ts_sequence.append(label_ts)
            origin_input_sequence.append(origin_input)
            
        image_sequence = torch.stack(image_sequence)
        label_sequence = torch.stack(label_sequence)
        label_ts_sequence = torch.stack(label_ts_sequence)
        origin_input_sequence = torch.stack(origin_input_sequence)

        return image_sequence, label_sequence, label_ts_sequence, origin_input_sequence, scene_data_files, scene_label_files

    def __len__(self):
        return len(self.train_data_files)    
    
def create_model(args):
    EDSNN_LIF_denoise = EDSNN_LIF_ADAPTIVE_final(kwargs=args)
    return EDSNN_LIF_denoise

def compute_loss(original, prediction, target, thres_prediction, thres_target):
    count_noevents = torch.sum(original == 0).item()
    count_signals = torch.sum(target == 1).item()
    count_noise = torch.sum((original-target) == 1).item()    
    weight_factor = torch.ones_like(target)

    if count_noise != 0:
        weight_factor[(original - target) == 1] = count_noevents / count_noise
    if count_signals != 0:
        weight_factor[target == 1] = count_noevents / count_signals

    prediction_final = prediction.clone()
    prediction_final = prediction_final * original
    prediction_final[prediction_final >= thres_prediction] = 1
    prediction_final[prediction_final < thres_prediction] = 0

    criterion = nn.BCELoss(weight=weight_factor)

    loss1 = 1.0 * torch.mean(weight_factor * torch.abs(prediction - target))
    loss2 = 1.0 * criterion(prediction_final, target)
    loss3 = 1.0 * torch.mean(torch.abs(thres_prediction - thres_target))

    print()
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print("预测值和目标值Loss1", loss1)
    print("截断后预测值和目标值Loss2", loss2)
    print("预测阈值和目标阈值Loss3", loss3)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

    # loss = loss1 + loss2 + loss3
    loss = loss1 + loss2

    # 计算去噪精度
    result_signals = prediction_final * original
    GT_N = torch.sum((original-target) == 1).item()
    GT_P = torch.sum(target == 1).item()
    FP =  torch.sum(((result_signals == 1) & ((original-target) == 1))).item()  
    TN = GT_N - FP
    TP = torch.sum(((result_signals == 1) & (target == 1))).item()
    if GT_P != 0:
        signal_retain = TP / GT_P
    else:
        signal_retain = 0
        
    if GT_N != 0:
        noise_removal = TN / GT_N
    else:
        noise_removal = 0
        
    return loss, loss3, signal_retain, noise_removal

def postprocess(original, prediction, target, prediction_thres, target_thres, polarity_mask, datanames_batch, out_dir, epoch):
    prediction_channel = prediction
    polarity_flat = polarity_mask
    # 对输出进行原始输入过滤
    result = prediction_channel * original
    result[result >= prediction_thres] = 1
    result[result < prediction_thres] = 0
     
    epoch_result_dir = os.path.join(out_dir, f"epoch_result_{epoch}")
    os.makedirs(epoch_result_dir, exist_ok=True)

    precision_file_path = os.path.join(epoch_result_dir, "precision.txt")
    
    with open(precision_file_path, "a") as f:
        for i in range(result.shape[1]):
            # Get the data name
            data_name = datanames_batch[i][0]
            scene_name, image_name = data_name.split('/')[-2:]
            
            # Create a folder for the scene_name if it doesn't exist
            scene_folder = os.path.join(epoch_result_dir, scene_name)
            os.makedirs(scene_folder, exist_ok=True)
            
            GT_N = torch.sum((original[0, i]-target[0, i]) == 1).item()
            GT_P = torch.sum(target[0, i] == 1).item()
            FP =  torch.sum(((result[0, i] == 1) & ((original[0, i]-target[0, i]) == 1))).item()  
            TN = GT_N - FP
            TP = torch.sum(((result[0, i] == 1) & (target[0, i] == 1))).item()
            
            # 计算去噪精度
            if GT_P != 0:
                signal_retain = TP / GT_P
            else:
                signal_retain = 0
                
            if GT_N != 0:
                noise_removal = TN / GT_N
            else:
                noise_removal = 0

            # Modify the save name
            save_name = f"{image_name.split('_')[0]}_denoised_{signal_retain:.4f}_{noise_removal:.4f}.png"

            # Concatenate the save path
            save_path = os.path.join(scene_folder, save_name)

            # Add black border
            result[0, i] = result[0, i] * polarity_flat[0, i]
            bordered_image = np.pad(result[0, i].cpu().numpy(), pad_width=1, mode='constant', constant_values=0)

            # Save the image with specified resolution and black border
            width = 1280
            height = 720
            dpi = 100
            cmap = ListedColormap(['red', 'white', 'blue'])

            fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
            ax.imshow(bordered_image, cmap=cmap, vmin=-1, vmax=1)
            ax.axis('off')
            fig.set_size_inches(width / dpi, height / dpi)
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0, facecolor='black')
            plt.close(fig)

            # Write the precision value to the precision.txt file
            f.write("GT_N: {}\n".format(GT_N))
            f.write("GT_P: {}\n".format(GT_P))
            f.write("Signal_retain: {:.4f}\n".format(signal_retain))
            f.write("Noise_removal: {:.4f}\n".format(noise_removal))

def freeze_net(net, whichpart, requires_grad):
    if whichpart == "thres":
        for name, param in net.named_parameters():
            if "thresadaptive" in name:
                param.requires_grad = requires_grad
    else:
        for name, param in net.named_parameters():
            if "thresadaptive" not in name:
                param.requires_grad = requires_grad
            
def train_denoisingsnn(args):
    device = args.device
    gpus = [0]
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    model_name = args.network
    event_files = args.data_dir
    save_path = args.save_path
    height = args.height
    width = args.width
    num_events_per_pixel = args.num_events_per_pixel

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
    print('net.device_ids=', net.device_ids)

    # 训练集文件路径
    train_data_dir = "/home/u2022010028/Data/EVK-EVK-DATA-NPY_CUT/train/original_rectify" 
    train_label_dir = "/home/u2022010028/Data/EVK-EVK-DATA-NPY_CUT/train/label"
    train_label_ts_dir = "/home/u2022010028/Data/EVK-EVK-DATA-NPY_CUT/train/label_ts"

    # 验证集文件路径
    validation_data_dir = "/home/u2022010028/Data/EVK-EVK-DATA-NPY_CUT/test/original_rectify"
    validation_label_dir = "/home/u2022010028/Data/EVK-EVK-DATA-NPY_CUT/test/label"
    validation_label_ts_dir = "/home/u2022010028/Data/EVK-EVK-DATA-NPY_CUT/test/label_ts"

    # 创建自定义数据集实例
    sample_sequence = 10  # 样本长度
    time_sequence = 10  # 时间序列

    start_epoch = 0
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)  # net.parameters()对网络中所有参数优化
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam([{'params': net.module.static_conv.parameters()}, 
                                      {'params': net.module.down1.parameters()},
                                      {'params': net.module.down2.parameters()},
                                      {'params': net.module.down3.parameters()},
                                      {'params': net.module.residualBlock.parameters()},
                                      {'params': net.module.up1.parameters()},
                                      {'params': net.module.up2.parameters()},
                                      {'params': net.module.up3.parameters()},
                                      {'params': net.module.temporalflat.parameters()},
                                      {'params': net.module.sigmoid.parameters()},], lr=args.lr)
        optimizer_thres = torch.optim.Adam(net.module.thresadaptive.parameters(), lr=0.002)
    else:
        raise NotImplementedError(args.opt)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True, min_lr=0)

    out_dir = os.path.join(args.out_dir, f'b{args.b}_{args.opt}_lr{args.lr}_L1_loadtest')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    checkpoint_dir = os.path.join(out_dir, 'checkpoints')
    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    max_average_train_precision = float('-inf')
    
    # 获取保存的模型文件列表
    checkpoint_files = glob.glob(os.path.join(out_dir, 'checkpoint_epoch_*.pth'))
    if checkpoint_files:
        # 根据文件名中的数字排序，获取最新的模型文件路径
        latest_model_file = max(checkpoint_files, key=lambda x: int(re.findall(r'\d+', x)[-1]))
        # 从文件名中提取出最新的 epoch 数字
        start_epoch = int(re.findall(r'\d+', latest_model_file)[-1]) + 1
        print(f"Loaded checkpoint from epoch {start_epoch - 1}")
        # 加载最新的模型文件
        checkpoint = torch.load(latest_model_file)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        average_train_loss = checkpoint['average_train_loss']
        average_train_precision = checkpoint['average_train_precision']
    else:
        start_epoch = 0  # 如果没有模型文件，则从头开始训练

    # 加载阈值参数
    # model_dict=net.state_dict()
    # thres_module_checkpoint = torch.load("/users/u2022010028/Project/CVPR_2024_Duan/EdSnn_ThresAdaptive_Seperate/logs_MiniThres/b1_adam_lr0.002_L1_loadtest/checkpoint_max.pth")
    # for k in model_dict:
    #     if "thresadaptive" in k:
    #         model_dict[k] = thres_module_checkpoint['net'][k.replace("thresadaptive.", '')]
    # net.load_state_dict(model_dict)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print('-------------------------------start_epoch: {}-------------------------------'.format(start_epoch))
        epoch = start_epoch+1
        reset_counter = 0
        iteration_counter = 0
        net.train()
        total_train_loss = 0
        total_train_loss_thres = 0
        total_train_precision = 0
        for time_slice in range(sample_sequence // time_sequence):
            print('-------------------------------time_slice: {}-------------------------------'.format(time_slice))
            # 创建训练数据集与数据加载器
            train_dataset = Training_Dataset(train_data_dir, train_label_dir, train_label_ts_dir, time_sequence=time_sequence, start_index=0 + time_slice * time_sequence)
            train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, drop_last=True, num_workers=12)
            for data_batch, label_batch, label_ts_batch, datanames_batch, labelnames_batch in train_loader:

                states = None  # 必须每个batch载入后，都对states 进行清零

                optimizer.zero_grad()
                optimizer_thres.zero_grad()

                original = data_batch.to(device)
                label = label_batch.to(device)  # label_batch.shape= torch.Size([batch, time_sequence, H, W])
                label_ts = label_ts_batch.to(device)
                results = torch.zeros_like(label_batch).to(device)
                results_ts = torch.zeros_like(label_ts_batch).to(device)
                for time_steps in range(time_sequence):
                    img = data_batch[:, time_steps, :, :].unsqueeze(dim=1).to(device)  # img.shape= torch.Size([batch, 1, H, W])
                    if model_name == 'EDSNN_LIF_ADAPTIVE_final':
                        membrane_potential, thres_map = net(img, states)  # membrane_potential.shape= torch.Size([batch, 1, H, W])
                        if states is None:
                            states = torch.zeros_like(membrane_potential)  # 根据 membrane_potential 的形状分配初始值
                        states.detach_()
                        states = membrane_potential
                        results[:, time_steps:time_steps + 1, :, :] = membrane_potential[:, :, :, :]
                        results_ts[:, time_steps:time_steps + 1, :, :] = thres_map[:, :, :, :]
                train_loss, train_loss_thres, train_signal_retain, train_noise_removal = compute_loss(original, results, label, results_ts, label_ts)
                print('Btach_train_loss=', train_loss)
                print('Btach_train_loss_thres=', train_loss_thres)
                print('Btach_train_signal_retain=', train_signal_retain)
                print('Btach_train_noise_removal=', train_noise_removal)
                total_train_loss += train_loss.item()
                total_train_loss_thres += train_loss_thres.item()
                total_train_precision += (train_signal_retain + train_noise_removal)/2
                iteration_counter += 1

                freeze_net(net, 'thres', requires_grad=False)
                freeze_net(net, 'others', requires_grad=True)
                train_loss.backward()
                optimizer.step()

                freeze_net(net, 'thres', requires_grad=True)
                freeze_net(net, 'others', requires_grad=False)
                train_loss_thres.backward()
                optimizer_thres.step()

                freeze_net(net, 'thres', requires_grad=True)
                freeze_net(net, 'others', requires_grad=True)

                # 优化一次参数后，需要重置网络的状态，需要清楚SNN神经元前一批次的“记忆”
                functional.reset_net(net)
                reset_counter += 1

        average_train_loss = total_train_loss / iteration_counter
        average_train_loss_thres = total_train_loss_thres / iteration_counter
        average_train_precision = total_train_precision / iteration_counter
        # scheduler.step(average_train_loss)
        print("训练集神经元重置次数:", reset_counter)  # 打印计数器的值
        print('训练集迭代次数', iteration_counter)
        print('average_train_loss=', average_train_loss)
        print('average_train_loss_thres=', average_train_loss_thres)
        print('average_train_precision=', average_train_precision)
        writer.add_scalar('average_train_loss', average_train_loss, epoch)
        writer.add_scalar('average_train_loss_thres', average_train_loss_thres, epoch)
        writer.add_scalar('average_train_precision', average_train_precision, epoch)

        # 在每个epoch结束后保存模型参数和相关信息
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        txt_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.txt')
        checkpoint_epoch = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'average_train_loss': average_train_loss,
            'average_train_precision': average_train_precision
        }
        # 保存模型参数
        torch.save(checkpoint_epoch, os.path.join(out_dir, f'checkpoint_epoch_{epoch}.pth'))

        # 更新 start_epoch 的值
        start_epoch = start_epoch + 1
        
        validation = False
        if average_train_precision > max_average_train_precision:
            max_average_train_precision = average_train_precision
            validation = True

        if validation:
            checkpoint_max = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'max_average_train_precision': average_train_precision
            }
            torch.save(checkpoint_max, os.path.join(out_dir, 'checkpoint_max.pth'))

        if validation and epoch > 10: 
            reset_counter = 0
            iteration_counter = 0
            net.eval()
            total_validation_loss = 0
            total_validation_loss_thres = 0
            total_validation_precision = 0
            with torch.no_grad():
                for time_slice in range(sample_sequence // time_sequence):
                    # 创建验证数据集与数据加载器
                    validation_dataset = Validation_Dataset(validation_data_dir, validation_label_dir, validation_label_ts_dir, time_sequence=time_sequence, start_index=0 + time_slice * time_sequence)
                    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=12)
                    for data_batch, label_batch, label_ts_batch, input_batch, datanames_batch, labelnames_batch in validation_loader:
                        # print('data_batch.shape=', data_batch.shape)
                        states = None
                        original = data_batch.to(device)
                        label = label_batch.to(device)
                        label_ts = label_ts_batch.to(device)
                        polarity_mask = input_batch.to(device)
                        results = torch.zeros_like(label_batch).to(device)
                        results_ts = torch.zeros_like(label_ts_batch).to(device)
                        for time_steps in range(time_sequence):
                            img = data_batch[:, time_steps, :, :].unsqueeze(dim=1).to(device)  # img.shape= torch.Size([batch, 1, H, W])
                            if model_name == 'EDSNN_LIF_ADAPTIVE_final':
                                membrane_potential, thres_map = net(img, states)  # membrane_potential.shape= torch.Size([batch, 1, H+4, W])
                                if states is None:
                                    states = torch.zeros_like(membrane_potential)
                                states.detach_()
                                states = membrane_potential
                                results[:, time_steps:time_steps + 1, :, :] = membrane_potential[:, :, :, :]
                                results_ts[:, time_steps:time_steps + 1, :, :] = thres_map[:, :, :, :]
                        validation_loss, validation_loss_thres, validation_signal_retain, validation_noise_removal = compute_loss(original, results, label, results_ts, label_ts)
                        postprocess(original=original, prediction=results, target=label, prediction_thres=results_ts, target_thres=label_ts, polarity_mask=polarity_mask, datanames_batch=datanames_batch, out_dir=out_dir, epoch=epoch)
                        print('Btach_validation_loss=', validation_loss)
                        print('Btach_validation_loss_thres=', validation_loss_thres)
                        print('Btach_validation_signal_retain=', validation_signal_retain)
                        print('Btach_validation_noise_removal=', validation_noise_removal)
                        total_validation_loss += validation_loss.item()
                        total_validation_loss_thres += validation_loss_thres.item()
                        total_validation_precision += (validation_signal_retain + validation_noise_removal)/2
                        iteration_counter += 1
                        functional.reset_net(net)
                        reset_counter += 1
            # 计算平均损失
            average_validation_loss = total_validation_loss / iteration_counter
            average_validation_loss_thres = total_validation_loss_thres / iteration_counter
            average_validation_precision = total_validation_precision / iteration_counter
            print('average_validation_loss=', average_validation_loss)
            print('average_validation_loss_thres=', average_validation_loss_thres)
            print('average_validation_precision=', average_validation_precision)
            writer.add_scalar('average_validation_loss', average_validation_loss, epoch)
            writer.add_scalar('average_validation_loss_thres', average_validation_loss_thres, epoch)
            writer.add_scalar('average_validation_precision', average_validation_precision, epoch)

if __name__ == '__main__':
    train_opt = parse_arguments()
    train_denoisingsnn(train_opt)
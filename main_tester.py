"""
DTSNN 测试代码
"""
import os
import re
from tqdm import tqdm
from utils.preprocess import data_load, data_crop_DTSNN
from model_zoo.test_net_DT import load_model_DT, infer_DTSNN
from utils.postprocess import data_vis_DTSNN
from utils.metric import Metric

def main():
    data_root = "data"
    save_root = "results"
    for dataset in os.listdir(data_root):
        metric.currentDataset = dataset
        metric.metric_store[dataset] = {}
        if dataset == "DVSNOISE20":
            crop = (0, 344, 0, 256)
        elif dataset == "DVSClean":
            crop = (0, 1280, 0, 720)
        elif dataset == "LED_Test":
            crop = (0, 1204, 0, 678)
        
        data_path = os.path.join(data_root, dataset)
        for scene_name in os.listdir(data_path):
            print("Now is inferring: {}-{}".format(dataset, scene_name))
            metric.currentScene = scene_name
            metric.metric_store[dataset][scene_name] = {}
            label_root = os.path.join(data_path, scene_name, "label")
            original_root = os.path.join(data_path, scene_name, "original")
            save_path = os.path.join(save_root, dataset, scene_name)
            DT(label_root, original_root, crop, save_path)
        
        metric.Calculate_Metric_All(os.path.join(save_root, dataset))

def sort_by_number(filename):
    number = re.findall(r'\d+', filename)
    if number:
        return int(number[0])
    else:
        return 0

def DTSNN(label_root, original_root, crop, save_path):
    print("Processing events denoising with DTSNN")
    metric.metric_store[metric.currentDataset][metric.currentScene]["OURS_DT"] = []
    states = None
    device, net = load_model_DT()
    for label_name in tqdm(sorted(os.listdir(label_root), key=sort_by_number)):
        label_path = os.path.join(label_root, label_name)
        original_path = os.path.join(original_root, label_name.replace("_label", ''))

        load_results = data_load(label_path, original_path, crop)
        if load_results == None:
            continue

        crop_data_OURS = data_crop_DTSNN(load_results, crop)
        states, result = infer_DTSNN(states, device, net, crop_data_OURS)
        data_vis_DTSNN(result, crop, save_path, "DT", label_name)
        metric.metric_store[metric.currentDataset][metric.currentScene]["OURS_DT"].append(metric.OURS(crop_data_OURS, result, crop))
    metric.Calculate_Metric_Scene("OURS_DT", save_path)

def DT(label_root, original_root, crop, save_path):
    print("Processing with DTSNN")
    DTSNN(label_root, original_root, crop, save_path)

if __name__ == "__main__":
    metric = Metric()
    main()
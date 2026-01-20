import numpy as np
import os
import sys
realmin = sys.float_info.min

class Metric():
    def __init__(self):
        super(Metric, self).__init__()
        self.currentDataset = None
        self.currentScene = None
        self.metric_store = {}
        self.label_table = np.array([0, 1], dtype=np.uint8)

    def AED(self, data, result, crop):
        labels = data[:, 0].astype(np.int32)
        xs = data[:, 1].astype(np.int32)
        ys = data[:, 2].astype(np.int32)
        ps = data[:, 4].astype(np.int32)
        labels = 1 - labels

        xs_result = result[:, 1].astype(np.int32)
        ys_result = result[:, 2].astype(np.int32)
        ps_result = result[:, 4].astype(np.int32)

        metric_label = np.zeros((crop[3] - crop[2], crop[1] - crop[0], 2), dtype=np.uint8)
        metric_label[ys, xs, ps] = self.label_table[labels]
        metric_output = np.zeros((crop[3] - crop[2], crop[1] - crop[0], 2), dtype=np.uint8)
        metric_output[ys_result, xs_result, ps_result] = 1

        if self.currentDataset == "DVSNOISE20":
            return self.DVSNoise20_DenoiseScore(metric_label, metric_output)
        elif self.currentDataset == "DVSClean":
            return self.DVSClean_SNR(metric_label, metric_output)

    def EDNCNN(self, data, result, crop):
        labels = data[1][0].astype(np.int32)
        xs = data[2][0][:, 0].astype(np.int32)
        ys = data[2][0][:, 1].astype(np.int32)
        ps = data[2][0][:, 2].astype(np.int32)

        xs_result = result[:, 0].astype(np.int32)
        ys_result = result[:, 1].astype(np.int32)
        ps_result = result[:, 2].astype(np.int32)

        metric_label = np.zeros((crop[3] - crop[2], crop[1] - crop[0], 2), dtype=np.uint8)
        metric_label[ys, xs, ps] = self.label_table[labels]
        metric_output = np.zeros((crop[3] - crop[2], crop[1] - crop[0], 2), dtype=np.uint8)
        metric_output[ys_result, xs_result, ps_result] = 1

        if self.currentDataset == "DVSNOISE20":
            return self.DVSNoise20_DenoiseScore(metric_label, metric_output)
        elif self.currentDataset == "DVSClean":
            return self.DVSClean_SNR(metric_label, metric_output)

    def DTSNN(self, data, result, crop):
        events = data[0]
        labels = data[1]
        labels_p = events * labels

        labels_on, labels_off = self.plority_restore(labels_p)

        label_result, data_result = result
        result_p = data_result * label_result

        result_on, result_off = self.plority_restore(result_p)

        metric_label = np.zeros((crop[3] - crop[2], crop[1] - crop[0], 2), dtype=np.uint8)
        metric_label[:, :, 0] = labels_on
        metric_label[:, :, 1] = labels_off
        metric_output = np.zeros((crop[3] - crop[2], crop[1] - crop[0], 2), dtype=np.uint8)
        metric_output[:, :, 0] = result_on
        metric_output[:, :, 1] = result_off

        if self.currentDataset == "DVSNOISE20":
            return self.DVSNoise20_DenoiseScore(metric_label, metric_output)
        elif self.currentDataset == "DVSClean":
            return self.DVSClean_SNR(metric_label, metric_output)

    def plority_restore(self, mat_p):
        mat_on = mat_p.copy()
        mat_on[mat_on == 1] = 0
        mat_on = -mat_on
        mat_off = mat_p.copy()
        mat_off[mat_off == -1] = 0
        return mat_on, mat_off

    def DVSNoise20_DenoiseScore(self, label, output):
        N = label.shape[0] * label.shape[1]
        on_data = label[:, :, 0]
        off_data = label[:, :, 1]

        sum_geq_05 = np.sum(np.log(on_data[on_data == 1]))
        sum_lt_05 = np.sum(np.log(1 - on_data[on_data == 0]))
        on_sum = sum_geq_05 + sum_lt_05

        sum_geq_05 = np.sum(np.log(off_data[off_data == 1]))
        sum_lt_05 = np.sum(np.log(1 - off_data[off_data == 0]))
        off_sum = sum_geq_05 + sum_lt_05

        logOptimalScore = (on_sum + off_sum) / N

        on_denoise = output[:, :, 0]
        off_denoise = output[:, :, 1]

        temp1 = on_data[on_denoise == 1]
        sum_geq_05 = np.sum(np.log(np.maximum(temp1, np.ones_like(temp1)*realmin)))
        temp2 = 1 - on_data[(on_data - on_denoise) == 1]
        sum_lt_05 = np.sum(np.log(np.maximum(temp2, np.ones_like(temp2)*realmin)))
        on_sum = sum_geq_05 + sum_lt_05

        temp1 = off_data[off_denoise == 1]
        sum_geq_05 = np.sum(np.log(np.maximum(temp1, np.ones_like(temp1)*realmin)))
        temp2 = 1 - off_data[(off_data - off_denoise) == 1]
        sum_lt_05 = np.sum(np.log(np.maximum(temp2, np.ones_like(temp2)*realmin)))
        off_sum = sum_geq_05 + sum_lt_05

        logDenoiseScore = (on_sum + off_sum) / N

        RPMD = logOptimalScore - logDenoiseScore  # DAVISNOISE20去噪指标结果,越低越好
        return RPMD

    def DVSClean_SNR(self, label, output):
        off_label = label[:, :, 1]  # off事件标签

        TP = np.count_nonzero(off_label * output[:, :, 1])
        FP = np.count_nonzero((1 - off_label) * output[:, :, 1])

        SNR = np.log(TP/(FP+1))  # AEDNet去噪指标结果,越高越好
        return SNR

    def Calculate_Metric_Scene(self, model, save_path):
        metric_scene = np.array(self.metric_store[self.currentDataset][self.currentScene][model])
        metric_scene_average = np.mean(metric_scene)
        message = "{}-{}-{}: {}".format(self.currentDataset, self.currentScene, model, metric_scene_average)
        print(message)
        with open(os.path.join(save_path, "metric_scene.txt"), 'a') as f:
            f.write(message + '\n')

    def Calculate_Metric_All(self, save_path):
        metric_average_store = {}

        for scene in self.metric_store[self.currentDataset].keys():
            metric_models_scene = self.metric_store[self.currentDataset][scene]
            for model in metric_models_scene.keys():
                metric_average_store[model] = []
            break

        for scene in self.metric_store[self.currentDataset].keys():
            metric_models_scene = self.metric_store[self.currentDataset][scene]
            for model in metric_models_scene.keys():
                metric_average_store[model].append(np.mean(np.array(metric_models_scene[model])))

        for model in metric_average_store.keys():
            metric_all_average = np.mean(np.array(metric_average_store[model]))
            message = "{}-{}: {}".format(self.currentDataset, model, metric_all_average)
            print(message)
            with open(os.path.join(save_path, "metric_all.txt"), 'a') as f:
                f.write(message + '\n')
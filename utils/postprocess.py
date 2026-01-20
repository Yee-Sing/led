import os
import cv2
import numpy as np

events_color = np.array([[0, 0, 255], [255, 0, 0]], dtype=np.uint8)

def get_events_frame_AED(events, crop):
    xs = events[:, 1].astype(np.int32)
    ys = events[:, 2].astype(np.int32)
    ps = events[:, 4].astype(np.int32)
    events_frame = np.ones((crop[3] - crop[2], crop[1] - crop[0], 3), dtype=np.uint8) * 255
    events_frame[ys, xs] = events_color[ps]
    return events_frame
def data_vis_AED(events, events_signal, crop, save_root, label_name):

    raw = get_events_frame_AED(events, crop)
    signal = get_events_frame_AED(events_signal, crop)
    noise = raw.copy()
    noise[signal != [255, 255, 255]] = 255

    save_path = os.path.join(save_root, "AED", label_name.replace("_label.npy", ""))
    os.makedirs(save_path, exist_ok=True)
    
    cv2.imwrite(os.path.join(save_path, "raw.png"), raw)
    cv2.imwrite(os.path.join(save_path, "signal.png"), signal)
    cv2.imwrite(os.path.join(save_path, "noise.png"), noise)

def get_events_frame_EDNCNN(events, crop):
    xs = events[:, 0].astype(np.int32)
    ys = events[:, 1].astype(np.int32)
    ps = events[:, 2].astype(np.int32)
    events_frame = np.ones((crop[3] - crop[2], crop[1] - crop[0], 3), dtype=np.uint8) * 255
    events_frame[ys, xs] = events_color[ps]
    return events_frame
def data_vis_EDNCNN(result, crop, save_root, label_name):
    labels = result[:, 3].astype(np.int32)
    events_signal = result[labels == 1]
    events_noise = result[labels == 0]

    raw = get_events_frame_EDNCNN(result, crop)
    signal = get_events_frame_EDNCNN(events_signal, crop)
    noise = get_events_frame_EDNCNN(events_noise, crop)

    save_path = os.path.join(save_root, "EDNCNN", label_name.replace("_label.npy", ""))
    os.makedirs(save_path, exist_ok=True)
    
    cv2.imwrite(os.path.join(save_path, "raw.png"), raw)
    cv2.imwrite(os.path.join(save_path, "signal.png"), signal)
    cv2.imwrite(os.path.join(save_path, "noise.png"), noise)

def get_events_frame_DTSNN(data, crop):
    events_frame = np.ones((crop[3] - crop[2], crop[1] - crop[0], 3), dtype=np.uint8) * 255
    events_frame[data == -1] = events_color[0]
    events_frame[data == 1] = events_color[1]
    return events_frame

def data_vis_DTSNN(result, crop, save_root, model_name, label_name):
    pred, data = result

    data_signal = pred * data
    data_noise = (1 - pred) * data

    raw = get_events_frame_DTSNN(data, crop)
    signal = get_events_frame_DTSNN(data_signal, crop)
    noise = get_events_frame_DTSNN(data_noise, crop)

    save_path = os.path.join(save_root, "DTSNN", model_name, label_name.replace("_label.npy", ""))
    os.makedirs(save_path, exist_ok=True)
    
    cv2.imwrite(os.path.join(save_path, "raw.png"), raw)
    cv2.imwrite(os.path.join(save_path, "signal.png"), signal)
    cv2.imwrite(os.path.join(save_path, "noise.png"), noise)
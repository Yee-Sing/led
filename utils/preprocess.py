import numpy as np

def data_load(label_path, original_path, crop):
    data = np.load(original_path, 'r')

    if data.shape[0] == 0:
        return None

    xs, ys, ps, ts = data
    retain_ids = np.where(np.logical_and(np.logical_and(crop[0] <= xs, xs < crop[1]), np.logical_and(crop[2] <= ys, ys < crop[3])))

    label_data = np.load(label_path, 'r')

    if label_data.shape[0] > label_data.shape[1]:
        label_data = np.rot90(label_data)
        
    label = label_data[:, :, 0] + label_data[:, :, 1]
    label[label != 0] = 1

    return retain_ids, xs, ys, ps, ts, label

def data_crop_AED(load_results, crop):
    retain_ids, xs, ys, ps, ts, label = load_results
    
    label = 1 - label

    xs_retain = np.expand_dims(xs[retain_ids], axis=1)
    ys_retain = np.expand_dims(ys[retain_ids], axis=1)
    ps_retain = np.expand_dims(ps[retain_ids], axis=1)
    ts_retain = np.expand_dims(ts[retain_ids] / 1000000., axis=1)
    ts_retain = np.expand_dims(ts[retain_ids], axis=1)
    label_retain = label[ys_retain.astype(np.int32), xs_retain.astype(np.int32)]

    xs_retain -= crop[0]
    ys_retain -= crop[2]

    return np.concatenate([label_retain, xs_retain, ys_retain, ts_retain, ps_retain], axis=1)

def generate_time_maps(t, x, y, p, X, Y):
    # 构造 Pos_tuple
    Pos_tuple = {}
    Pos_tuple['t'] = t[p == 1]
    Pos_tuple['x'] = x[p == 1]
    Pos_tuple['y'] = y[p == 1]

    # 构造 Neg_tuple
    Neg_tuple = {}
    Neg_tuple['t'] = t[p == 0]
    Neg_tuple['x'] = x[p == 0]
    Neg_tuple['y'] = y[p == 0]

    # 构造 Pos_CountMap 和 Neg_CountMap
    Pos_CountMap = np.zeros((Y, X))
    Neg_CountMap = np.zeros((Y, X))

    # 构造 Pos_AveTimeMap 和 Neg_AveTimeMap
    Pos_AveTimeMap = np.zeros((Y, X))
    Neg_AveTimeMap = np.zeros((Y, X))

    # 读取 ON 事件
    for j in range(len(Pos_tuple['t'])):
        row = Pos_tuple['y'][j]
        col = Pos_tuple['x'][j]

        Pos_CountMap[row, col] += 1
        time_temp = Pos_tuple['t'][j]
        Pos_AveTimeMap[row, col] += time_temp

    division = Pos_CountMap.copy()
    division[Pos_CountMap == 0] = 1
    Pos_AveTimeMap = Pos_AveTimeMap / division

    # 读取 OFF 事件
    for j in range(len(Neg_tuple['t'])):
        row = Neg_tuple['y'][j]
        col = Neg_tuple['x'][j]

        Neg_CountMap[row, col] += 1
        time_temp = Neg_tuple['t'][j]
        Neg_AveTimeMap[row, col] += time_temp

    division = Neg_CountMap.copy()
    division[Neg_CountMap == 0] = 1
    Neg_AveTimeMap = Neg_AveTimeMap / division

    return Pos_AveTimeMap, Neg_AveTimeMap

def construct_feature_vectors(center_points, Pos_AveTimeMap, Neg_AveTimeMap, region_size, X, Y):
    center_points_num = center_points.shape[0]
    feature_vectors = np.zeros((center_points_num, 2, region_size, region_size))

    center_x = center_points[:, 0]
    center_y = center_points[:, 1]

    pos_start_x = center_x - (region_size // 2)
    pos_start_x[pos_start_x < 0] = 0
    pos_start_y = center_y - (region_size // 2)
    pos_start_y[pos_start_y < 0] = 0
    pos_end_x = center_x + (region_size // 2)
    pos_end_x[pos_end_x > (X - 1)] = X - 1
    pos_end_y = center_y + (region_size // 2)
    pos_end_y[pos_end_y > (Y - 1)] = Y - 1

    neg_start_x = center_x - (region_size // 2)
    neg_start_x[neg_start_x < 0] = 0
    neg_start_y = center_y - (region_size // 2)
    neg_start_y[neg_start_y < 0] = 0
    neg_end_x = center_x + (region_size // 2)
    neg_end_x[neg_end_x > (X - 1)] = X - 1
    neg_end_y = center_y + (region_size // 2)
    neg_end_y[neg_end_y > (Y - 1)] = Y - 1

    for i in range(center_points_num):
        pos_region = Pos_AveTimeMap[pos_start_y[i]:pos_end_y[i] + 1, pos_start_x[i]:pos_end_x[i] + 1]
        nonzero_values = pos_region[pos_region > 0]
        if len(nonzero_values) > 0:
            min_nonzero = np.min(nonzero_values)
            pos_region[pos_region > 0] -= min_nonzero
        max_value = np.max(pos_region)
        if max_value > 0:
            pos_region = pos_region / max_value
        feature_vectors[i, 0, :pos_region.shape[0], :pos_region.shape[1]] = pos_region

        neg_region = Neg_AveTimeMap[neg_start_y[i]:neg_end_y[i] + 1, neg_start_x[i]:neg_end_x[i] + 1]
        nonzero_values = neg_region[neg_region > 0]
        if len(nonzero_values) > 0:
            min_nonzero = np.min(nonzero_values)
            neg_region[neg_region > 0] -= min_nonzero
        max_value = np.max(neg_region)
        if max_value > 0:
            neg_region = neg_region / max_value
        feature_vectors[i, 1, :neg_region.shape[0], :neg_region.shape[1]] = neg_region

    return feature_vectors

def data_crop_EDNCNN(load_results, crop):
    retain_ids, xs, ys, ps, ts, label = load_results

    xs = xs.astype(int)
    ys = ys.astype(int)
    ps = ps.astype(int)
    ts = ts / 1000000.
    X = np.max(xs) + 1
    Y = np.max(ys) + 1
    Pos_AveTimeMap, Neg_AveTimeMap = generate_time_maps(ts, xs, ys, ps, X, Y)

    xs_retain = np.expand_dims(xs[retain_ids], axis=1)
    ys_retain = np.expand_dims(ys[retain_ids], axis=1)
    ps_retain = np.expand_dims(ps[retain_ids], axis=1)
    center_points = np.concatenate([xs_retain, ys_retain], axis=1)
    feature_vectors = construct_feature_vectors(center_points, Pos_AveTimeMap, Neg_AveTimeMap, region_size=25, X=X, Y=Y)

    pts = np.array(center_points).T
    labels = label[pts[1], pts[0]]

    xs_retain -= crop[0]
    ys_retain -= crop[2]

    return np.array([feature_vectors, np.expand_dims(labels, axis=0), np.expand_dims(np.concatenate([xs_retain, ys_retain, ps_retain], axis=1), axis=0)], dtype=object)

def data_crop_DTSNN(load_results, crop):
    retain_ids, xs, ys, ps, ts, label = load_results

    label_use = label[crop[2]:crop[3], crop[0]:crop[1]]

    xs_retain = np.expand_dims(xs[retain_ids], axis=1)
    ys_retain = np.expand_dims(ys[retain_ids], axis=1)
    ps_retain = np.expand_dims(ps[retain_ids], axis=1)

    xs_retain -= crop[0]
    ys_retain -= crop[2]

    events_frame = np.zeros((crop[3] - crop[2], crop[1] - crop[0]))
    ps_retain[ps_retain == 0] = -1
    events_frame[ys_retain.astype(np.int32), xs_retain.astype(np.int32)] = ps_retain

    return np.concatenate([np.expand_dims(events_frame, axis=0), np.expand_dims(label_use, axis=0)], axis=0)
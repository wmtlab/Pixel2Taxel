import numpy as np


class DataProcess(object):
    def __init__(self, file):
        self.acc_sensor_data, self.aubo_pos_data, self.data_length = None, None, None
        self.update_data(file)
        _, idx_aubo_pos_data = self.get_sensor_data_idx(0)
        y = idx_aubo_pos_data[::, 3]
        self.move_length = y[-1] - y[0]
        self.w = 0.45
        
    def get_sensor_data(self, file):
        original_data = np.load(file, allow_pickle=True).item()
        acc_sensor_data = np.array(original_data["acc_sensor_data_list"])
        aubo_pos_data = np.array(original_data["aubo_pos_data_list"])
        return acc_sensor_data, aubo_pos_data

    def update_data(self, file):
        self.acc_sensor_data, self.aubo_pos_data = self.get_sensor_data(file)
        self.data_length = int(self.acc_sensor_data[:, 0].max() + 1)

    def get_sensor_data_idx(self, idx=0):
        """
        idx: represents the index of the trajectory
        """
        idx_acc_sensor_data = self.acc_sensor_data[self.acc_sensor_data[:, 0] == idx]
        idx_aubo_pos_data = self.aubo_pos_data[self.aubo_pos_data[:, 0] == idx]

        return idx_acc_sensor_data, idx_aubo_pos_data

    def get_idx_uniform_motion(self, axis="y", idx=0):
        """
        axis: the axis of movement
        idx: the index of the trajectory
        return: start_idx, end_idx: the indices for the start and end of the constant velocity motion
        """
        xyz2i = {"x": 2, "y": 3, "z": 4}
        aubo_pos_data_xyz = self.get_sensor_data_idx(idx)[1]
        data_axis = aubo_pos_data_xyz[:, xyz2i[axis]]
        gradient = np.gradient(data_axis)  # 计算y的梯度
        mid_gradient = gradient[len(gradient) // 2]
        low_gradient_idx = np.where(np.abs(gradient - mid_gradient) > 0.0001)[0]
        diff = np.diff(low_gradient_idx)
        split_index = np.where(diff > max(diff) - 1)[0].item()
        start_idx, end_idx = low_gradient_idx[split_index : split_index + 2]
        return start_idx, end_idx

    def get_idx_uniform_motion2(self, data_axis):
        """
        data_axis: 1D array representing the movement along a specific axis
        return: start_idx, end_idx: indices for the start and end of constant velocity motion
        """
        mean_data_axis = data_axis.mean()
        start_idx = np.argsort(abs(data_axis - (mean_data_axis + self.w * self.move_length)))[0]
        end_idx = np.argsort(abs(data_axis - (mean_data_axis - self.w * self.move_length)))[0]
        start_idx, end_idx = min(start_idx, end_idx), max(start_idx, end_idx)
        return start_idx, end_idx

    def idx_time_in_sequence(self, time, sequence):
        """
        time: represents the moment in time
        sequence: represents a time sequence
        return: idx represents the index of the moment in the time sequence
        """
        idx = np.argmin(np.abs(time - sequence))
        return idx

    def get_idx_pos2acc(self, idx):
        idx_acc_sensor_data, idx_aubo_pos_data = self.get_sensor_data_idx(idx)

        idx_y_axis_data = idx_aubo_pos_data[:, 3]
        idx_z_acc_data = idx_acc_sensor_data[:, 4]
        idx_aubo_time_sequence = idx_aubo_pos_data[:, 1]
        idx_acc_time_sequence = idx_acc_sensor_data[:, 1]

        start_idx, end_idx = self.get_idx_uniform_motion2(idx_y_axis_data)
        start_idx, end_idx = start_idx + 2, end_idx - 2
        start_time, end_time = idx_aubo_time_sequence[[start_idx, end_idx]]

        # Fit a line
        slope, intercept = np.polyfit(
            idx_aubo_time_sequence[start_idx : end_idx + 1],
            idx_y_axis_data[start_idx : end_idx + 1],
            1)

        start_acc_idx = self.idx_time_in_sequence(start_time, idx_acc_time_sequence)
        end_acc_idx = self.idx_time_in_sequence(end_time, idx_acc_time_sequence)

        pos2acc_ = []
        for i in range(start_acc_idx, end_acc_idx + 1):
            t = idx_acc_time_sequence[i]
            pos2acc_.append([t, slope * t + intercept, idx_z_acc_data[i]])  # time, position, acceleration

        return np.array(pos2acc_)

    def get_pos2acc(self, sort=True, reversed=False):
        """
        sort: indicates if the data should be sorted, default is True
        reversed: default is False (ascending order). It is only effective when sort=True.
        """
        pos2acc_arr = []
        for i in range(self.data_length):
            pos2acc_arr.append(self.get_idx_pos2acc(i)[:, 1:])  #don't need time, only position and acceleration
        pos2acc_arr = np.concatenate(pos2acc_arr, axis=0)
        if sort:
            pos2acc_arr = self.sort_arr(pos2acc_arr, 0, reversed)

        return pos2acc_arr

    def sort_arr(self, arr, axis=0, reversed=False):
        """
        arr: a 2D array with n rows and m columns, n is the length of the array
        axis: represents the axis along which to sort, default is 0, indicating sorting along the first axis
        reversed: default is False (ascending order)
        """
        sort_index = np.argsort(arr[:, axis])
        if reversed:
            sort_index = sort_index[::-1]

        return arr[sort_index]

    def get_acc_patch(self, interval=0.002, move_distance=0.1, start_pos=None):
        """
        interval: represents the interval, default is 2 millimeters
        move_distance: represents the distance to move, default is 100 millimeters
        start_pos: represents the starting position, if not provided, the starting position is taken from pos2acc
        """
        pos, acc = self.get_pos2acc(sort=True, reversed=False).T  # position and acceleration
        acc = abs(acc - 1)  # remove gravity acceleration and take absolute value
        if start_pos is None:
            start_pos = pos[0]
        end_pos = start_pos + 2*self.w * self.move_length

        pos_patch, acc_patch = [], []
        for i in np.arange(start_pos, end_pos, interval):
            lower_bound = i
            upper_bound = i + interval
            idx = np.where((pos >= lower_bound) & (pos <= upper_bound))[0]
            if len(idx) != 0:
                pos_patch.append([lower_bound, upper_bound])
                acc_patch.append(acc[idx].mean())
            else:
                break

        return np.array(pos_patch), np.array(acc_patch)
    
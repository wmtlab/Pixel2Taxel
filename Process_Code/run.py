from src.CaliBrateCamera import CaliBrateCamera
from src.DataProcess import DataProcess
import argparse
import numpy as np
import cv2
import json
import os

def load_idx2name(name_file):
    with open(name_file) as f:
        idx2name = json.load(f)
        f.close()
    return idx2name


def check_required_folder(base_folder, name):
    for f in ["out", "out/crop", "out/transform", "out/{}".format(name)]:
        folder = os.path.join(base_folder, f)
        if not os.path.exists(folder):
            os.makedirs(folder)


def get_file_path(folder):
    for file in os.listdir(folder):
        if file[:3] == "img":
            image_file = os.path.join(folder, file)
        elif file[:3] == "pos":
            calibrate_robot_file = os.path.join(folder, file)  # coordinate of robot
        elif file[:13] == "select_points":
            calibrate_pixel_file = os.path.join(folder, file)  # coordinate of pixel
        elif file == "sensor_data":
            sensor_data_root_file = os.path.join(folder, file)
        elif file == "out":
            out_data_root_file = os.path.join(folder, file)
        elif file[-8:] == "crop.txt":
            crop_location_file = os.path.join(folder, file)
    return image_file, calibrate_robot_file, calibrate_pixel_file, sensor_data_root_file, out_data_root_file, crop_location_file


def get_project_image_points(pos_patch, aubo_x, calibrate):
    pos_patch_reshape = pos_patch.reshape(-1,)  # 50x2->100
    world_points = []
    for i in range(pos_patch_reshape.shape[0]):
        world_points.append([aubo_x, pos_patch_reshape[i], 0])
    world_points = np.array(world_points).astype(np.float32)  # the input of opencv must be float32
    project_image_points = calibrate.pose2pixel(world_points)
    project_image_points[:, 1] = project_image_points[:, 1].mean()   # the error of y is very small, so take the mean of y directly
    project_image_points = np.round(project_image_points).astype(np.int32) 
    project_image_points = project_image_points.reshape(-1, 2, 2)  # nx2x2
    return project_image_points


def get_crop_location(crop_location_file):
    left_top = []
    right_bottom = []
    with open(crop_location_file, 'r') as f:
        re = f.read()
        re = re.replace(';', ',').replace('\n', ',').split(',')
        left_top.append(max(int(re[0]), int(re[4])))
        left_top.append(max(int(re[1]), int(re[3])))
        right_bottom.append(min(int(re[2]), int(re[6])))
        right_bottom.append(min(int(re[5]), int(re[7])))
    return left_top, right_bottom


def get_crop_result(src, left_top, right_bottom):
    target = src[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]
    return target


def get_warpPerspective_result(src, M):
    # interpolation = cv2.INTER_NEAREST
    if len(src) == 1:
        return cv2.warpPerspective(src[0], M, (src[0].shape[1], src[0].shape[0] * 2))
    target = []
    for i in range(len(src)):
        target.append(cv2.warpPerspective(src[i], M, (src[i].shape[1], src[i].shape[0] * 2)))
    return target


def gradient_color(alpha):
    """
    :param alpha: 0~1, 0:start color, 1:end color.
    :return: gradient color(RGB)
    """
    # BGR
    start = np.array([0, 255, 0])
    end = np.array([255, 0, 0])
    max_distance = end - start
    return start + alpha * max_distance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", "-r", type=str, default="Raw_Data", help="the root Directory of Raw_Data")
    parser.add_argument("--index", "-i", type=int, default=1, choices=range(1, 9), help="The Data Index for Processing")
    arg = parser.parse_args()

    name = load_idx2name(os.path.join(arg.root, "name.json"))[str(arg.index)]
    if arg.index == 2 or arg.index == 3:
        perspective_transform = "PerspectiveTransform1.npy"
    else:
        perspective_transform = "PerspectiveTransform2.npy" 
    
    folder = os.path.join(arg.root, str(arg.index))
    check_required_folder(folder,name)
    image_file, calibrate_robot_file, calibrate_pixel_file,\
    sensor_data_root_file, out_data_root_file, crop_location_file = get_file_path(folder)

    calibrate = CaliBrateCamera(calibrate_robot_file, calibrate_pixel_file)
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    
    acc_patch_all = []
    project_image_points_all = []
    min_acc, max_acc = np.inf, -np.inf
    for file in os.listdir(sensor_data_root_file):
        dataprocess = DataProcess(os.path.join(sensor_data_root_file, file))
        _, idx_aubo_pos_data = dataprocess.get_sensor_data_idx(0)
        aubo_x = idx_aubo_pos_data[::, 2][100:-100].mean()
        pos_patch, acc_patch = dataprocess.get_acc_patch(interval=0.001)
        acc_patch_all.append(acc_patch)
        cur_min_acc, cur_max_acc = acc_patch.min(), acc_patch.max()
        if cur_min_acc < min_acc:
            min_acc = cur_min_acc
        if cur_max_acc > max_acc:
            max_acc = cur_max_acc
        project_image_points = get_project_image_points(pos_patch, aubo_x, calibrate)
        project_image_points_all.append(project_image_points)
        
    print("mean: ", np.mean(np.concatenate(acc_patch_all, axis=0)))
    print("scale: ", max_acc - min_acc)
    print("std: ", np.concatenate(acc_patch_all, axis=0).std())

    transparency = 0.8  # transparency, higher means more transparent
    acc_matrix = np.zeros_like(image)[..., 0].astype(np.float32)
    label = image.copy()
    for acc_patch, project_image_points in zip(acc_patch_all, project_image_points_all):
        for i in range(len(project_image_points)):
            x, y = project_image_points[i][0]
            x2, y2 = project_image_points[i][1]
            color = gradient_color((acc_patch[i] - min_acc) / (max_acc - min_acc))
            cv2.rectangle(label, (x, y - 4), (x2, y2 + 4), color, -1)
            acc_matrix[y - 4 : y2 + 4, x2:x] = acc_patch[i]
    image_label = cv2.addWeighted(label, 1 - transparency, image, transparency, 0)
    acc_matrix = acc_matrix * (1 / acc_matrix.max())

    #show the result of raw result
    cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imshow("image_label", cv2.cvtColor(image_label, cv2.COLOR_RGB2BGR))
    cv2.imshow("label", cv2.cvtColor(label, cv2.COLOR_RGB2BGR))
    cv2.imshow("acc_matrix", acc_matrix)
    key = cv2.waitKey(0)
    if key == ord("q") or key == ord("Q") or key == 27:
        cv2.destroyAllWindows()

    #show the result of warpPerspective
    M = np.load(os.path.join(arg.root, perspective_transform))
    transform_image, transform_label, transform_image_label, transform_acc_matrix =\
          get_warpPerspective_result([image, label, image_label, acc_matrix], M)
    cv2.imshow("transform_image", cv2.cvtColor(transform_image, cv2.COLOR_RGB2BGR))
    cv2.imshow("transform_label", cv2.cvtColor(transform_label, cv2.COLOR_RGB2BGR))
    cv2.imshow("transform_image_label", cv2.cvtColor(transform_image_label, cv2.COLOR_RGB2BGR))
    cv2.imshow("transform_acc_matrix", transform_acc_matrix)
    key = cv2.waitKey(0)
    if key == ord("q") or key == ord("Q") or key == 27:
        cv2.destroyAllWindows()
    #save the result of warpPerspective
    transform_folder = os.path.join(folder, "out", "transform")
    cv2.imwrite(os.path.join(transform_folder, "transform_image.png"),cv2.cvtColor(transform_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(transform_folder, "transform_image_label.png"),cv2.cvtColor(transform_image_label, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(transform_folder, "transform_label.png"),cv2.cvtColor(transform_label, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(transform_folder, "transform_acc_matrix.png"),cv2.cvtColor(transform_acc_matrix * 255, cv2.COLOR_RGB2BGR))
    print("Succeed! The result of warpPerspective is in the folder: ", transform_folder)

    #show the result of crop
    left_top, right_bottom = get_crop_location(crop_location_file)
    crop_image = get_crop_result(transform_image, left_top, right_bottom)
    crop_label = get_crop_result(transform_label, left_top, right_bottom)
    crop_image_label = get_crop_result(transform_image_label, left_top, right_bottom)
    crop_acc_matrix = get_crop_result(transform_acc_matrix, left_top, right_bottom)
    cv2.imshow("crop_image", cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))
    cv2.imshow("crop_label", cv2.cvtColor(crop_label, cv2.COLOR_RGB2BGR))
    cv2.imshow("crop_image_label", cv2.cvtColor(crop_image_label, cv2.COLOR_RGB2BGR))
    cv2.imshow("crop_acc_matrix", crop_acc_matrix)
    key = cv2.waitKey(0)
    if key == ord("q") or key == ord("Q") or key == 27:
        cv2.destroyAllWindows()
    #save the result of crop
    crop_folder = os.path.join(folder, "out", "crop")
    cv2.imwrite(os.path.join(crop_folder, "crop_image.png"),cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(crop_folder, "crop_label.png"),cv2.cvtColor(crop_label, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(crop_folder, "crop_image_label.png"),cv2.cvtColor(crop_image_label, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(crop_folder, "crop_acc_matrix.png"),cv2.cvtColor(crop_acc_matrix * 255, cv2.COLOR_RGB2BGR))
    print("Succeed! The result of crop is in the folder: ", transform_folder)
    #final result
    result_folder = os.path.join(folder, "out", name)
    cv2.imwrite(os.path.join(result_folder,"{}.png".format(name)),cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(result_folder,"{}_acc.png".format(name)),cv2.cvtColor(crop_acc_matrix * 255, cv2.COLOR_RGB2BGR))
    print("Succeed! The final result is in the folder: ", result_folder)

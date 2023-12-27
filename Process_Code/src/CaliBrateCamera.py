import numpy as np
import cv2
import csv


class CaliBrateCamera():
    def __init__(self, csv_file, txt_file):
        # Calibration
        csv_file_r = open(csv_file, 'r')
        csv_reader = csv.reader(csv_file_r)
        next(csv_reader)
        world_points = []
        for row in csv_reader:
            world_points.append([float(row[0]),float(row[1]), 0])
        csv_file_r.close()
        world_points = [np.array(world_points).astype(np.float32)]

        f = open(txt_file, 'r')
        image_points = []
        for row in f.readlines():
            row = row.strip('\n').split(',')
            image_points.append([float(row[0]),float(row[1])])
        image_points = [np.array(image_points, dtype=np.float32)]

        retval, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(
            world_points, image_points, (1920,1080), None, None
        )
        self.rvecs = rvecs[0]
        self.tvecs = tvecs[0]
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients

        project_image_points, _ = cv2.projectPoints(world_points[0], rvecs[0], tvecs[0], camera_matrix, distortion_coefficients)
        project_image_points = project_image_points.reshape(-1,2)
        error = np.mean(abs(project_image_points-image_points)/image_points)*1e2
        print(f'Relative error after calibration: {round(error, 4)}%')

    def pose2pixel(self, world_points):
        project_image_points, _ = cv2.projectPoints(world_points, self.rvecs, self.tvecs, self.camera_matrix, self.distortion_coefficients)
        project_image_points = project_image_points.reshape(-1,2)
        return project_image_points



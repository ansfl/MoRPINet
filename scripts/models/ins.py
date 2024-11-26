from typing import Union

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.integrate as integrate
import matplotlib.pyplot as plt

from scripts.configs.config import Config
from scripts.data_loader import sync_imu_dir_path, sync_rtk_dir_path, raw_imu_dir_path
from scripts.utils.test_utils import interpolate_gt
from scripts.utils.utils import moving_average, load_data


class INS:
    def __init__(self, config: Config):
        self.config = config

    def run_ins(self,
                task: tuple,
                use_three_dim: bool = True,
                graph: tuple = (False, False, False)):
        # load data
        try:
            recording_df, gt_df, task_len = load_data(imu_path=sync_imu_dir_path / f'IMU_{task[1]}.npy',
                                                      gt_path=sync_rtk_dir_path / f'ned_{task[1].split("_")[0]}.npy')
        except FileNotFoundError:
            print("ERROR: couldn't load data")
            exit()
        # load raw data for calibration
        try:
            raw_df = pd.read_csv(raw_imu_dir_path / f'raw_imu_{task[1]}.csv', skiprows=1)
        except FileNotFoundError:
            print('missing raw imu file for testing INS - calibration will be skipped')
            raw_df = None

        recording_df = recording_df[:gt_df.shape[0]*self.config.imu_rtk_ratio, :]

        # get initial angles
        psi0 = np.arctan2(gt_df[1, 0], gt_df[1, 1])
        if recording_df.shape[1] > 6:
            psi0 = recording_df[0, -1]
        if psi0 < 0:  # change yaw to range [0, 2pi]
            psi0 = psi0 + 2 * np.pi
        init_angles = (0, 0, psi0)

        reconstruct_pos, gt_interp = self.data_processing(imu=recording_df,
                                                          rtk=gt_df,
                                                          raw_imu=raw_df,
                                                          use_three_dim=use_three_dim,
                                                          initial_angles=init_angles,
                                                          graph=graph)

        gt_interp = gt_interp[:reconstruct_pos.shape[0], :]
        steps_err = np.sqrt(np.sum((reconstruct_pos[:, :2] - gt_interp[:, :2]) ** 2, axis=1))
        mean_pos_err = np.sqrt(np.mean(steps_err ** 2))

        duration = len(gt_df) / self.config.rtk_freq

        return reconstruct_pos, gt_interp, mean_pos_err, duration

    def data_processing(self,
                        imu: np.ndarray,
                        raw_imu: pd.DataFrame = None,
                        rtk: np.ndarray = None,
                        initial_angles: Union[tuple, list, np.ndarray] = (0, 0, 0),
                        use_three_dim: bool = True,
                        graph: tuple = (False, False, False)):

        sf_table = imu[:, :3]
        gyro_table = imu[:, 3:]
        if raw_imu is not None:
            raw = pd.DataFrame.to_numpy(raw_imu)
            sf_raw = raw[:, 5:8].astype(float)
            gyro_raw = raw[:, 8:-1].astype(float)
        else:
            sf_raw = None
            gyro_raw = None

        # process the specific force and gyro data
        f = self.accelerometer_processing(sf_table, sf_raw, use_three_dim, graph[0])
        c = self.gyro_processing(gyro_table, gyro_raw, initial_angles, use_three_dim, graph[1])

        # calculate the acceleration, velocity and position in inertial system
        pos, vel, acc = self.position_calculation(specific_force=f,
                                                  transform_mat=c,
                                                  rtk=rtk,
                                                  graph=graph[2])

        # do interpolation to GT data
        gt_interp = interpolate_gt(rtk, rtk_freq=self.config.rtk_freq, new_freq=self.config.imu_freq)

        return pos[:, :2], gt_interp

    def accelerometer_processing(self,
                                 f: np.ndarray,
                                 f_raw: np.ndarray = None,
                                 use_three_dim: bool = True,
                                 graph: bool = False):
        # noise cancellation
        if self.config.do_lpf:
            f = moving_average(f, 20)

        # calibration (zero order)
        if self.config.do_ins_calibration and f_raw is not None:
            bias = self.config.calibration_func(f_raw,
                                                start_time=3,
                                                duration=3,
                                                frequency=self.config.imu_freq,
                                                f_z_axis_index=2,
                                                g=self.config.g)
            f = f - bias

        f[:, 2] = f[:, 2] * int(use_three_dim)
        f[np.abs(f) < 2e-13] = 0

        if graph:
            plt.figure()
            plt.plot(np.arange(0, f.shape[0] / self.config.imu_freq, 1 / self.config.imu_freq), f[:, 0], label=r'$f_x$')
            plt.plot(np.arange(0, f.shape[0] / self.config.imu_freq, 1 / self.config.imu_freq), f[:, 1], label=r'$f_y$')
            plt.plot(np.arange(0, f.shape[0] / self.config.imu_freq, 1 / self.config.imu_freq), f[:, 2], label=r'$f_z$')
            plt.xlabel('Time [s]')
            plt.ylabel(r'Specific Force [$m/s^2$]')
            plt.legend()
            plt.grid(True)

        return f

    def gyro_processing(self,
                        w: np.ndarray,
                        w_raw: np.ndarray = None,
                        initial_angles: Union[tuple, list, np.ndarray] = (0, 0, 0),
                        use_three_dim: bool = True,
                        graph: bool = False):

        if self.config.gyro_in_degrees:
            w[:, :3] = np.radians(w[:, :3])
            if w_raw is not None:
                w_raw = np.radians(w_raw)

        # sampling rate
        tau = 1 / self.config.imu_freq

        # initial angles
        theta = initial_angles[0] * int(use_three_dim)
        phi = initial_angles[1] * int(use_three_dim)
        psi = initial_angles[2]

        # noise cancellation
        if self.config.do_lpf:
            w = moving_average(w, 20)
        # calibration
        if self.config.do_ins_calibration and w_raw is not None:
            bias = self.config.calibration_func(w_raw,
                                                start_time=3,
                                                duration=3,
                                                frequency=self.config.imu_freq,
                                                f_z_axis_index=None,
                                                g=self.config.g)
            w = w - bias

        w[:, :2] = w[:, :2] * int(use_three_dim)
        w[np.abs(w) < 2e-13] = 0

        # initial C matrix
        c = np.zeros((w.shape[0] + 1, 3, 3))
        c[0] = np.array([[np.cos(psi), -np.sin(psi), 0],
                               [np.sin(psi),   np.cos(psi), 0],
                               [0,             0,           1]], dtype=float)

        for t in range(1, w.shape[0] + 1):
            c_prev = c[t - 1]
            omega = np.array([[0,            -w[t - 1, 2],  w[t - 1, 1]],
                              [w[t - 1, 2],       0,       -w[t - 1, 0]],
                              [-w[t - 1, 1],  w[t - 1, 0],        0]])
            c[t] = np.matmul(c_prev, la.expm(omega * tau))

        if graph:
            plt.figure()
            plt.plot(np.arange(0, w.shape[0] / self.config.imu_freq, 1 / self.config.imu_freq), w[:, 0], label=r'$\omega_x$')
            plt.plot(np.arange(0, w.shape[0] / self.config.imu_freq, 1 / self.config.imu_freq), w[:, 1], label=r'$\omega_y$')
            plt.plot(np.arange(0, w.shape[0] / self.config.imu_freq, 1 / self.config.imu_freq), w[:, 2], label=r'$\omega_z$')
            plt.ylabel('Angular Velocity [m/s]')
            plt.xlabel('Time [s]')
            plt.legend()
            plt.grid(True)

        return c

    def position_calculation(self,
                             specific_force: np.ndarray,
                             transform_mat: np.ndarray,
                             rtk: np.ndarray = None,
                             graph: bool = 0):
        # sampling rate
        tau = 1 / self.config.imu_freq
        # gravity
        g_vec = np.array([0, 0, self.config.g])

        initial_pos = rtk[0, :3]
        if rtk.shape[1] > 3:
            initial_vel = rtk[0, 3:6]
        else:
            initial_vel = (rtk[1, :] - rtk[0, :]) * self.config.rtk_freq
            initial_vel[2] = 0

        # transfer specific force to navigation coordinates
        acc = np.zeros_like(specific_force)
        for t in range(1, specific_force.shape[0] + 1):
            acc[t - 1] = np.matmul(transform_mat[t-1].T, specific_force[t - 1, :]) + g_vec

        # calculate the velocity
        vel = integrate.cumtrapz(acc, dx=tau, initial=0, axis=0) + initial_vel

        # calculate the position
        pos = integrate.cumtrapz(vel, dx=tau, initial=0, axis=0) + initial_pos

        if graph:
            plt.figure()
            plt.plot(pos[:, 1], pos[:, 0], label='received course INS')
            plt.plot((rtk[:, 1]), (rtk[:, 0]), 'k', label='ground truth')
            plt.xlabel('East [m]')
            plt.ylabel('North [m]')
            plt.legend()
            plt.axis('equal')
            plt.grid(True)

        return pos, vel, acc

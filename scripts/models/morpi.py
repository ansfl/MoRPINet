import webbrowser
from typing import Optional

import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt

from scripts.configs.config import Config
from scripts.data_loader import raw_imu_dir_path, sync_imu_dir_path, sync_rtk_dir_path
from scripts.models import morpi_train_imu_dir, morpi_train_rtk_dir
from scripts.models.ins import INS
from scripts.utils.test_utils import interpolate_gt, dead_reckoning
from scripts.utils.utils import load_data, moving_average


class MoRPI:
    def __init__(self, config: Config):
        self.config = config
        self.ins = INS(self.config)

    def run_morpi(self,
                  task: Optional[tuple] = None,
                  data_type: str = 'sf',
                  graph: tuple = (False, False, False)):
        # get gain
        weinberg_gain = self.find_gain(data_type, graph=False)

        #  load data
        try:
            recording_df, gt_df, task_len = load_data(imu_path=sync_imu_dir_path / f'IMU_{task[1]}.npy',
                                                      gt_path=sync_rtk_dir_path / f'ned_{task[1].split("_")[0]}.npy')
        except FileNotFoundError:
            print("ERROR: couldn't load data")
            exit()

        # load raw data for calibration
        try:
            raw_df = pd.read_csv(raw_imu_dir_path / f'raw_imu_{task[1]}.csv', skiprows=1)
            raw_df = pd.DataFrame.to_numpy(raw_df)[:, 5:-1].astype(float)
        except FileNotFoundError:
            print('missing raw imu file for testing MoRPI - calibration will be skipped')
            raw_df = None

        # get initial angles
        psi0 = np.arctan2(gt_df[self.config.window_size, 0], gt_df[self.config.window_size, 1])
        if psi0 < 0:  # change yaw to range [0, 2pi]
            psi0 = psi0 + 2 * np.pi
        init_angles = (0, 0, psi0)

        reconstruct_pos, gt_interp = self.data_processing(imu=recording_df,
                                                          rtk=gt_df,
                                                          raw_imu=raw_df,
                                                          data_type=data_type,
                                                          gain=weinberg_gain,
                                                          initial_angles=init_angles,
                                                          graph=graph)

        steps_err = np.sqrt(np.sum((reconstruct_pos[:, :2] - gt_interp[:, :2]) ** 2, axis=1))
        mean_pos_err = np.sqrt(np.mean(steps_err ** 2))

        return reconstruct_pos, gt_interp, mean_pos_err,

    def find_gain(self, data_type: str = 'sf', graph: bool = False):
        files = []
        for mission in self.config.MoRPI_train_missions:
            files.append([morpi_train_imu_dir / f'IMU_{mission}.npy', morpi_train_rtk_dir / f'ned_{mission}.npy'])

        gain = 0
        train_counter = 0
        duration = 0
        for track in files:
            train_df, gt_df, true_len = load_data(imu_path=track[0], gt_path=track[1])
            train_counter += 1
            data, peaks = self.data_extracting(imu=train_df,
                                               data_type=data_type,
                                               graph=graph)

            gain += self.calculate_gain(data, peaks, true_len)

            duration += len(gt_df) / self.config.rtk_freq

        if train_counter:
            gain = gain / train_counter
        if __name__ == '__main__':
            print(f'final gain: {gain}')

        self.config.data_to_file.add_to_res_file(id_key='morpi',
                                                 msg=f'MoRPI-{data_type} gain train total duration (one IMU)',
                                                 data=duration)

        return gain

    def data_extracting(self,
                        imu: np.ndarray,
                        data_type: str = 'sf',
                        min_val_between_peaks: float = None,
                        graph: bool = False):

        if data_type == 'gyro':
            main_axis = imu[:, 5]
            if self.config.gyro_in_degrees:
                main_axis = np.radians(main_axis)
            h = 0.25
            p = 0.3
            d = 110
            t = None
            w = None
            r_h = 0.5
            smooth = 20
        else:  # data_type == 'sf'
            main_axis = imu[:, 1]
            h = None
            p = 0.9
            d = 100
            t = None
            w = 65
            r_h = 1
            smooth = 40

        if self.config.do_lpf:
            main_axis = moving_average(main_axis, smooth)
        main_axis = np.array(main_axis)

        periodic_peaks = signal.find_peaks(main_axis[:],
                                           distance=d,
                                           prominence=p,
                                           height=h,
                                           threshold=t,
                                           width=w,
                                           rel_height=r_h)[0]

        if min_val_between_peaks is not None:
            valid_peaks = []
            flag_no_min_val = False  # flag to indicate if peak p need to be added to valid peaks list
            # Check the condition for successive peaks
            for p in range(len(periodic_peaks) - 1):
                # Check if there is any value smaller than -0.5 between the current peak and the next peak
                if np.any(main_axis[periodic_peaks[p]:periodic_peaks[p + 1]] < min_val_between_peaks):
                    if not flag_no_min_val:
                        valid_peaks.append(periodic_peaks[p])
                    else:
                        flag_no_min_val = False
                    valid_peaks.append(periodic_peaks[p + 1])
                else:  # the two peaks belongs to the same period
                    if main_axis[periodic_peaks[p+1]] > main_axis[periodic_peaks[p]]:  # add the bigger peak value
                        if len(valid_peaks) > 0:
                            valid_peaks[-1] = periodic_peaks[p+1]
                        else:
                            valid_peaks.append(periodic_peaks[p + 1])
                    flag_no_min_val = True  # in the next iteration the peak p (now its p+1) not need to be added

            # Remove duplicates (since pairs of peaks might be added multiple times)
            periodic_peaks = np.unique(np.array(valid_peaks))

        periodic_peaks = np.insert(periodic_peaks, 0, 0)
        periodic_peaks = np.append(periodic_peaks, len(main_axis) - 1)

        if graph:
            plt.figure()
            label_peaks = r'$\omega_z$' if data_type == 'gyro' else r'$f_y$'
            label_yaxis = 'Angular Velocity [m/s]' if data_type == 'gyro' else r'Specific Force [$m/s^2$]'
            t = np.arange(0, main_axis.shape[0] / self.config.imu_freq, 1 / self.config.imu_freq)
            plt.plot(t, main_axis, label=label_peaks)
            plt.plot(t[periodic_peaks], main_axis[periodic_peaks], 'kx')
            plt.ylabel(label_yaxis)
            plt.xlabel('Time [s]')
            plt.legend()
            plt.grid(True)

        return main_axis, periodic_peaks

    @staticmethod
    def calculate_gain(data, peaks_idx, distance):
        extract_gain = distance / np.sum(
            [np.power((np.max(data[peaks_idx[n]:peaks_idx[n + 1]]) - np.min(data[peaks_idx[n]:peaks_idx[n + 1]])), 0.25)
             for n in range(len(peaks_idx) - 1)])

        return extract_gain

    def data_processing(self,
                        imu: np.ndarray,
                        rtk: np.ndarray,
                        raw_imu: np.ndarray,
                        gain: float,
                        data_type: str = 'sf',
                        initial_angles: tuple = (0, 0, 0),
                        graph: tuple = (False, False, False)):

        data, peaks = self.data_extracting(imu=imu,
                                           data_type=data_type,
                                           graph=graph[0])

        peaks = peaks[1:]
        psi, delta_psi = self.calculate_heading(imu=imu,
                                                raw_imu=raw_imu,
                                                initial_angles=initial_angles,
                                                samples=peaks,
                                                graph=graph[1])

        steps = [self.calculate_step(data[peaks[n]:peaks[n + 1]], gain) for n in range(len(peaks) - 1)]

        gt_intrp = interpolate_gt(rtk=rtk, rtk_freq=self.config.rtk_freq, new_freq=self.config.imu_freq)
        inrtplt_samples = np.vstack([gt_intrp[peaks[n], :]for n in range(len(steps))])
        inrtplt_samples = np.insert(inrtplt_samples, 0, [0., 0.], axis=0)

        reconstruct_pos, reconstruct_psi = self.get_morpi_trajectory(steps=steps,
                                                                     delta_psi=delta_psi,
                                                                     peaks=peaks,
                                                                     psi=psi,
                                                                     initial_angles=initial_angles)

        if graph[2]:
            plt.figure()
            plt.plot(reconstruct_pos[:, 0], reconstruct_pos[:, 1], label='reconstruct trajectory')
            plt.plot(reconstruct_pos[:, 0], reconstruct_pos[:, 1], 'o')
            plt.plot((rtk[:, 1]), (rtk[:, 0]), 'k', label='ground truth')
            plt.plot(gt_intrp[:, 1], gt_intrp[:, 0], label='interpolate gt')
            plt.plot(inrtplt_samples[:, 1], inrtplt_samples[:, 0], 'o')
            plt.axis('equal')
            plt.legend()
            plt.xlabel('East [m]')
            plt.ylabel('North [m]')
            plt.grid(True)

        return reconstruct_pos, inrtplt_samples

    def calculate_heading(self,
                          imu: np.ndarray,
                          raw_imu: np.ndarray = None,
                          initial_angles: tuple = (0, 0, 0),
                          samples: np.ndarray = None,
                          graph: bool = False):

        w = imu[:, 3:].copy()
        if raw_imu is not None:
            w_raw = raw_imu[:, 3:]
            w_raw[:, [0, 2]] *= -1  # transform sensor coordinate to body framework
        else:
            w_raw = None

        rotation_matrix = self.ins.gyro_processing(w, w_raw=w_raw, initial_angles=initial_angles)

        psi = np.arctan2(rotation_matrix[:, 1, 0], rotation_matrix[:, 0, 0])

        if samples is not None:
            delta_psi = psi[samples[1:]] - psi[samples[:-1]]
            # delta_ang = angle[samples[1:]] - angle[samples[:-1]]

            delta_psi = np.array([self.convert_large_angles(angle) for angle in delta_psi])
        else:
            delta_psi = None
            # delta_ang = None

        if graph:
            plt.figure()
            t = np.arange(0, w.shape[0] / self.config.imu_freq, 1 / self.config.imu_freq)
            # plt.plot(t, np.degrees(angle), label='integral')
            plt.plot(t, np.degrees(psi[:-1]), label='C matrix')
            if samples is not None:
                plt.plot(t[samples], np.degrees(psi[samples]), 'o', label='psi samples')
                # plt.plot(t[samples], np.degrees(angle[samples]), 'o', label='ang samples')
            plt.legend()
            plt.xlabel('Time [s]')
            plt.ylabel('Degrees')
            plt.grid(True)
            plt.draw()
        return psi, delta_psi

    @staticmethod
    def convert_large_angles(ang):
        if ang > np.pi:
            new_ang = ang % -np.pi
        elif ang < -np.pi:
            new_ang = ang % np.pi
        else:
            return ang
        return new_ang

    @staticmethod
    def calculate_step(data, gain):
        step = gain * np.power((np.max(data) - np.min(data)), 0.25)
        return step

    def get_morpi_trajectory(self,
                             steps: list,
                             delta_psi: np.ndarray,
                             peaks: np.ndarray,
                             psi: np.ndarray,
                             initial_angles: tuple):
        # init arrays
        reconstruct_pos = np.zeros((len(steps) + 1, 2))
        update_deg = initial_angles[-1]
        psi_arr = [update_deg.item()]

        for k in range(1, len(steps) + 1):
            # assuming trajectory made of straight lines, the angles are following the line
            if k > 1:
                up_delta_psi = delta_psi[k - 1] + update_deg
            else:
                up_delta_psi = update_deg  # first step heading equal to initial heading

            # check if there was a turn in the trajectory during the current step
            if np.degrees(delta_psi[k - 1]) > self.config.morpi_turn_threshold:
                update_deg = np.max(psi[peaks[k - 1]:peaks[k] - 1])
            elif np.degrees(delta_psi[k - 1]) < -self.config.morpi_turn_threshold:
                update_deg = np.min(psi[peaks[k - 1]:peaks[k] - 1])

            psi_arr.append(up_delta_psi.item())
            reconstruct_pos[k] = dead_reckoning(reconstruct_pos[k - 1], steps[k - 1], up_delta_psi)

        return reconstruct_pos, psi_arr

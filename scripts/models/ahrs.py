import numpy as np
import imufusion

from scripts.configs.config import Config


class AHRS:
    def __init__(self, config: Config):
        self.config = config

    def calculate_heading(self,
                          imu: np.ndarray,
                          raw_imu: np.ndarray = None,
                          initial_angles: tuple = (0, 0, 0),
                          samples: np.ndarray = None):

        # Process sensor data
        accelerometer = imu[:, :3]
        gyroscope = imu[:, 3:]
        init_yaw = initial_angles[-1]

        # Instantiate algorithms
        start_time = 3
        duration = 3
        start_sample = start_time * self.config.imu_freq
        stop_sample = start_sample + duration * self.config.imu_freq
        t = np.arange(start_sample, stop_sample)[:359]
        static_period = raw_imu[t, :]
        static_period[:, [0, 2, 3, 5]] *= -1
        ext_acc = np.vstack([static_period[:, :3], accelerometer])
        ext_gyr = np.vstack([static_period[:, 3:], gyroscope])
        len_samples = ext_gyr.shape[0]

        fusion = imufusion.Ahrs()

        fusion_euler = np.empty((len_samples, 3))
        internal_states = np.empty((len_samples, 6))
        flags = np.empty((len_samples, 5))

        for index in range(len_samples):
            fusion.update_no_magnetometer(ext_gyr[index], ext_acc[index], 1/self.config.imu_freq)
            fusion_euler[index] = np.radians(fusion.quaternion.to_euler())

            ahrs_internal_states = fusion.internal_states
            internal_states[index] = np.array([ahrs_internal_states.acceleration_error,
                                               ahrs_internal_states.accelerometer_ignored,
                                               ahrs_internal_states.acceleration_rejection_timer,
                                               ahrs_internal_states.magnetic_error,
                                               ahrs_internal_states.magnetic_rejection_timer,
                                               ahrs_internal_states.magnetometer_ignored])

            ahrs_flags = fusion.flags
            flags[index] = np.array([ahrs_flags.initialising,
                                     ahrs_flags.acceleration_rejection_timeout,
                                     ahrs_flags.acceleration_rejection_warning,
                                     ahrs_flags.magnetic_rejection_timeout,
                                     ahrs_flags.magnetic_rejection_warning])

        fusion_euler[fusion_euler > np.pi] = fusion_euler[fusion_euler > np.pi] - 2*np.pi
        last_idx_init = np.argwhere(flags[:, 0] == 1)[-1].item()
        fusion_euler = fusion_euler[last_idx_init+1:, :]

        fusion_euler = fusion_euler + init_yaw % (2 * np.pi)
        fusion_euler[fusion_euler > np.pi] = fusion_euler[fusion_euler > np.pi] - 2 * np.pi

        if samples is not None:
            avg_fusion = np.array([np.mean(fusion_euler[samples[k]:samples[k+1], 2]) for k in range(len(samples)-1)])

        else:
            avg_fusion = None

        return avg_fusion

    @staticmethod
    def convert_large_angles(ang):
        if ang > np.pi:
            new_ang = ang % -np.pi
        elif ang < -np.pi:
            new_ang = ang % np.pi
        else:
            return ang
        return new_ang

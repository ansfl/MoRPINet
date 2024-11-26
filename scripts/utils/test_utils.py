import numpy as np
import scipy as sp


def dead_reckoning(pos_prev, step, psi=0, deg2rad=False):
    """
    0-axis is North and 1-axis is East
    :param pos_prev: the previous position
    :param step: the distance of progress
    :param psi: the direction of progress
    :param deg2rad: if psi is in degrees convert it to radians
    :return: current position
    """
    if deg2rad:
        psi = np.radians(psi)
    east_next = pos_prev[1] + step * np.cos(psi)
    north_next = pos_prev[0] + step * np.sin(psi)
    return np.array((north_next, east_next)).reshape(-1)


def downsampling_pos(pos, jump):
    downsampled_pos = pos[0::jump, :].copy()
    downsampled_pos = downsampled_pos[:-1, :]
    return downsampled_pos


def interpolate_gt(rtk, rtk_freq, new_freq):
    samples = int(rtk.shape[0] * new_freq / rtk_freq)
    t = np.arange(0, rtk.shape[0] / rtk_freq, 1 / rtk_freq)
    new_x = sp.interpolate.interp1d(t, rtk[:, 0], kind='cubic', fill_value="extrapolate")
    new_y = sp.interpolate.interp1d(t, rtk[:, 1], kind='cubic', fill_value="extrapolate")
    t_new = np.arange(0, samples / new_freq, 1 / new_freq)
    new_rtk = np.array([new_x(t_new), new_y(t_new)])
    return new_rtk.T

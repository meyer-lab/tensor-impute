from impute_helper import *
from imp_plots import generateTensor
from initialization import initialize_fac
from method_DO import perform_DO
from method_ALS import perform_ALS
from method_CLS import perform_CLS
from tracker import tracker

from figures import q2x_plot, rgbs
from common import *

from time import process_time
import numpy as np
import pickle
import os


def figure_2_data(impute_type = 'entry', reps = 50, track_rrs = [6,1,6], drop_perc = 0.1, testReg = False, save = False):
    assert impute_type == 'entry' or impute_type == 'chord'

    if save:
        dirname = "./figures/zohar-nonDecomp/" + impute_type
        if os.path.isdir(dirname) == False: os.makedirs(dirname)

    tensor = generateTensor()
    maxRank = 6
    imp = dict()
    fit = dict()
    tot = dict()

    methods = [perform_CLS, perform_ALS, perform_DO]
    # for aID, a in enumerate(alphas):
    for mID, m in enumerate(methods):
        savename = m.__name__

        start = process_time()
        track = tracker(tensor, track_runtime=True)
        for rep in range(reps):
            missingCube = np.copy(tensor)
            if impute_type == 'chord':
                mask = chord_drop(missingCube, int(drop_perc*tensor.size/tensor.shape[0]))
            if impute_type == 'entry':
                mask = entry_drop(missingCube, int(drop_perc*np.sum(np.isfinite(tensor))), dropany=True)

            # track masks
            track.set_mask(mask)
            imputed_vals = np.ones_like(missingCube) - mask # (1) represents artifically dropped values
            fitted_vals = np.isfinite(tensor) - imputed_vals # (1) represents untouched values that were not originally missing

            imp_err_temp = np.ndarray(0)
            fit_err_temp = np.ndarray(0)
            tot_err_temp = np.ndarray(0)

            for rr in range(1,maxRank+1):
                np.random.seed(rep)
                CPinit = initialize_fac(tensor, rank=rr, method='random')
                if rr == track_rrs[mID]:
                    track.begin()
                    tfac = m(missingCube, rank=rr, init=CPinit, n_iter_max=50, callback=track)
                else:
                    tfac = m(missingCube, rank=rr, init=CPinit, n_iter_max=50)

                imp_err_temp = np.hstack((imp_err_temp, calcR2X(tfac, tensor, calcError=True, mask=imputed_vals)))
                fit_err_temp = np.hstack((fit_err_temp, calcR2X(tfac, tensor, calcError=True, mask=fitted_vals)))
                tot_err_temp = np.hstack((tot_err_temp, calcR2X(tfac, tensor, calcError=True)))
                # [1, ..., 10]

            if rep == 0:
                imp_err = imp_err_temp
                fit_err = fit_err_temp
                tot_err = tot_err_temp
            else:
                imp_err = np.vstack((imp_err, imp_err_temp))
                fit_err = np.vstack((fit_err, fit_err_temp))
                tot_err = np.vstack((tot_err, tot_err_temp))

            if rep+1 < reps: track.new()
        
        imp[savename] = imp_err
        fit[savename] = fit_err
        tot[savename] = tot_err

        track.combine()
        if save: track.save(f"{dirname}/{savename}-track")
        print(f"{savename}: {process_time() - start} seconds total\n")

    if testReg:
        methods_ridge = [perform_CLS]
        alpha = 0.1
        # for aID, a in enumerate(alphas):
        for mID, m in enumerate(methods_ridge):
            savename = m.__name__ + "-Ridge"

            start = process_time()
            track = tracker(tensor, track_runtime=True)
            for rep in range(reps):
                missingCube = np.copy(tensor)
                mask = chord_drop(missingCube, int(drop_perc*tensor.size/tensor.shape[0]))

                # track masks
                track.set_mask(mask)
                imputed_vals = np.ones_like(missingCube) - mask # (1) represents artifically dropped values
                fitted_vals = np.isfinite(tensor) - imputed_vals # (1) represents untouched values that were not originally missing

                imp_err_temp = np.ndarray(0)
                fit_err_temp = np.ndarray(0)
                tot_err_temp = np.ndarray(0)

                for rr in range(1,maxRank+1):
                    np.random.seed(rep)
                    CPinit = initialize_fac(tensor, rank=rr, method='random')
                    if rr == track_rrs[mID]:
                        track.begin()
                        tfac = m(missingCube, rank=rr, init=CPinit, alpha=alpha, n_iter_max=50, callback=track)
                    else:
                        tfac = m(missingCube, rank=rr, init=CPinit, alpha=alpha, n_iter_max=50)
                    
                    imp_err_temp = np.hstack((imp_err_temp, calcR2X(tfac, tensor, calcError=True, mask=imputed_vals)))
                    fit_err_temp = np.hstack((fit_err_temp, calcR2X(tfac, tensor, calcError=True, mask=fitted_vals)))
                    tot_err_temp = np.hstack((tot_err_temp, calcR2X(tfac, tensor, calcError=True)))
                    # [1, ..., 10]

                if rep == 0:
                    imp_err = imp_err_temp
                    fit_err = fit_err_temp
                    tot_err = tot_err_temp
                else:
                    imp_err = np.vstack((imp_err, imp_err_temp))
                    fit_err = np.vstack((fit_err, fit_err_temp))
                    tot_err = np.vstack((tot_err, tot_err_temp))

                if rep+1 < reps: track.new()
            
            imp[savename] = imp_err
            fit[savename] = fit_err
            tot[savename] = tot_err

            track.combine()
            if save: track.save(f"{dirname}/{savename}-track")

    if save:
        with open(f"{dirname}/imp-array", "wb") as output_file: pickle.dump(imp, output_file)
        with open(f"{dirname}/fit-array", "wb") as output_file: pickle.dump(fit, output_file)
        with open(f"{dirname}/tot-array", "wb") as output_file: pickle.dump(tot, output_file)

def figure_2(imp_type = 'chord', names = ["perform_CLS", "perform_ALS", "perform_DO"], save=False):

    dirname = f"./figures/zohar-nonDecomp/{imp_type}"
    
    with open(f"{dirname}/imp-array", "rb") as input_file: imp = pickle.load(input_file)
    with open(f"{dirname}/fit-array", "rb") as input_file: fit = pickle.load(input_file)
    with open(f"{dirname}/tot-array", "rb") as input_file: tot = pickle.load(input_file)

    ax,f = getSetup((15,15), (2,2))
    m_track = tracker()

    threshold = 0.1
    unmet = list()
    for nID, n in enumerate(names):
        m_track.load(f"{dirname}/{n}-track")
        q2x_plot(ax[0], n, imp[n], fit[n], tot[n], color=rgbs(nID), offset=nID, logbound=-2)
        m_track.plot_iteration(ax[1], color=rgbs(nID, transparency=0.8), offset=nID, log=True, logbound=-2)
        
        thresholds = m_track.time_thresholds(threshold)
        ax[2].hist(thresholds, label=f"{n} ({len(thresholds)})", fc=rgbs(nID, transparency=0.25), edgecolor=rgbs(nID), bins=50, range=(0,0.2))
        print(np.mean(thresholds))
        ax[2].axvline(np.mean(thresholds), color=rgbs(nID), linestyle='dashed', linewidth=1)
        unmet.append(m_track.unmet_thresholds(threshold))

    ax[2].legend(loc='upper right')
    ax[2].set_xlabel('Runtime')
    ax[2].set_ylabel('Count')
    unmet[:] = [x / m_track.imputed_array.shape[0] * 100 for x in unmet]
    ax[3].bar(names, unmet)
    ax[3].set_xlabel('Method')
    ax[3].set_ylabel('Percent Unmet')
    ax[3].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    if save: f.savefig(f"{dirname}.png", bbox_inches="tight", format="png")


# figure_2_data_entry()
# figure_2_data_chord()
# figure_2('entry')
# figure_2('chord')
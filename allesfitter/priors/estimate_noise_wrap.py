import csv
import os

import numpy as np

from allesfitter.utils.latex_printer import round_txt_separately

from .. import config
from . import estimate_noise


def estimate_noise_wrap(pathdata):

    print(
        "\nEstimating errors and baselines... this will take a couple of minutes. Please be patient, you will get notified once everything is completed.\n"
    )

    #::: run MCMC fit to estimate errors and baselines
    estimate_noise.estimate_noise(pathdata)

    def fwrite(text):
        fname_params = os.path.join(pathdata, "params.csv")
        with open(fname_params, "a") as f:
            f.write(text + "\\n")

    def clean_up_csv(fname, N_last_rows=0):

        with open(fname) as f:
            params_csv = list(csv.reader(f))

        with open(fname, "w") as f:
            writer = csv.writer(f)
            for i in range(len(params_csv) - N_last_rows):
                row = params_csv[i]
                writer.writerow(row)

    #::: delete the rows containing the default (zero) errors and baselines from the params.csv file
    N_default_rows = 0
    clean_up_csv(os.path.join(pathdata, "params.csv"), N_last_rows=N_default_rows)

    config.init(pathdata)

    #::: write new rows into params.csv
    #::: errors
    fwrite("#errors per instrument,")

    for i, inst in enumerate(config.BASEMENT.settings["inst_phot"]):
        #::: read in the summary file
        summaryfile = os.path.join(pathdata, "priors", "summary_phot.csv")
        priors2 = np.genfromtxt(summaryfile, names=True, delimiter=",", dtype=None)
        priors = {}
        for key in priors2.dtype.names:
            priors[key] = np.atleast_1d(priors2[key])

        median = priors["ln_yerr_median"][i]
        err = 5.0 * np.max([float(priors["ln_yerr_ll"][i]), float(priors["ln_yerr_ul"][i])])
        median, err, _ = round_txt_separately(median, err, err)
        fwrite(
            "ln_err_flux_"
            + inst
            + ","
            + median
            + ",1,trunc_normal -23 0 "
            + median
            + " "
            + err
            + ",$\log{\sigma_\mathrm{"
            + inst
            + "}}$,"
        )

    for i, inst in enumerate(config.BASEMENT.settings["inst_rv"]):
        #::: read in the summary file
        summaryfile = os.path.join(pathdata, "priors", "summary_rv.csv")
        priors2 = np.genfromtxt(summaryfile, names=True, delimiter=",", dtype=None)
        priors = {}
        for key in priors2.dtype.names:
            priors[key] = np.atleast_1d(priors2[key])

        median = priors["ln_yerr_median"][i]
        err = 5.0 * np.max([float(priors["ln_yerr_ll"][i]), float(priors["ln_yerr_ul"][i])])
        median, err, _ = round_txt_separately(median, err, err)
        fwrite(
            "ln_jitter_rv_"
            + inst
            + ","
            + median
            + ",1,trunc_normal -23 0 "
            + median
            + " "
            + err
            + ",$\log{\sigma_\mathrm{jitter; "
            + inst
            + "}}$,"
        )

    #::: write new rows into params.csv
    #::: baselines
    fwrite("#baseline per instrument,")

    for i, inst in enumerate(config.BASEMENT.settings["inst_phot"]):
        #::: read in the summary file
        summaryfile = os.path.join(pathdata, "priors", "summary_phot.csv")
        priors2 = np.genfromtxt(summaryfile, names=True, delimiter=",", dtype=None)
        priors = {}
        for key in priors2.dtype.names:
            priors[key] = np.atleast_1d(priors2[key])

        median = priors["gp_ln_sigma_median"][i]
        err = 5.0 * np.max([float(priors["gp_ln_sigma_ll"][i]), float(priors["gp_ln_sigma_ul"][i])])
        median, err, _ = round_txt_separately(median, err, err)
        fwrite(
            "baseline_gp1_flux_"
            + inst
            + ","
            + median
            + ",1,trunc_normal -23 23 "
            + median
            + " "
            + err
            + ",$\mathrm{gp: \log{\sigma} ("
            + inst
            + ")}$,"
        )

        median = priors["gp_ln_rho_median"][i]
        err = 5.0 * np.max([float(priors["gp_ln_rho_ll"][i]), float(priors["gp_ln_rho_ul"][i])])
        median, err, _ = round_txt_separately(median, err, err)
        fwrite(
            "baseline_gp2_flux_"
            + inst
            + ","
            + median
            + ",1,trunc_normal -23 23 "
            + median
            + " "
            + err
            + ",$\mathrm{gp: \log{\\rho} ("
            + inst
            + ")}$,"
        )

import argparse
import yaml

from analysis import data_bootstrap
from analysis import data_binnedbootstrap
from analysis import sweep_subtraction
from analysis import sweep_BStrapModalDecomposition
from analysis import sweep_BStrapSmearedSPF
from analysis import sweep_BStrapHFilteringCt
from analysis import sweep_BStrapEnergyMoment

from analysis import gen_mock_Ct

from config import Config


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("config")
    parser.add_argument("--task", required=True)

    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    if args.task == "data":
        data_binnedbootstrap.main(cfg)
        sweep_subtraction.main(cfg)
        sweep_BStrapModalDecomposition.main(cfg)

    if args.task == "SmearedSPF":
        sweep_BStrapSmearedSPF.main(cfg)

    if args.task == "HPassFilter":
        sweep_BStrapHFilteringCt.main(cfg)

    if args.task == "EnergyMoment":
        sweep_BStrapEnergyMoment.main(cfg)

    if args.task == "mock":
        gen_mock_Ct.main(cfg)
        data_binnedbootstrap.main(cfg)
        sweep_subtraction.main(cfg)
        sweep_BStrapModalDecomposition.main(cfg)


if __name__ == "__main__":
    main()

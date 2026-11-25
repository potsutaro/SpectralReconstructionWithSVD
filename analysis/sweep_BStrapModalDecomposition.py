import sys
import numpy as np
import matplotlib.pyplot as plt

from . import DataNormalization
from . import SmearedSPF

from config import Config


#=============================================================#
# call as function

def main(cfg: Config):

    tolerance = 1e-14
    regulator = 0.0
    
    wmin = 0.01
    wmax = 3.01
    nw = 1000
    
    lower = cfg.analysis.init 
    upper = cfg.analysis.size
    dof = cfg.analysis.dof
    size = cfg.analysis.dof
    flag = cfg.statistics.stati
    directory = cfg.paths.corr

    #=============================================================#
    #=============================================================#
    # Reconstruction parameter
    
    wstep = (wmax - wmin) / nw
    w1 = []
    for i in range(nw):
        w = wmin + wstep*i
        w1.append(w)
    print(wmin, wmax, nw, wstep)
    
    #=============================================================#
    #=============================================================#
    #set the fitting range and target files
    
    fitrange = [lower, upper]
    
    jkfile = './' + directory + '/ndata_jk'
    avfile = './' + directory + '/ndata'
    
    f = open(jkfile,'rt');
    f2 = open(avfile, 'rt')
    
    exx_list = []
    exy_list = []
    
    inx_list = []
    iny_list = []
    
    lin = 0
    for string in f:
        if fitrange[0] <= (lin % size) <= fitrange[1]:
            data = string[:-1].split(' ')
            time = int(data[0])
            value = float(data[1]) 
            inx_list.append(time)
            iny_list.append(value)
            if (lin % size) == fitrange[1]:
                exx_list.append(inx_list)
                exy_list.append(iny_list)
                inx_list = []
                iny_list = []
        lin += 1
    
    lin = 0
    
    sig_list = []
    
    t1 = []
    for string in f2:
        if fitrange[0] <= (lin % size) <= fitrange[1]:
            data = string[:-1].split(' ')
            t1.append(float(data[0]))
            sig_list.append(float(data[2]))
        lin += 1
    
    tmin = int(t1[0])
    tmax = int(t1[-1])
    nt = len(t1)
    tstep = 1 
    print(exx_list[0])
    print(tmin, tmax, nt, tstep)
    
    array_s = np.array(sig_list)
    
    conf = len(exy_list)
    
    #=============================================================#
    #=============================================================#
    # Prepare the data
    # cov_S = Summation of all cov for each Jackknife sample
    
    cov_S = DataNormalization.CovarianceMatrix(exx_list, exy_list, conf, dof, flag)
    
    #=============================================================#
    #=============================================================#
    # Basis functions 
    
    S_diag, UFunc, VFunc, VFuncA = SmearedSPF.GetBasisFunc(nt,tmin,tstep,nw,wmin,wstep,t0=0)
    fullrank = len(S_diag)
    print(S_diag)
    Nmax = SmearedSPF.GetMaximalN(S_diag, tolerance)
    Nl = len(VFuncA)
    srun = SmearedSPF.FigSingularValues(S_diag, nw)
    
    #=============================================================#
    #=============================================================#
    # l-th strength from correlator
    
    rho_c_jk = []
    c_l_jk = []
    for i in range(conf):
        via_rho = []
        via_c = []
        for l in range(fullrank):
            val = 0
            for t in range(len(t1)):
                val += UFunc[l][t] * exy_list[i][t] * np.sqrt(tstep)
            via_rho.append(val)
            via_c.append(val / S_diag[l])
        rho_c_jk.append(via_rho)
        c_l_jk.append(via_c)
    
    rho_c = []
    c_l = []
    rho_c_transpose = np.array(rho_c_jk).T
    c_l_transpose = np.array(c_l_jk).T
    for l in range(len(rho_c_transpose)):
        ave0, err0 = DataNormalization.StatisticalAnal(rho_c_transpose[l], conf, 1)
        ave1, err1 = DataNormalization.StatisticalAnal(c_l_transpose[l], conf, 1)
        rho_c.append([ave0, err0])
        c_l.append([ave1, err1])
    
    
    #=============================================================#
    #=============================================================#
    
    jkoutput = SmearedSPF.PrintJKFile(c_l_jk, 'mdata', 'modaldecomposition_data')
    avoutput = SmearedSPF.PrintAVFile(c_l_jk, 'mdata', 'modaldecomposition_data', flag)
    
    jkoutput = SmearedSPF.PrintJKFile(rho_c_jk, 'rdata', 'modaldecomposition_data')
    avoutput = SmearedSPF.PrintAVFile(rho_c_jk, 'rdata', 'modaldecomposition_data', flag)
    
    
    #=============================================================#
    #=============================================================#
    
    rrun = SmearedSPF.FigRhoAbs(c_l, Nmax)
    
    #=============================================================#
    #=============================================================#
    
    colors = plt.get_cmap("tab20c")(np.linspace(0, 1, 3))
    
    svdrank = []
    Cl_mean = []
    Cl_sdev = []
    
    
    for l in range(fullrank):
        svdrank.append(l)
        Cl_mean.append(c_l[l][0])
        Cl_sdev.append(c_l[l][1])
    
    plt.errorbar(svdrank, Cl_mean, yerr = Cl_sdev, capsize=5, fmt='o', markersize=5, ecolor=colors[0], markeredgecolor=colors[0], color=colors[0])
    plt.xlabel('$l$', fontsize=15)
    plt.ylabel('$\\rho_{l}$', fontsize=15)
    plt.ylim(-1.5,1.5)
    # plt.legend()
    plt.savefig('./results/SpecFunc/ReconstRho.pdf')
    plt.show()


#=============================================================#
# from command line

if __name__ == "__main__":

    import sys

    main(
        int(sys.argv[1]),
        int(sys.argv[2]),
        int(sys.argv[3]),
        int(sys.argv[4]),
        int(sys.argv[5]),
        sys.argv[6],
    )

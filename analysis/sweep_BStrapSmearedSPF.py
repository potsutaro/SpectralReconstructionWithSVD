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
    
    wmin = cfg.analysis.wmin
    wmax = cfg.analysis.wmax
    nw = cfg.analysis.nw

    size = cfg.analysis.dof
    delta = cfg.analysis.delta
    Nin = cfg.analysis.nin 
    Ntr = cfg.analysis.ntr
    Neb = cfg.analysis.neb
    
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
    
    CL = 1
    
    mjkfile = './modaldecomposition_data/mdata_jk'
    mavfile = './modaldecomposition_data/mdata'
    
    mjk = open(mjkfile,'rt');
    mav = open(mavfile, 'rt')
    
    c_l_jk = []
    
    iny_list = []
    
    lin = 0
    for string in mjk:
        if 0 <= int(lin % size) <= size:
            data = string[:-1].split(' ')
            iny_list.append(float(data[1]))
            if (lin % size) == (size-1):
                c_l_jk.append(iny_list)
                iny_list = []
        lin += 1
    
    c_l = []
    t1 = []
    
    lin = 0
    for string in mav:
        if 0 <= (lin % size) <= size:
            data = string[:-1].split(' ')
            t1.append(float(data[0]))
            c_l.append([float(data[1]),float(data[2])])
        lin += 1
    print(c_l)
    
    tmin = int(t1[0])
    tmax = int(t1[-1])
    nt = len(t1)
    tstep = 1 
    print(tmin, tmax, nt, tstep)
    
    conf = len(c_l_jk)
    
    
    rjkfile = './modaldecomposition_data/rdata_jk'
    ravfile = './modaldecomposition_data/rdata'
    
    rjk = open(rjkfile,'rt');
    rav = open(ravfile, 'rt')
    
    
    rho_c_jk = []
    
    iny_list = []
    
    lin = 0
    for string in rjk:
        if 0 <= int(lin % size) <= size:
            data = string[:-1].split(' ')
            iny_list.append(float(data[1]))
            if (lin % size) == (size-1):
                rho_c_jk.append(iny_list)
                iny_list = []
        lin += 1
    
    rho_c = []
    
    lin = 0
    for string in rav:
        if 0 <= (lin % size) <= size:
            data = string[:-1].split(' ')
            rho_c.append([float(data[1]),float(data[2])])
        lin += 1
    
    #=============================================================#
    #=============================================================#
    # Reconstruction parameter
    
    tolerance = 1e-14
    regulator = 0.0
    
    
    wmin = 0.01
    wmax = 3.01
    nw = 1000
    wstep = (wmax - wmin) / nw
    w1 = []
    for i in range(nw):
        w = wmin + wstep*i
        w1.append(w)
    print(wmin, wmax, nw, wstep)
    
    #=============================================================#
    #=============================================================#
    # reconstruct Spectral function
    
    skip = 10
    rw = int(nw / skip) 
    rstep = (wmax - wmin) / rw
    
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
    # reconstruct Spectral function
    
    lmax_list = [Ntr]
    lmaxSpecF_jk_list = []
    for lmax in lmax_list:
        lmaxSpecF_jk = []
        for i in range(conf):
            lmaxSpecF = []
            for w in range(rw): # nw
                val = SmearedSPF.regularizedSPF(w, wstep, VFuncA, S_diag, rho_c_jk[i], lmax, regulator)
                lmaxSpecF.append(val)
            lmaxSpecF_jk.append(lmaxSpecF)
        lmaxSpecF_jk_list.append(lmaxSpecF_jk)
    
    lmaxSpecF_list = []
    for lmax in range(len(lmax_list)):
        lmaxSpecF_stat = []
        lmaxSpecF_transpose = np.array(lmaxSpecF_jk_list[lmax]).T
        for w in range(rw): # nw
            lave, lerr = DataNormalization.StatisticalAnal(lmaxSpecF_transpose[w], conf, 1)
            lmaxSpecF_stat.append([lave, lerr])
        lmaxSpecF_list.append(lmaxSpecF_stat)
    
    wa_list = []
    lmaxSpecF_mean = []
    lmaxSpecF_sdev = []
    for i in range(rw): # nw
        w = wmin + rstep*i #nw
        wa_list.append(w)
        lmaxSpecF_mean.append(lmaxSpecF_list[0][i][0])
        lmaxSpecF_sdev.append(lmaxSpecF_list[0][i][1])
    
    #=============================================================#
    #=============================================================#
    
    colors = plt.get_cmap("tab20c")(np.linspace(0, 1, 3))
    
    strtitle = 'Up to $l=$' + str(lmax_list[0]) 
    plt.title(strtitle)
    plt.errorbar(wa_list, lmaxSpecF_mean, yerr=lmaxSpecF_sdev, capsize=5, fmt='o', markersize=5, color=colors[1], label="Reconst.")
    plt.xlabel('$\\omega a$', fontsize=15)
    plt.ylabel('$\\rho(\\omega)$', fontsize=15)
    plt.ylim(0,1)
    plt.legend()
    plt.show()
    
    #=============================================================#
    #=============================================================#
    # truncation error
    
    # A =  c_l[0][0]
    # B = 0
    # print("fit",A,B)
    
    A, B = SmearedSPF.LinearUpperbounds(c_l, Nin, Neb, CL)
    print("fit",A,B)
    
    
    rrun = SmearedSPF.FigRhoAbsWithUpperBound(c_l, nt, A, B)
    
    truncation_org = SmearedSPF.AbsTruncationError(wa_list, VFuncA, wstep, Ntr, A, B)
    
    colors = plt.get_cmap("tab20c")(np.linspace(0, 1, 3))
    
    strtitle = 'Up to $l=$' + str(lmax_list[0]) 
    plt.title(strtitle)
    plt.errorbar(wa_list, lmaxSpecF_mean, yerr=truncation_org, capsize=0, fmt='o', markersize=5, color=colors[1], label="Total")
    plt.errorbar(wa_list, lmaxSpecF_mean, yerr=lmaxSpecF_sdev, capsize=0, fmt='o', markersize=5, color=colors[0], label="Statistical")
    plt.xlabel('$\\omega a$', fontsize=15)
    plt.ylabel('$\\rho(\\omega)$', fontsize=15)
    plt.ylim(0,1)
    plt.legend()
    plt.savefig('./results/SpecFunc/ReconstSpecFNtr='+str(Ntr)+'.pdf')
    plt.show()
    #=============================================================#
    #=============================================================#
    # Smeared spectral function - Recon
    
    KFunc_trans = []
    for i in range(rw): # nw
        w = wmin + i*rstep # wstep
        KFunc_w = SmearedSPF.SpfCoeff(w, Nl, wmin, wmax, nw, delta, VFuncA)
        # KFunc_w = SmearedSPF.SpfCoeff(w, Ne, wmin, wmax, nw, delta, VFuncA)
        KFunc_trans.append(KFunc_w)
    KFuncA = list(np.array(KFunc_trans).T)
    
    lmax_list = [Ntr]
    lmaxSSpecF_jk_list = []
    for lmax in lmax_list:
        lmaxSSpecF_jk = []
        for i in range(conf):
            lmaxSSpecF = []
            for w in range(rw): # nw
                val = SmearedSPF.regularizedSPF(w, wstep, KFuncA, S_diag, rho_c_jk[i], lmax, regulator)
                lmaxSSpecF.append(val)
            lmaxSSpecF_jk.append(lmaxSSpecF)
        lmaxSSpecF_jk_list.append(lmaxSSpecF_jk)
    
    lmaxSSpecF_list = []
    for lmax in range(len(lmax_list)):
        lmaxSSpecF_stat = []
        lmaxSSpecF_transpose = np.array(lmaxSSpecF_jk_list[lmax]).T
        for w in range(rw): # nw
            lave, lerr = DataNormalization.StatisticalAnal(lmaxSSpecF_transpose[w], conf, 1)
            lmaxSSpecF_stat.append([lave, lerr])
        lmaxSSpecF_list.append(lmaxSSpecF_stat)
    
    wa_list = []
    lmaxSSpecF_mean = []
    lmaxSSpecF_sdev = []
    for i in range(rw): # nw
        w = wmin + rstep*i #nw
        wa_list.append(w)
        lmaxSSpecF_mean.append(lmaxSSpecF_list[0][i][0])
        lmaxSSpecF_sdev.append(lmaxSSpecF_list[0][i][1])
    
    #=============================================================#
    #=============================================================#
    
    colors = plt.get_cmap("tab20c")(np.linspace(0, 1, 3))
    
    strtitle = 'Up to $l=$' + str(lmax_list[0]) 
    plt.title(strtitle)
    plt.errorbar(wa_list, lmaxSSpecF_mean, yerr=lmaxSSpecF_sdev, capsize=5, fmt='o', markersize=5, color=colors[1], label="Reconst.")
    plt.xlabel('$\\omega a$', fontsize=15)
    plt.ylabel('$\\rho_{\\sigma_{\\rm{smr}}}(\\omega)$', fontsize=15)
    plt.ylim(0,1)
    plt.legend()
    plt.show()
    
    #=============================================================#
    #=============================================================#
    # truncation error
    
    # A = c_l[0][0]
    # B = 0
    # print("fit",A,B)
    
    A, B = SmearedSPF.LinearUpperbounds(c_l, Nin, Neb, CL)
    
    rrun = SmearedSPF.FigRhoAbsWithUpperBound(c_l, nt, A, B)
    
    truncation_smear = SmearedSPF.AbsTruncationError(wa_list, KFuncA, wstep, Ntr, A, B)
    
    colors = plt.get_cmap("tab20c")(np.linspace(0, 1, 3))
    
    strtitle = 'Up to $l=$' + str(lmax_list[0]) 
    plt.title(strtitle)
    plt.errorbar(wa_list, lmaxSSpecF_mean, yerr=truncation_smear, capsize=0, fmt='o', markersize=5, color=colors[1], label="Total")
    plt.errorbar(wa_list, lmaxSSpecF_mean, yerr=lmaxSSpecF_sdev, capsize=0, fmt='o', markersize=5, color=colors[0], label="Statistical")
    plt.xlabel('$\\omega a$', fontsize=15)
    plt.ylabel('$\\rho_{\\sigma_{\\rm{smr}}}(\\omega)$', fontsize=15)
    plt.ylim(0,1)
    plt.legend()
    plt.savefig('./results/SpecFunc/ReconstSSpecFDelta='+str(delta)+'Ntr='+str(Ntr)+'.pdf')
    plt.show()

#=============================================================#
# from command line

if __name__ == "__main__":

    import sys

    main(
        int(sys.argv[1]),
        float(sys.argv[2]),
        int(sys.argv[3]),
        int(sys.argv[4]),
        int(sys.argv[5]),
    )

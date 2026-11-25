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
    
    delta = float(args[2])
    size = int(args[1])
    Nin = int(args[3])
    Ntr = int(args[4])
    Neb = int(args[5])
    
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
    
    
    cjkfile = './correlator_data/ndata_jk'
    cavfile = './correlator_data/ndata'
    
    cjk = open(cjkfile, 'rt')
    cav = open(cavfile, 'rt')
    
    iny_list = []
    
    Ct_jk = []
    lin = 0
    for string in cjk:
        if 0 <= int(lin % size) <= size:
            data = string[:-1].split(' ')
            iny_list.append(float(data[1]))
            if (lin % size) == (size-1):
                Ct_jk.append(iny_list)
                iny_list = []
        lin += 1
    
    avCt = []
    erCt = []
    lin = 0
    for string in cav:
        if 0 <= int(lin % size) <= size:
            data = string[:-1].split(' ')
            avCt.append(float(data[1]))
            erCt.append(float(data[2]))
        lin += 1
    
    
    ejkfile = './correlator_data/nass_jk'
    eavfile = './correlator_data/nass'
    
    ejk = open(ejkfile, 'rt')
    eav = open(eavfile, 'rt')
    
    En_jk = []
    lin = 0
    for string in ejk:
        if 0 <= int(lin % size) <= size:
            data = string[:-1].split(' ')
            iny_list.append(float(data[1]))
            if (lin % size) == (size-1):
                En_jk.append(iny_list)
                iny_list = []
        lin += 1
    
    avEn = []
    erEn = []
    lin = 0
    for string in eav:
        if 0 <= int(lin % size) <= size:
            data = string[:-1].split(' ')
            avEn.append(float(data[1]))
            erEn.append(float(data[2]))
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
    # Energy moment -0th
    
    KFunc_trans0th = []
    for t in range(nt): 
        KFunc_t0th = SmearedSPF.EMomCoeff(t, Nl, wmin, wmax, nw, 0, avCt, VFuncA)
        KFunc_trans0th.append(KFunc_t0th)
    KFuncA0th = list(np.array(KFunc_trans0th).T)
    
    lmax_list = [Ntr]
    lmaxEMom0th_jk_list = []
    for lmax in lmax_list:
        lmaxEMom0th_jk = []
        for i in range(conf):
            lmaxEMom0th = []
            for t in range(nt): 
                val = SmearedSPF.regularizedSPF(t, wstep, KFuncA0th, S_diag, rho_c_jk[i], lmax, regulator)
                lmaxEMom0th.append(val)
            lmaxEMom0th_jk.append(lmaxEMom0th)
        lmaxEMom0th_jk_list.append(lmaxEMom0th_jk)
    
    lmaxEMom0th_list = []
    for lmax in range(len(lmax_list)):
        lmaxEMom0th_stat = []
        lmaxEMom0th_transpose = np.array(lmaxEMom0th_jk_list[lmax]).T
        for t in range(nt):
            lave, lerr = DataNormalization.StatisticalAnal(lmaxEMom0th_transpose[t], conf, 1)
            lmaxEMom0th_stat.append([lave, lerr])
        lmaxEMom0th_list.append(lmaxEMom0th_stat)
    
    ta_list = []
    lmaxEMom0th_mean = []
    lmaxEMom0th_sdev = []
    for t in range(nt):
        ta_list.append(t)
        lmaxEMom0th_mean.append(lmaxEMom0th_list[0][t][0])
        lmaxEMom0th_sdev.append(lmaxEMom0th_list[0][t][1])
    
    #=============================================================#
    #=============================================================#
    
    colors = plt.get_cmap("tab20c")(np.linspace(0, 1, 3))
    
    strtitle = 'Up to $l=$' + str(lmax_list[0]) 
    plt.title(strtitle)
    plt.errorbar(ta_list, lmaxEMom0th_mean, yerr=lmaxEMom0th_sdev, capsize=5, fmt='o', markersize=5, color=colors[1], label="Reconst.")
    plt.xlabel('$t/a$', fontsize=15)
    plt.ylabel('$C(t)$', fontsize=15)
    plt.yscale('log')
    plt.legend()
    plt.show()
    
    #=============================================================#
    #=============================================================#
    # truncation error
    
    A, B = SmearedSPF.LinearUpperbounds(c_l, Nin, Neb, 1)
    
    rrun = SmearedSPF.FigRhoAbsWithUpperBound(c_l, nt, A, B)
    
    truncation_smear = SmearedSPF.AbsTruncationError(ta_list, KFuncA0th, wstep, Ntr, A, B)
    
    colors = plt.get_cmap("tab20c")(np.linspace(0, 1, 3))
    
    strtitle = 'Up to $l=$' + str(lmax_list[0]) 
    plt.title(strtitle)
    plt.errorbar(ta_list, lmaxEMom0th_mean, yerr=truncation_smear, capsize=0, fmt='o', markersize=5, color=colors[1], label="w/ error")
    plt.errorbar(ta_list, lmaxEMom0th_mean, yerr=lmaxEMom0th_sdev, capsize=0, fmt='o', markersize=5, color=colors[0], label="truncated")
    plt.errorbar(ta_list, avCt, yerr=erCt, capsize=0, fmt='o', markersize=5, color=colors[2], label="original")
    plt.xlabel('$t/a$', fontsize=15)
    plt.ylabel('$C(t)$', fontsize=15)
    plt.legend()
    plt.yscale('log')
    plt.savefig('./results/EnergyMoment/ReconstEMom0thNtr='+str(Ntr)+'.pdf')
    plt.show()
    
    jkrun = SmearedSPF.PrintJKFile(lmaxEMom0th_jk_list[0], 'emom0th', 'results/EnergyMoment')
    torun = SmearedSPF.PrintTOFile(lmaxEMom0th_jk_list[0], truncation_smear, 'emom0th', 'results/EnergyMoment', 1)
    
    
    #=============================================================#
    #=============================================================#
    # Energy moment -1st
    
    KFunc_trans1st = []
    for t in range(nt): 
        KFunc_t1st = SmearedSPF.EMomCoeff(t, Nl, wmin, wmax, nw, 1, avCt, VFuncA)
        KFunc_trans1st.append(KFunc_t1st)
    KFuncA1st = list(np.array(KFunc_trans1st).T)
    
    lmax_list = [Ntr]
    lmaxEMom1st_jk_list = []
    for lmax in lmax_list:
        lmaxEMom1st_jk = []
        for i in range(conf):
            lmaxEMom1st = []
            for t in range(nt): 
                val = SmearedSPF.regularizedSPF(t, wstep, KFuncA1st, S_diag, rho_c_jk[i], lmax, regulator)
                val = val / Ct_jk[i][t]
                lmaxEMom1st.append(val)
            lmaxEMom1st_jk.append(lmaxEMom1st)
        lmaxEMom1st_jk_list.append(lmaxEMom1st_jk)
    
    lmaxEMom1st_list = []
    for lmax in range(len(lmax_list)):
        lmaxEMom1st_stat = []
        lmaxEMom1st_transpose = np.array(lmaxEMom1st_jk_list[lmax]).T
        for t in range(nt):
            lave, lerr = DataNormalization.StatisticalAnal(lmaxEMom1st_transpose[t], conf, 1)
            lmaxEMom1st_stat.append([lave, lerr])
        lmaxEMom1st_list.append(lmaxEMom1st_stat)
    
    ta_list = []
    lmaxEMom1st_mean = []
    lmaxEMom1st_sdev = []
    for t in range(nt):
        ta_list.append(t)
        lmaxEMom1st_mean.append(lmaxEMom1st_list[0][t][0])
        lmaxEMom1st_sdev.append(lmaxEMom1st_list[0][t][1])
    
    #=============================================================#
    #=============================================================#
    # truncation error
    
    A, B = SmearedSPF.LinearUpperbounds(c_l, Nin, Neb, 1)
    
    rrun = SmearedSPF.FigRhoAbsWithUpperBound(c_l, nt, A, B)
    
    truncation_smear = SmearedSPF.AbsTruncationError(ta_list, KFuncA1st, wstep, Ntr, A, B)
    for t in range(nt):
        truncation_smear[t] = truncation_smear[t] / avCt[t]
    
    colors = plt.get_cmap("tab20c")(np.linspace(0, 1, 3))
    
    strtitle = 'Up to $l=$' + str(lmax_list[0]) 
    plt.title(strtitle)
    plt.errorbar(ta_list, lmaxEMom1st_mean, yerr=truncation_smear, capsize=0, fmt='o', markersize=5, color=colors[1], label="w/ error")
    plt.errorbar(ta_list, lmaxEMom1st_mean, yerr=lmaxEMom1st_sdev, capsize=0, fmt='o', markersize=5, color=colors[0], label="truncated")
    plt.errorbar(ta_list, avEn, yerr=erEn, capsize=0, fmt='o', markersize=5, color=colors[2], label="original")
    plt.xlabel('$t/a$', fontsize=15)
    plt.ylabel('$\\langle \\omega \\rangle(t)$', fontsize=15)
    plt.legend()
    plt.savefig('./results/EnergyMoment/ReconstEMom1stNtr='+str(Ntr)+'.pdf')
    plt.show()
    
    jkrun = SmearedSPF.PrintJKFile(lmaxEMom1st_jk_list[0], 'emom1st', 'results/EnergyMoment')
    torun = SmearedSPF.PrintTOFile(lmaxEMom1st_jk_list[0], truncation_smear, 'emom1st', 'results/EnergyMoment', 1)
    
    #=============================================================#
    #=============================================================#
    # Energy moment -2nd
    
    KFunc_trans2nd = []
    for t in range(nt): 
        KFunc_t2nd = SmearedSPF.EMomCoeff(t, Nl, wmin, wmax, nw, 2, avCt, VFuncA)
        KFunc_trans2nd.append(KFunc_t2nd)
    KFuncA2nd = list(np.array(KFunc_trans2nd).T)
    
    lmax_list = [Ntr]
    lmaxEMom2nd_jk_list = []
    for lmax in lmax_list:
        lmaxEMom2nd_jk = []
        for i in range(conf):
            lmaxEMom2nd = []
            for t in range(nt): 
                sav = SmearedSPF.regularizedSPF(t, wstep, KFuncA2nd, S_diag, rho_c_jk[i], lmax, regulator) 
                val = sav / Ct_jk[i][t] - (En_jk[i][t])**2
                lmaxEMom2nd.append(val)
            lmaxEMom2nd_jk.append(lmaxEMom2nd)
        lmaxEMom2nd_jk_list.append(lmaxEMom2nd_jk)
    
    lmaxEMom2nd_list = []
    for lmax in range(len(lmax_list)):
        lmaxEMom2nd_stat = []
        lmaxEMom2nd_transpose = np.array(lmaxEMom2nd_jk_list[lmax]).T
        for t in range(nt):
            lave, lerr = DataNormalization.StatisticalAnal(lmaxEMom2nd_transpose[t], conf, 1)
            lmaxEMom2nd_stat.append([lave, lerr])
        lmaxEMom2nd_list.append(lmaxEMom2nd_stat)
    
    ta_list = []
    lmaxEMom2nd_mean = []
    lmaxEMom2nd_sdev = []
    for t in range(nt):
        ta_list.append(t)
        lmaxEMom2nd_mean.append(lmaxEMom2nd_list[0][t][0])
        lmaxEMom2nd_sdev.append(lmaxEMom2nd_list[0][t][1])
    
    
    #=============================================================#
    #=============================================================#
    # truncation error
    
    A, B = SmearedSPF.LinearUpperbounds(c_l, Nin, Neb, 1)
    
    rrun = SmearedSPF.FigRhoAbsWithUpperBound(c_l, nt, A, B)
    
    truncation_smear = SmearedSPF.AbsTruncationError(ta_list, KFuncA2nd, wstep, Ntr, A, B)
    for t in range(nt):
        truncation_smear[t] = truncation_smear[t] / avCt[t]
    
    colors = plt.get_cmap("tab20c")(np.linspace(0, 1, 3))
    
    strtitle = 'Up to $l=$' + str(lmax_list[0]) 
    plt.title(strtitle)
    plt.errorbar(ta_list, lmaxEMom2nd_mean, yerr=truncation_smear, capsize=0, fmt='o', markersize=5, color=colors[1], label="w/ error")
    plt.errorbar(ta_list, lmaxEMom2nd_mean, yerr=lmaxEMom2nd_sdev, capsize=0, fmt='o', markersize=5, color=colors[0], label="truncated")
    plt.xlabel('$t/a$', fontsize=15)
    plt.ylabel('$C(t)$', fontsize=15)
    plt.ylabel('$\\langle (\\omega - \\langle \\omega \\rangle)^2 \\rangle(t)$', fontsize=15)
    plt.legend()
    plt.savefig('./results/EnergyMoment/ReconstEMom2ndNtr='+str(Ntr)+'.pdf')
    plt.show()
    
    jkrun = SmearedSPF.PrintJKFile(lmaxEMom2nd_jk_list[0], 'emom2nd', 'results/EnergyMoment')
    torun = SmearedSPF.PrintTOFile(lmaxEMom2nd_jk_list[0], truncation_smear, 'emom2nd', 'results/EnergyMoment', 1)

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

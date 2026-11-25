import sys
import numpy as np
import scipy as sp
import scipy.optimize
from scipy.optimize import minimize
import lsqfit
import gvar as gv
import matplotlib.pyplot as plt
import matplotlib.colors as cls

EPSILON = 0.000001

ref = 1.0

#=============================================================#
#=============================================================#
# Statistics 

def StatisticalAnal(result, n, flag):
    ave = np.sum(np.array(result)) / n
    var = 0
    for r in result:
        if (flag == 0):
            var += (1. - 1. / n) * ((ave - r)**2)
        else:
            var += 1. / n * ((ave - r)**2)
    err = np.sqrt(var)
    return ave, err


#=============================================================#
#=============================================================#
# Singular value decomposition

def SingularValueDecomposition(K):
    # SVD for T
    U, S_diag, V_t = sp.linalg.svd(K)
    size = len(S_diag)
    N_diag = S_diag / S_diag[0]
    S_list = list(N_diag)

    return U, S_diag, V_t, N_diag

def SingleUFunc(U, l):
    U_l = U[:,l]

    return U_l

def SingleVFunc(V_t, l):
    # V_l = V_t[l,:].T
    V_l = V_t.T[:,l]

    return V_l

def UVUnitary(U,V_t):
    # Udelta = (U.T).conjugate() @ U
    # urun = HeatMap(Udelta)

    Vdelta = V_t @ (V_t.T).conjugate()
    vrun = HeatMap(Vdelta.real)

    return 0

def SVDbasis(exy_list, U, t_list, V_t, w_list, S_diag, svdrank):
    # whole
    C = (U.T).conjugate() @ np.array(exy_list)
    S = []
    for l in range(len(S_diag)):
        value = C[l] / S_diag[l]
        S.append(value)

    # sliced
    tC = C[:svdrank]
    tS = S[:svdrank]

    return tC, tS

def GetBasisFunc(nt,tmin,tstep,nw,wmin,wstep,t0):
    # Kernel
    K = construct_K(nt,tmin,tstep,nw,wmin,wstep,t0)

    # Singular value decomposition of the Kernel
    U, S_diag, V_t, N_diag= SingularValueDecomposition(K)
    fullrank = len(S_diag)

    # Basis functions
    UFunc_org = []
    VFunc_org = []
    
    for l in range(fullrank):
        UFunc_l = SingleUFunc(U, l)
        VFunc_l = SingleVFunc(V_t, l)
        UFunc_org.append(UFunc_l)
        VFunc_org.append(VFunc_l)
    
    UFunc = []
    VFunc = []
    for l in range(fullrank):
        flag = VFunc_org[l][1]
        if flag < 0:
            UFunc.append((-1)*UFunc_org[l])
            VFunc.append((-1)*VFunc_org[l])
        else:
            UFunc.append(UFunc_org[l])
            VFunc.append(VFunc_org[l])

    # For all
    VFunc_org = []
    
    for l in range(len(V_t)):
        VFunc_l = SingleVFunc(V_t, l)
        VFunc_org.append(VFunc_l)
    
    VFuncA = []
    
    for l in range(len(V_t)):
        flag = VFunc_org[l][1]
        if flag < 0:
            VFuncA.append((-1)*VFunc_org[l])
        else:
            VFuncA.append(VFunc_org[l])

    return S_diag, UFunc, VFunc, VFuncA

def calcIntegralOmega(Func, step):
    area = 0
    N = len(Func)
    for j in range(N):
        area += Func[j] * step
    return area

def GetMaximalN(S_diag, tole):
    Nmax = 0
    for l in range(len(S_diag)):
        if S_diag[l] > tole:
            Nmax = Nmax + 1
        else:
            break

    return Nmax

def GetPeakIndex(y):
    index = y.index(max(y))

    return index

#=============================================================#
#=============================================================#
# smeared Spectral function

def construct_K(nt, tmin, tstep, nw, wmin, wstep, t0):
    K = np.zeros((nt,nw))
    
    # matrix K
    for i in range(nt):
        t = tmin + tstep * i
        for j in range(nw):
            w = wmin + wstep * j
            K[i,j] = true_Kernel(t,w,t0) * np.sqrt(tstep*wstep)

    return K


def true_Kernel(x, y, t0):
    K = np.exp(-1.0 * x * y) # normalized

    return K

def SpfCoeff(w, svdrank, a, b, N, delta, VFunc):
    coeff = []
    for l in range(svdrank):
        val = calcIntegralSmearedSPF(a, b, N, w, VFunc[l], delta)
        coeff.append(val)

    return coeff

def calcIntegralSmearedSPF(a, b, N, w, rho, delta):
    # initialize
    h = (b - a) / N
    x = a
    s = 0.      # integrated value

    # computation
    for k in range(1, N):
        x = x + h
        s = s + true_smearingKernel(w, x, delta) * rho[k] 

    left = true_smearingKernel(w, a, delta) * rho[0] 
    right = true_smearingKernel(w, b, delta) * rho[-1]

    s = h * ( (left + right) * 0.5 + s )

    # Kernel normalization
    norm = NormSmearingKernel(a, b, N, w, delta)
    s = s / norm

    return s

def true_smearingKernel(x, y, delta):
    # S = LorentzianDelta(x,y,delta)
    # S = GaussianDelta(x,y,delta)
    # S = SincDelta(x,y,delta)
    # S = FermiDelta(x,y,delta)
    # S = ExponentialDelta(x,y,delta)
    S = BreitWignerDelta(x,y,delta)

    # S = 0.0
    # flag = np.abs(round(x-y, 8))
    # if flag == 0:
    #     #if x == y:
    #     S = 1.0
    return S

def NormSmearingKernel(a, b, N, w, delta):
    # initialize
    h = (b - a) / N
    x = a
    s = 0.      # integrated value

    # computation
    for k in range(1, N):
        x = x + h
        s = s + true_smearingKernel(w, x, delta) 

    left = true_smearingKernel(w, a, delta) 
    right = true_smearingKernel(w, b, delta)

    s = h * ( (left + right) * 0.5 + s )

    return s

def FittedRunJK(c, nt, edata_jk):

    ave = 0
    for j in range(nt):
        ave += c[j] * edata_jk[j]

    return ave


#=============================================================#
#=============================================================#
# smeared Kernel

def calcIntegralSmearedKernel(a, b, N, i, w, t0, delta):
    # initialize
    h = (b - a) / N
    x = a
    s = 0.      # integrated value

    # computation
    for k in range(1, N):
        x = x + h
        s = s + true_smearingKernel(w, x, delta) * true_Kernel(i, x, t0)

    left = true_smearingKernel(w, a, delta) * true_Kernel(i, a, t0)
    right = true_smearingKernel(w, b, delta) * true_Kernel(i, b, t0)

    s = h * ( (left + right) * 0.5 + s )

    return s


#=============================================================#
#=============================================================#
# reconstruction of the spectral function from modes

def calcIntegralReconstCt(a, b, N, t, bR):
    # initialize
    h = (b - a) / N
    x = a
    s = 0.      # integrated value

    # computation
    for k in range(1, N):
        x = x + h
        s = s + true_reconstKernel(t, x) * bR[k] 

    left = true_reconstKernel(t, a) * bR[0] 
    right = true_reconstKernel(t, b) * bR[-1]

    s = h * ( (left + right) * 0.5 + s )

    return s

def true_reconstKernel(x, y):
    K = np.exp(-1.0 * x * y) # normalized

    return K

def reconstSPF(w, VFunc, S_diag, rho_s, lmax):
    val = 0
    for l in range(lmax+1):
        val += rho_s[l] / S_diag[l] * VFunc[l][w]

    return val


def regularizedSPF(w, wstep, VFunc, S_diag, rho_s, lmax, regulator):
    val = 0
    for l in range(lmax+1):
        val += rho_s[l] * (S_diag[l]/(S_diag[l]**2 + regulator**2)) * VFunc[l][w]
    val = val / np.sqrt(wstep)

    return val

#=============================================================#
#=============================================================#
# Delta functions

def GaussianDelta(x,y,delta):
    w = delta * 0.5 / (np.sqrt(2.*np.log(2)))
    nume = np.exp(-(x-y)**2 / (2. * w**2))
    deno = np.sqrt(2.*np.pi) * w
    value = nume / deno

    return value

def LorentzianDelta(x,y,delta):
    w = delta * 0.5
    nume = w * 2.
    deno = np.pi * ((x-y)**2+w**2)
    value = nume / deno

    return value

def SincDelta(x,y,delta):
    w = 3.7909885 / delta
    nume = np.sinc(w * (x-y) / np.pi) * w
    deno = np.pi
    value = nume / deno

    return value

def FermiDelta(x,y,delta):
    w = delta * 0.25 / np.log(1.+np.sqrt(2))
    nume = 1.0
    deno = 4. * w * np.cosh((x-y)/(2.*w))**2
    value = nume / deno

    return value

def ExponentialDelta(x,y,delta):
    w = 2. * np.log(2)
    nume = np.exp(-np.abs(x-y)/w)
    deno = 2. * w
    value = nume / deno

    return value

def BreitWignerDelta(x,y,delta):
    nume = 2. * delta
    deno = (x - y)*(x - y) + delta**2
    S = 1. / np.pi * nume / deno
    value = nume / deno

    return value

#=============================================================#
#=============================================================#
# rectangular kernel

def calcIntegralSharpRect(a, b, N, rho, lower, upper):
    # initialize
    h = (b - a) / N
    x = a
    s = 0.      # integrated value

    # computation
    for k in range(1, N):
        x = x + h
        s = s + true_sharpRect(x, lower, upper) * rho[k] 

    left = true_sharpRect(a, lower, upper) * rho[0] 
    right = true_sharpRect(b, lower, upper) * rho[-1]

    s = h * ( (left + right) * 0.5 + s )

    return s

def true_sharpRect(x, lower, upper):
    S = 0.0
    if (x >= lower and x <= upper):
        S = 1.

    return S


def RectCoeff(svdrank, a, b, N, lower, upper, delta, VFunc, NormKernel):
    coeff = []
    for l in range(svdrank):
        val = calcIntegralSmearedRect(a, b, N, VFunc[l], lower, upper, delta, NormKernel)
        coeff.append(val)

    return coeff

def calcIntegralSmearedRect(a, b, N, VFunc_l, lower, upper, delta, NormKernel):
    # initialize
    h = (b - a) / N
    x = a
    s = 0.      # integrated value

    # computation
    for k in range(1, N):
        x = x + h
        s = s + true_smearingRect(x, lower, upper, delta) * VFunc_l[k] 

    left = true_smearingRect(a, lower, upper, delta) * VFunc_l[0] 
    right = true_smearingRect(b, lower, upper, delta) * VFunc_l[-1]

    s = h * ( (left + right) * 0.5 + s )

    # kernel normalization
    s = s / NormKernel[0] 

    return s


def true_smearingRect(x, lower, upper, delta):
    # S = ErrorFunctionRect(x,lower,upper,delta)
    S = SigmoidFunctionRect(x,lower,upper,delta)

    # S = 0.0
    # if (x >= lower and x <= upper):
    #     # S = 1. / (upper - lower)
    #     S = 1. 

    return S

def NormSmearingRect(a, b, N, lower, upper, delta):
    # initialize
    h = (b - a) / N
    x = a
    s = 0.      # integrated value

    # computation
    for k in range(1, N):
        x = x + h
        s = s + true_smearingRect(x, lower, upper, delta) 

    left = true_smearingRect(a, lower, upper, delta) 
    right = true_smearingRect(b, lower, upper, delta)

    s = h * ( (left + right) * 0.5 + s )

    return s

def ErrorFunctionRect(x,lower,upper,delta):
    left = (x-lower) / (np.sqrt(2) * delta)
    right = (x-upper) / (np.sqrt(2) * delta)
    value = 0.5 * (sp.special.erf(left) - sp.special.erf(right))

    return value

def SigmoidFunctionRect(x,lower,upper,delta):
    value = 1. / (1. + np.exp(-(x-lower) / delta))
    value = value - 1. / (1. + np.exp(-(x-upper) / delta))

    return value

def HalfMaxIndices(x):
    x = list(x)
    i_max = max(range(len(x)), key=lambda i: x[i])
    half = 0.5 * x[i_max]

    # left crossing
    for i in range(i_max, 0, -1):
        if x[i-1] < half <= x[i]:
            t = (half - x[i-1]) / (x[i] - x[i-1])
            i_left = (i-1) + t
            break

    # right crossing
    for i in range(i_max, len(x) - 1):
        if x[i] >= half > x[i+1]:
            t = (half - x[i]) / (x[i+1] - x[i])
            i_right = i + t
            break

    return i_max, int(i_left), int(i_right)


#=============================================================#
#=============================================================#
# Filtering of correlation function


def hfCtCoeff(t, svdrank, a, b, N, wcut_min, delta, VFunc):
    coeff = []
    for l in range(svdrank):
        val = calcIntegralHFilteringCt(a, b, N, t, VFunc[l], wcut_min, delta)
        coeff.append(val)

    return coeff

def calcIntegralHFilteringCt(a, b, N, t, VFunc_l, wcut_min, delta):
    # initialize
    h = (b - a) / N
    x = a
    s = 0.      # integrated value

    # computation
    for k in range(1, N):
        x = x + h
        s = s + true_hfilteringKernel(t, x, wcut_min, delta) * VFunc_l[k] 

    left = true_hfilteringKernel(t, a, wcut_min, delta) * VFunc_l[0] 
    right = true_hfilteringKernel(t, b, wcut_min, delta) * VFunc_l[-1]

    s = h * ( (left + right) * 0.5 + s )

    return s

def true_hfilteringKernel(x, y, wcut_min, delta):
    # S = ErrorFunctionStep(x,y,wcut_min,delta)
    S = SigmoidFunctionStep(x,y,wcut_min,delta)

    # S = 0.
    # if y > wcut_min:
    #     S = np.exp(-x*y)

    return S

def NormHFilteringKernel(a, b, N, t, wcut_min, delta):
    # initialize
    h = (b - a) / N
    x = a
    s = 0.      # integrated value

    # computation
    for k in range(1, N):
        x = x + h
        s = s + true_hfilteringKernel(t, x, wcut_min, delta) 

    left = true_hfilteringKernel(t, a, wcut_min, delta) 
    right = true_hfilteringKernel(t, b, wcut_min, delta)

    s = h * ( (left + right) * 0.5 + s )

    return s

def ErrorFunctionStep(x,y,wcut_min,delta):
    left = (y-wcut_min) / (np.sqrt(2) * delta)
    value = 0.5 * (1 + sp.special.erf(left))
    value = value * np.exp(-x*y)
    # value = np.exp(-x*y)

    return value

def SigmoidFunctionStep(x,y,wcut_min,delta):
    value = 1. / (1. + np.exp(-(y-wcut_min) / delta))
    value = value * np.exp(-x*y)


    return value

def UnityCheck(t1, fy1, fe1, y1, e1):
    mean = []
    error = []

    for t in range(len(t1)):
        mean.append(fy1[t]/y1[t])
        error.append(fe1[t])

    return mean, error


#=============================================================#
#=============================================================#
# energy moment 


def EMomCoeff(t, svdrank, a, b, N, poly, TFunc, VFunc):
    coeff = []
    for l in range(svdrank):
        val = calcIntegralEnergyMoment(a, b, N, t, VFunc[l], poly, TFunc)
        coeff.append(val)

    return coeff

def calcIntegralEnergyMoment(a, b, N, t, VFunc_l, poly, TFunc):
    # initialize
    h = (b - a) / N
    x = a
    s = 0.      # integrated value

    # computation
    for k in range(1, N):
        x = x + h
        s = s + true_energymomentKernel(t, x, poly, TFunc) * VFunc_l[k] 

    left = true_energymomentKernel(t, a, poly, TFunc) * VFunc_l[0] 
    right = true_energymomentKernel(t, b, poly, TFunc) * VFunc_l[-1]

    s = h * ( (left + right) * 0.5 + s )

    return s

def true_energymomentKernel(x, y, poly, TFunc):
    S = EnergyPolynomial(x,y,poly,TFunc)

    return S


def EnergyPolynomial(x,y,poly,TFunc):
    if poly == 0:
        value = np.exp(-x*y)
    elif poly == 9:
        value = (y-TFunc[int(x)]) * (y-TFunc[int(x)]) * np.exp(-x*y)
    else:
        value = (y**poly) * np.exp(-x*y)


    return value


#=============================================================#
#=============================================================#
# Exponential Upperbounds

def ExponentialUpperbounds(rho_l, lower, upper, CI):
    # data set
    x = []
    y = []
    s = []
    for l in range(lower, upper+1):
        x.append(l)
        y.append(np.abs(rho_l[l].mean))
        s.append(np.abs(rho_l[l].sdev))
    X = np.array(x)
    Y = np.array(y)
    S = np.array(s)
    print(np.log(Y))

    y_bound = []
    for l in range(len(Y)):
        y_bound.append(np.log(Y[l]+CI*S[l]))
    Y_bound = np.array(y_bound)
    n = len(X)

    # Objective: minimize sum f(x_i) = n*a + b*sum(x_i)
    c = np.array([n, np.sum(X)])

    # Constraints: a + b*x_i >= Y_bound_i
    # Convert to linprog form: -a - b*x_i <= -Y_bound_i
    A_ub = np.column_stack([-np.ones(n), -X])
    b_ub = -Y_bound

    # No bounds on a, b
    bounds = [(None, None), (None, None)]

    result = sp.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not result.success:
        raise RuntimeError("Linear upper-bound fit failed")

    a, b = result.x

    A = np.exp(a)
    B = -b

    return A, B

def LinearUpperbounds(rho_l, lower, upper, CI):
    # data set
    x = []
    y = []
    s = []
    for l in range(lower, upper+1):
        x.append(l)
        if rho_l[l][0] > 0:
            y.append(np.log(np.abs(rho_l[l][0]+CI*rho_l[l][1])))
        else:
            y.append(np.log(np.abs(rho_l[l][0]-CI*rho_l[l][1])))
    X = np.asarray(x, dtype=float)
    Y = np.asarray(y, dtype=float)

    def objective(z):
        a, B = z
        return np.exp(a) * np.sum(np.exp(-B*X))

    def upper_bound_constraint(z):
        a, B = z
        return a - B*X - Y

    constraints = ({
        'type': 'ineq',
        'fun': upper_bound_constraint
    },)

    a0 = 1.0
    B0 = 0.5
    
    res = minimize(
        objective,
        x0=np.array([a0, B0]),
        constraints=constraints,
        bounds=[(None, None), (0, None)],
        method="SLSQP",
    )
    
    a, B = res.x
    A = np.exp(a)

    return A, B

def ConstUpperbounds(rho_l, lower, upper, CI):
    y = []
    s = []
    for l in range(lower, upper+1):
        y.append(np.abs(rho_l[l][0]))
        s.append(np.abs(rho_l[l][1]))
    Y = np.array(y)
    S = np.array(s)

    return np.max(Y + CI * S)


def BasisUpperbounds(wref, VFuncA, wstep):

    # results
    bounds = []

    Nmax = len(VFuncA)
    for i in wref:
        bound_l = [] 
        norm_0 = np.abs(VFuncA[0][i]) / wstep
        for l in range(0, Nmax):
            bound_l.append(np.abs(VFuncA[l][i]) / wstep / norm_0)
        bounds.append(bound_l)

    return bounds

def TruncationError(wa_list, VFunc, Ntr, A_fit, B_fit):

    # results
    error = []

    for i in range(len(wa_list)):
        vpart = 1. - np.sum(np.array(VFunc).T[i]**2)
        rpart = A_fit**2 * (np.exp(-2*B_fit*Ntr)) / (1. - np.exp(-2.*B_fit) )
        error.append(np.sqrt(vpart * rpart))

    return error

def AbsTruncationError(wa_list, VFuncA, wstep, Ntr, A_fit, B_fit):

    # results
    error = []

    Nmax = len(VFuncA)
    for i in range(len(wa_list)):
        value = 0
        for l in range(Ntr, Nmax):
            value += np.abs(VFuncA[l][i])*np.abs(A_fit*np.exp(-B_fit*l))
        error.append(value / np.sqrt(wstep))

    return error


def KFuncUpperbounds(KFunc, lower, upper):
    #.mean data set
    x.sdev = []
    y = []
    for l in range(lower, upper+1):
        x.append(l)
        y.append(np.abs(KFunc[l]))
    X = np.array(x)
    Y = np.array(y)
    print(np.log(Y))

    y_bound = []
    for l in range(len(Y)):
        y_bound.append(np.log(Y[l]))
    Y_bound = np.array(y_bound)
    n = len(X)

    # Objective: minimize sum f(x_i) = n*a + b*sum(x_i)
    c = np.array([n, np.sum(X)])

    # Constraints: a + b*x_i >= Y_bound_i
    # Convert to linprog form: -a - b*x_i <= -Y_bound_i
    A_ub = np.column_stack([-np.ones(n), -X])
    b_ub = -Y_bound

    # No bounds on a, b
    bounds = [(None, None), (None, None)]

    result = sp.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not result.success:
        raise RuntimeError("KFunc upper-bound fit failed")

    a, b = result.x

    A = np.exp(a)
    B = -b

    return A, B

#=============================================================#
#=============================================================#
# Fit spectral function

def GvarFitSpectralFunction(x, y, dy, p0=None, plot=False):
    """
    Fit data with a Breit–Wigner distribution using gvar + lsqfit.

    Parameters
    ----------
    x : array_like
        x data
    y : array_like
        y data (central values)
    dy : array_like
        1-sigma errors for y
    p0 : dict, optional
        Initial guesses / priors:
        {
            'A'     : (mean, width),
            'x0'    : (mean, width),
            'Gamma' : (mean, width)
        }
    plot : bool
        If True, plot data and fit with uncertainty band

    Returns
    -------
    fit : lsqfit.nonlinear_fit
        Fit object (contains parameters, chi2, etc.)
    """

    # ----------------------------
    # Model
    # ----------------------------
    def breit_wigner(x, p):
        A     = p['A']
        x0    = p['x0']
        Gamma = p['Gamma']
        return A * (0.5 * Gamma)**2 / ((x - x0)**2 + (0.5 * Gamma)**2)

    # ----------------------------
    # gvar data
    # ----------------------------
    y_gv = gv.gvar(y, dy)

    # ----------------------------
    # Priors
    # ----------------------------
    if p0 is None:
        p0 = {
            'A'     : (max(y), max(y)),
            'x0'    : (x[y.argmax()], (x.max() - x.min()) / 2),
            'Gamma' : ((x.max() - x.min()) / 10, (x.max() - x.min()) / 2),
        }

    priors = {k: gv.gvar(v[0], v[1]) for k, v in p0.items()}

    # ----------------------------
    # Fit
    # ----------------------------
    fit = lsqfit.nonlinear_fit(
        data=(x, y_gv),
        fcn=breit_wigner,
        prior=priors
    )

    # ----------------------------
    # Optional plot
    # ----------------------------
    if plot:
        x_plot = np.linspace(x.min(), x.max(), 400)
        y_fit = breit_wigner(x_plot, fit.p)
        colors = plt.get_cmap("tab20c")(np.linspace(0, 1, 3))

        plt.errorbar(x, y, yerr=dy, fmt='o', label='data', color=colors[1], zorder=1)
        plt.plot(x_plot, gv.mean(y_fit), label='fit', color=colors[0], zorder=2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return fit

#=============================================================#
#=============================================================#
# Matrix elements with fitting

def Prediction(p, S_diag, UFunc, dof):
    mu = []
    nt = len(p)
    for i in range(dof):
        value = 0
        for a in range(nt):
            value += S_diag[a] * (UFunc[a][i+1] - UFunc[a][0] ) * p[a]
        mu.append(value)
        value = 0
    return mu

def Chisquare(a, x, y, J, bar, cov_jk_i, S_diag, UFunc):
    #=============================================================#
    #=============================================================#
    # Initialization
    dof = len(y)

    #=============================================================#
    #=============================================================#
    # Residual vector
    residual = np.array(y) - np.array(Prediction(a, S_diag, UFunc, dof))

    #=============================================================#
    #=============================================================#
    # Choresky decomposition of the covariance matrix
    L = np.linalg.cholesky(cov_jk_i)

    #=============================================================#
    #=============================================================#
    # Data vector
    y_data = np.reshape(residual, (dof,1))
    V_data = np.dot(L.I, y_data)

    #=============================================================#
    #=============================================================#
    # Total vector
    V_total = []
    size_total = dof
    for j in range(size_total):
        if j < dof:
            V_total.append(V_data[j][0,0])

    return V_total


def MatrixElement(param, x, y, J, bar, cov_jk_i, s_diag, ufunc):
    #=============================================================#
    #=============================================================#
    # least square fitting
    res_lsq = scipy.optimize.least_squares(Chisquare, param, args=(x, y, J, bar, cov_jk_i, s_diag, ufunc))
    result = (res_lsq.x)
    ME = []
    for a in range(len(result)):
        ME.append(result[a])

    return ME

#=============================================================#
#=============================================================#
# Misc

def PrintJKFile(data_list, name, directory):
    djkfile = './' + directory + '/' + name + '_jk'
    d = open(djkfile, 'w')

    for i in range(len(data_list)):
        for j in range(len(data_list[0])):
            strdata = str(j) + ' ' + str(data_list[i][j]) + '\n'
            d.write(strdata)
    d.close

    return 0


def PrintAVFile(data_list, name, directory, flag):
    davfile = './' + directory + '/' + name
    d = open(davfile, 'w')

    conf = len(data_list)
    dav_list = np.array(data_list).T

    for i in range(len(dav_list)):
        ave, err = StatisticalAnal(dav_list[i], conf, flag)
        strdata = str(i) + ' ' + str(ave) + ' ' + str(err) + '\n'
        d.write(strdata)
    d.close

    return 0

def PrintTOFile(data_list, err_list, name, directory, flag):
    davfile = './' + directory + '/' + name
    d = open(davfile, 'w')

    conf = len(data_list)
    dav_list = np.array(data_list).T

    for i in range(len(dav_list)):
        ave, err = StatisticalAnal(dav_list[i], conf, flag)
        strdata = str(i) + ' ' + str(ave) + ' ' + str(err) + ' ' + str(err_list[i]) + '\n'
        d.write(strdata)
    d.close

    return 0

def FigSingularValues(singularvalues_jk, nw):

    # Figure
    hcolors = plt.get_cmap("tab20c")(np.linspace(0, 1, 6))
    ax = []
    ay = []
    for l in range(len(singularvalues_jk)):
        ax.append(l)
        ay.append(singularvalues_jk[l])
    frun = PrintSingularValues(ax, ay, nw)
    plt.xlabel('$l$', fontsize=15)
    plt.ylabel('$\\sigma_l$', fontsize=15)
    #plt.ylim(1e-18, 100)
    plt.xlim(-1,60)
    plt.scatter(ax,ay, s=10, color=hcolors[0], label='Singular values')
    # plt.legend(fontsize=18)
    plt.yscale('log')
    plt.show()
    return 0


def FigInvertedSingularValues(singularvalues_jk, nw, regulator):

    # Figure
    hcolors = plt.get_cmap("tab20c")(np.linspace(0, 1, 3))
    ax = []
    ay = []
    az = []
    for l in range(len(singularvalues_jk)):
        ax.append(l)
        ay.append(1./singularvalues_jk[l])
        az.append(singularvalues_jk[l]/(singularvalues_jk[l]**2+regulator**2))
    plt.title('Inverted Singular values all one Gauge configurations')
    plt.xlabel('$l$', fontsize=15)
    plt.ylabel('$1/\\sigma_l$', fontsize=15)
    plt.scatter(ax,ay, s=5, color=hcolors[0], label='Naive')
    plt.scatter(ax,az, s=5, color=hcolors[1], label='Regularized')
    plt.legend(fontsize=18)
    plt.yscale('log')
    plt.show()
    return 0

def PrintSingularValues(ax, ay, nw):
    dsvfile = './singularvalues_data/sv-' + str(nw) + '_jk'
    d = open(dsvfile, 'w')

    for i in range(len(ax)):
        strdata = str(ax[i]) + ' ' + str(ay[i]) + '\n'
        d.write(strdata)
    d.close

    return 0

def FigBasisFunc(UFunc, t_list, VFunc, w_list, rank):
    urun = MScattFig(t_list, UFunc, rank)
    vrun = MScattFig(w_list, VFunc, rank)

    return 0



def PrintBasisFunc(UFunc, t_list, VFunc, w_list, rank, S_diag, cutoff, pwmin, pwmax, ptmin, ptmax):
    l = rank
    nt = len(UFunc[0])
    nw = len(VFunc[0])

    # V_Func
    vfile = './basis_data/VFunc_l=' + str(l) + 'nw=' + str(nw) + 'nt=' + str(nt) + 'ainv=' + str(cutoff)
    vfile += '_pwmin=' + str(pwmin) + 'pwmax=' + str(pwmax) + '_ptmin=' + str(ptmin) + 'ptmax=' + str(ptmax)
    v = open(vfile, 'w')

    for i in range(len(w_list)):
        strdata = str(w_list[i]*cutoff) + ' ' + str(S_diag[l]*VFunc[l][i]) + '\n'
        v.write(strdata)
    v.close

    # U_Func
    ufile = './basis_data/UFunc_l=' + str(l) + 'nw=' + str(nw) + 'nt=' + str(nt) + 'ainv=' + str(cutoff)
    ufile += '_pwmin=' + str(pwmin) + 'pwmax=' + str(pwmax) + '_ptmin=' + str(ptmin) + 'ptmax=' + str(ptmax)
    u = open(ufile, 'w')

    for i in range(len(t_list)):
        strdata = str(t_list[i]/cutoff) + ' ' + str(UFunc[l][i]) + '\n'
        u.write(strdata)
    u.close


    return 0


def MScattFig(xdata, ydata, size):
    # Figure
    hcolors = plt.get_cmap("tab20c")(np.linspace(0, 1, 6))
    for l in range(size):
        llabel = str(l) + '-th'
        ax = []
        ay = []
        for w in range(len(xdata)):
            ax.append(xdata[w])
            ay.append(ydata[l][w])
        plt.scatter(ax,ay, s=5, color=hcolors[l], label=llabel)
        plt.legend(fontsize=18)
    plt.xlabel('$t/w$', fontsize=15)
    plt.ylabel('$U/V_l$', fontsize=15)
    plt.show()

    return 0

def HeatMap(data):
    rank = len(data)

    fig, ax = plt.subplots()
    im = ax.imshow(data)
    plt.colorbar(im)
    plt.show()

    return 0

def FigSmearingFac(ffac, nw_0):

    # Figure
    hcolors = plt.get_cmap("tab20c")(np.linspace(0, 1, 6))
    ax = []
    ay = []
    axp = []
    axn = []
    ayp = []
    ayn = []
    for l in range(len(ffac[0])):
        ax.append(l)
        ay.append(ffac[nw_0][l])
        if ffac[nw_0][l] > 0:
            axp.append(l)
            ayp.append(ffac[nw_0][l])
        else:
            axn.append(l)
            ayn.append(-ffac[nw_0][l])
    frun = PrintSmearingFac(ax, ay, nw_0)
    plt.title('Smearing Kernel')
    plt.xlabel('$l$', fontsize=15)
    plt.ylabel('$|\\sum_{n_w}\\Delta_{n_w,n_w^\\prime}V_{l,n_w}|$', fontsize=15)
    plt.scatter(axp,ayp, s=20, color=hcolors[0], label='positive')
    plt.scatter(axn,ayn, s=20, facecolors='none', color=hcolors[0], label='negative')
    # plt.ylim(-0.01, 0.25)
    plt.show()
    return 0

def PrintSmearingFac(ax, ay, nw_0):
    dsmfile = './smearingfactor_data/sm-' + str(nw_0) + '_jk'
    d = open(dsmfile, 'w')

    for i in range(len(ax)):
        strdata = str(ax[i]) + ' ' + str(ay[i]) + '\n'
        d.write(strdata)
    d.close

    return 0

def FigRhoAbs(rho_l, rank):

    # Figure
    hcolors = plt.get_cmap("tab20c")(np.linspace(0, 1, 6))
    axp = []
    axn = []
    ayp = []
    ayn = []
    for l in range(0, rank):
        if rho_l[l][0] > 0:
            axp.append(l)
            ayp.append(rho_l[l][0])
        else:
            axn.append(l)
            ayn.append(-rho_l[l][0])
    plt.xlabel('$l$', fontsize=15)
    plt.ylabel('$|\\rho_l|$', fontsize=15)
    plt.scatter(axp,ayp, s=20, color=hcolors[0], label='positive')
    plt.scatter(axn,ayn, s=20, facecolors='none', color=hcolors[0], label='negative')
    plt.yscale('log')
    plt.show()
    return 0


def FigRhoAbsWithUpperBound(rho_l, rank, A_fit, B_fit):

    # Figure
    hcolors = plt.get_cmap("tab20c")(np.linspace(0, 1, 6))

    # data points
    axp = []
    axn = []
    ayp = []
    ayn = []
    aep = []
    aen = []
    for l in range(0, rank):
        if rho_l[l][0] > 0:
            axp.append(l)
            ayp.append(rho_l[l][0])
            aep.append(rho_l[l][1])
        else:
            axn.append(l)
            ayn.append(-rho_l[l][0])
            aen.append(rho_l[l][1])
    plt.scatter(axp,ayp, s=20, color=hcolors[0], label='positive')
    plt.scatter(axn,ayn, s=20, facecolors='none', color=hcolors[0], label='negative')
    plt.errorbar(axp,ayp, yerr=aep, capsize=5, fmt=' ', markersize=0, color=hcolors[0])
    plt.errorbar(axn,ayn, yerr=aen, capsize=5, fmt=' ', markersize=0, color=hcolors[0])

    # upper bounds
    X_plot = np.linspace(0, rank, 200)
    Y_plot = A_fit * np.exp(-B_fit * X_plot)
    plt.plot(X_plot, Y_plot, color=hcolors[1], label='Upper bound fit', linewidth=2)


    plt.xlabel('$l$', fontsize=15)
    plt.ylabel('$|\\rho_l|$', fontsize=15)
    plt.yscale('log')
    plt.ylim(0.0001, 10)
    plt.xlim(-0.5, 13)
    plt.savefig('./results/SpecFunc/ReconstRhoSize.pdf')
    plt.show()
    return 0

def FigBasisUpperbounds(wa_list, wref, bounds):

    # Figure
    hcolors = plt.get_cmap("tab20c")(np.linspace(0, 1, 6))
    x = []
    y = []
    for i in range(len(wref)):
        slabel = '$\\omega a=$' + str(wa_list[wref[i]])
        Nmax = len(bounds[0])
        for l in range(0, Nmax):
            x.append(l)
            y.append(bounds[i][l])
        plt.xlabel('$l$', fontsize=15)
        plt.ylabel('$|V_l(\\omega)|/|V_0(\\omega)|$', fontsize=15)
        plt.scatter(x,y, s=20, color=hcolors[0], label=slabel)
        plt.legend()
        plt.yscale('log')
        plt.show()
    return 0


def FigBasisUpperboundsComp(wa_list, wref, Vbounds, Kbounds, delta, Ne, A, B):

    # Figure
    hcolors = plt.get_cmap("tab20c")(np.linspace(0, 1, 6))
    x = []
    y = []
    z = []
    for i in range(len(wref)):
        stitle = '$\\omega a=$' + str(wa_list[wref[i]])
        slabel = 'Smeared with $\\Delta=$' + str(delta)
        Nmax = len(Kbounds[0])
        for l in range(0, Nmax):
            x.append(l)
            y.append(Vbounds[i][l] * np.exp(-B*l))
            z.append(Kbounds[i][l] * np.exp(-B*l))
        ymax = 1.2 * max(y)
        ymin = 0.8 * min(z)
        plt.title(stitle)
        plt.xlabel('$l$', fontsize=15)
        plt.ylabel('$|\\rho_l V_l(\\omega)|/|\\rho_0 V_0(\\omega)|$', fontsize=15)
        plt.scatter(x,y, s=20, color=hcolors[0], label='Original')
        plt.scatter(x,z, s=20, facecolors='none', color=hcolors[0], label=slabel)
        plt.vlines(Ne, ymin, ymax, colors='black', linestyle='dashed', linewidth=2)
        plt.legend()
        plt.yscale('log')
        plt.show()
    return 0

def FigKFuncAbsWithUpperBound(KFunc, rank, A_fit, B_fit):

    # Figure
    hcolors = plt.get_cmap("tab20c")(np.linspace(0, 1, 6))

    # data points
    axp = []
    axn = []
    ayp = []
    ayn = []
    for l in range(0, rank):
        if KFunc[l] > 0:
            axp.append(l)
            ayp.append(KFunc[l])
        else:
            axn.append(l)
            ayn.append(-KFunc[l])
    plt.scatter(axp,ayp, s=20, color=hcolors[0], label='positive')
    plt.scatter(axn,ayn, s=20, facecolors='none', color=hcolors[0], label='negative')

    # upper bounds
    X_plot = np.linspace(0, rank, 200)
    Y_plot = A_fit * np.exp(-B_fit * X_plot)
    plt.plot(X_plot, Y_plot, color=hcolors[1], label='Upper bound fit', linewidth=2)


    plt.xlabel('$l$', fontsize=15)
    plt.ylabel('$|S_{{\\rm{smr}}l}|$', fontsize=15)
    plt.yscale('log')
    plt.show()
    return 0


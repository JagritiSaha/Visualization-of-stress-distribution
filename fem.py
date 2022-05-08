# A program to calculate the stresses/strains induced in a plate with hole
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axisartist.axislines import Subplot
from traits.api import HasTraits, observe, Range, Enum, Float, Str, Int
from traitsui.api import Item, View, Group

def void_mesh(d1, d2, p, m, R, element_type):
    q = np.array([[0, 0], [d1, 0], [0, d2], [d1, d2]])
    PD = 2
    NoN = 2 * (p + 1) * (m + 1) + 2 * (p - 1) * (m + 1)
    NoE = 4 * p * m
    NPE = 4  # D2QU4N

    ### Node ###

    NL = np.zeros([NoN, PD])
    a = (q[1, 0] - q[0, 0]) / p  # Increment in horizontal direction
    b = (q[2, 1] - q[0, 1]) / p  # Increment in vertical direction

    ### Region 1 ###

    coor11 = np.zeros([(p + 1) * (m + 1), PD])

    for i in range(1, p + 2):
        coor11[i - 1, 0] = q[0, 0] + (i - 1) * a
        coor11[i - 1, 1] = q[0, 1]

    for i in range(1, p + 2):
        coor11[m * (p + 1) + i - 1, 0] = R * np.cos((5 * math.pi / 4) + (i - 1) * ((math.pi / 2) / p)) + d1 / 2
        coor11[m * (p + 1) + i - 1, 1] = R * np.sin((5 * math.pi / 4) + (i - 1) * ((math.pi / 2) / p)) + d2 / 2

    for i in range(1, m):
        for j in range(1, p + 2):
            dx = (coor11[m * (p + 1) + j - 1, 0] - coor11[j - 1, 0]) / m
            dy = (coor11[m * (p + 1) + j - 1, 1] - coor11[j - 1, 1]) / m
            coor11[i * (p + 1) + j - 1, 0] = coor11[(i - 1) * (p + 1) + j - 1, 0] + dx
            coor11[i * (p + 1) + j - 1, 1] = coor11[(i - 1) * (p + 1) + j - 1, 1] + dy

    ### Region 2 ###

    coor22 = np.zeros([(p + 1) * (m + 1), PD])

    for i in range(1, p + 2):
        coor22[i - 1, 0] = q[2, 0] + (i - 1) * a
        coor22[i - 1, 1] = q[2, 1]

    for i in range(1, p + 2):
        coor22[m * (p + 1) + i - 1, 0] = R * np.cos((3 * math.pi / 4) - (i - 1) * ((math.pi / 2) / p)) + d1 / 2
        coor22[m * (p + 1) + i - 1, 1] = R * np.sin((3 * math.pi / 4) - (i - 1) * ((math.pi / 2) / p)) + d2 / 2

    for i in range(1, m):
        for j in range(1, p + 2):
            dx = (coor22[m * (p + 1) + j - 1, 0] - coor22[j - 1, 0]) / m
            dy = (coor22[m * (p + 1) + j - 1, 1] - coor22[j - 1, 1]) / m
            coor22[i * (p + 1) + j - 1, 0] = coor22[(i - 1) * (p + 1) + j - 1, 0] + dx
            coor22[i * (p + 1) + j - 1, 1] = coor22[(i - 1) * (p + 1) + j - 1, 1] + dy

    ### Region 3 ###

    coor33 = np.zeros([(p - 1) * (m + 1), PD])

    for i in range(1, p):
        coor33[i - 1, 0] = q[0, 0]
        coor33[i - 1, 1] = q[0, 1] + i * b

    for i in range(1, p):
        coor33[m * (p - 1) + i - 1, 0] = R * np.cos((5 * math.pi / 4) - (i) * ((math.pi / 2) / p)) + d1 / 2
        coor33[m * (p - 1) + i - 1, 1] = R * np.sin((5 * math.pi / 4) - (i) * ((math.pi / 2) / p)) + d2 / 2

    for i in range(1, m):
        for j in range(1, p):
            dx = (coor33[m * (p - 1) + j - 1, 0] - coor33[j - 1, 0]) / m
            dy = (coor33[m * (p - 1) + j - 1, 1] - coor33[j - 1, 1]) / m
            coor33[i * (p - 1) + j - 1, 0] = coor33[(i - 1) * (p - 1) + j - 1, 0] + dx
            coor33[i * (p - 1) + j - 1, 1] = coor33[(i - 1) * (p - 1) + j - 1, 1] + dy

    ### Region 4 ###

    coor44 = np.zeros([(p - 1) * (m + 1), PD])

    for i in range(1, p):
        coor44[i - 1, 0] = q[1, 0]
        coor44[i - 1, 1] = q[1, 1] + i * b

    for i in range(1, p):
        coor44[m * (p - 1) + i - 1, 0] = R * np.cos((7 * math.pi / 4) + (i) * ((math.pi / 2) / p)) + d1 / 2
        coor44[m * (p - 1) + i - 1, 1] = R * np.sin((7 * math.pi / 4) + (i) * ((math.pi / 2) / p)) + d2 / 2

    for i in range(1, m):
        for j in range(1, p):
            dx = (coor44[m * (p - 1) + j - 1, 0] - coor44[j - 1, 0]) / m
            dy = (coor44[m * (p - 1) + j - 1, 1] - coor44[j - 1, 1]) / m
            coor44[i * (p - 1) + j - 1, 0] = coor44[(i - 1) * (p - 1) + j - 1, 0] + dx
            coor44[i * (p - 1) + j - 1, 1] = coor44[(i - 1) * (p - 1) + j - 1, 1] + dy

    ### REORDERING THE NODES ###

    for i in range(1, m + 2):
        NL[(i - 1) * 4 * p:i * 4 * p, :] = np.vstack([coor11[(i - 1) * (p + 1):(i) * (p + 1), :],
                                                      coor44[(i - 1) * (p - 1):(i) * (p - 1), :],
                                                      np.flipud(coor22[(i - 1) * (p + 1):(i) * (p + 1), :]),
                                                      np.flipud(coor33[(i - 1) * (p - 1):(i) * (p - 1), :])])

    ### Element ###

    EL = np.zeros([NoE, NPE])

    for i in range(1, m + 1):
        for j in range(1, 4 * p + 1):
            if j == 1:
                EL[(i - 1) * (4 * p) + j - 1, 0] = (i - 1) * (4 * p) + j
                EL[(i - 1) * (4 * p) + j - 1, 1] = EL[(i - 1) * (4 * p) + j - 1, 0] + 1
                EL[(i - 1) * (4 * p) + j - 1, 3] = EL[(i - 1) * (4 * p) + j - 1, 0] + 4 * p
                EL[(i - 1) * (4 * p) + j - 1, 2] = EL[(i - 1) * (4 * p) + j - 1, 3] + 1

            elif j == 4 * p:
                EL[(i - 1) * (4 * p) + j - 1, 0] = i * (4 * p)
                EL[(i - 1) * (4 * p) + j - 1, 1] = (i - 1) * (4 * p) + 1
                EL[(i - 1) * (4 * p) + j - 1, 2] = EL[(i - 1) * (4 * p) + j - 1, 0] + 1
                EL[(i - 1) * (4 * p) + j - 1, 3] = EL[(i - 1) * (4 * p) + j - 1, 0] + 4 * p

            else:
                EL[(i - 1) * (4 * p) + j - 1, 0] = EL[(i - 1) * (4 * p) + j - 2, 1]
                EL[(i - 1) * (4 * p) + j - 1, 3] = EL[(i - 1) * (4 * p) + j - 2, 2]
                EL[(i - 1) * (4 * p) + j - 1, 2] = EL[(i - 1) * (4 * p) + j - 1, 3] + 1
                EL[(i - 1) * (4 * p) + j - 1, 1] = EL[(i - 1) * (4 * p) + j - 1, 0] + 1
    EL = EL.astype(int)
    return NL, EL

def assign_BCs(NL, BC_flag, defV, d1, d2):
    NoN = np.size(NL,0) # Number of nodes
    PD = np.size(NL,1) # Problem Dimension
    ENL = np.zeros([NoN, 6*PD])
    ENL[:,0:PD] = NL
    if BC_flag == 'extension':
        for i in range(0,NoN):
            if ENL[i,0] == 0:
                ENL[i,2] = -1
                ENL[i,3] = -1
                ENL[i,8] = -defV
                ENL[i,9] = 0
            elif ENL[i,0] == d1:
                ENL[i,2] = -1
                ENL[i,3] = -1
                ENL[i,8] = defV
                ENL[i,9] = 0
            else:
                ENL[i, 2] = 1
                ENL[i, 3] = 1
                ENL[i, 10] = 0
                ENL[i, 11] = 0

    if BC_flag == 'expansion':
        for i in range(0,NoN):
            if ENL[i,0] == 0 or ENL[i,0] == d1 or ENL[i,1] == 0 or ENL[i,1] == d2:
                ENL[i,2] = -1
                ENL[i,3] = -1
                ENL[i,8] = defV*ENL[i,0]
                ENL[i,9] = defV*ENL[i,1]
            else:
                ENL[i,2] = 1
                ENL[i,3] = 1
                ENL[i,10] = 0
                ENL[i,11] = 0

    if BC_flag == 'shear':
        for i in range(0,NoN):
            if ENL[i,1] == 0:
                ENL[i,2] = -1
                ENL[i,3] = -1
                ENL[i,8] = 0
                ENL[i,9] = 0
            elif ENL[i,1] == d2:
                ENL[i,2] = -1
                ENL[i,3] = -1
                ENL[i,8] = defV
                ENL[i,9] = 0
            else:
                ENL[i,2] = 1
                ENL[i,3] = 1
                ENL[i,10] = 0
                ENL[i,11] = 0

    DOFs = 0
    DOCs = 0

    for i in range(0,NoN):
        for j in range(0,PD):
            if ENL[i,PD+j] == -1:
                DOCs -= 1
                ENL[i, 2*PD+j] = DOCs
            else:
                DOFs += 1
                ENL[i,2*PD+j] = DOFs

    for i in range(0,NoN):
        for j in range(0,PD):
            if ENL[i,2*PD+j] < 0:
                ENL[i,3*PD+j] = abs(ENL[i,2*PD+j])+DOFs
            else:
                ENL[i,3*PD+j] = abs(ENL[i,2*PD+j])
    DOCs = abs(DOCs)
    return (ENL, DOFs, DOCs)


def element_stiffness(nl, NL, E, nu, GPE):
    NPE = np.size(nl,0)
    PD = np.size(NL,1)
    x = np.zeros([NPE,PD])
    x[0:NPE,0:PD] = NL[nl[0:NPE]-1,0:PD]
    K = np.zeros([NPE*PD,NPE*PD])
    coor = x.T

    for i in range(1, NPE+1):
        for j in range (1, NPE+1):
            k = np.zeros([PD, PD])
            for gp in range(1, GPE+1):
                J = np.zeros([PD,PD])
                grad =  np.zeros([PD,NPE])
                (xi, eta, alpha) = GaussPoint(NPE, GPE, gp)
                grad_nat = grad_N_nat(NPE, xi, eta)
                J = coor@grad_nat.T
                grad = np.linalg. inv(J).T@grad_nat
                for a in range(1, PD+1):
                    for c in range(1, PD+1):
                        for b in range(1, PD+1):
                            for d in range(1, PD+1):
                                k[a-1,c-1] = k[a-1,c-1] + grad[b-1,i-1]*constitutive(a,b,c,d,E,nu)*grad[d-1,j-1]*np.linalg.det(J)*alpha
                K[((i-1)*PD+1)-1:i*PD , ((j-1)*PD+1)-1:j*PD] = k
    return K


def GaussPoint(NPE, GPE, gp):
    if NPE == 3:
        if GPE == 1:
            if gp == 1:
                xi = 1/3
                eta = 1/3
                alpha = 1

    if NPE == 4:
        if GPE == 1:
            if gp == 1:
                xi = 0
                eta = 0
                alpha = 4
        if GPE == 4:
            if gp == 1:
                xi = -1/math.sqrt(3)
                eta = -1/math.sqrt(3)
                alpha = 1
            if gp == 2:
                xi = 1/math.sqrt(3)
                eta = -1/math.sqrt(3)
                alpha = 1
            if gp == 3:
                xi = 1/math.sqrt(3)
                eta = 1/math.sqrt(3)
                alpha = 1
            if gp == 4:
                xi = -1/math.sqrt(3)
                eta = 1/math.sqrt(3)
                alpha = 1

    return (xi, eta, alpha)

def grad_N_nat(NPE, xi, eta):
    PD = 2
    result = np.zeros([PD, NPE])

    if NPE == 3:
        result[0,0] = 1
        result[0,1] = 0
        result[0,2] = -1
        result[1,0] = 0
        result[1,1] = 1
        result[1,2] = -1

    if NPE == 4:
        result[0,0] = -(1/4)*(1 - eta)
        result[0,1] = (1/4)*(1 - eta)
        result[0,2] = (1/4)*(1 + eta)
        result[0,3] = -(1/4)*(1 + eta)

        result[1,0] = -(1/4)*(1 - xi)
        result[1,1] = -(1/4)*(1 + xi)
        result[1,2] = (1/4)*(1 + xi)
        result[1,3] = (1/4)*(1 - xi)

    return result

def constitutive(i, j, k, l, E, nu):
    C = (E/(2*(1+nu))) * (delta(i,l)*delta(j,k) + delta(i,k)*delta(j,l)) + (E*nu)/(1-nu**2) * delta(i,j)*delta(k,l)
    return C

def delta(i, j):
    if i == j:
        delta = 1
    else:
        delta = 0
    return delta



def assemble_stiffness(ENL, EL, NL, E, nu, GPE):
    NoE = np.size(EL,0)
    NPE = np.size(EL,1)
    NoN = np.size(NL,0)
    PD = np.size(NL,1)

    K = np.zeros([NoN*PD,NoN*PD])

    for i in range(1, NoE+1):
        nl = EL[i-1,0:NPE]
        k = element_stiffness(nl, NL, E, nu, GPE)
        for r in range(0,NPE):
            for p in range(0,PD):
                for q in range(0,NPE):
                    for s in range(0,PD):
                        row = ENL[nl[r]-1, p+3*PD]
                        column = ENL[nl[q]-1, s+3*PD]
                        value = k[r*PD+p, q*PD+s]
                        K[int(row)-1,int(column)-1] = K[int(row)-1, int(column)-1] +value
    return K

def assemble_displacements(ENL,NL):
    NoN = np.size(NL, 0)
    PD = np.size(NL,1)
    DOC = 0
    Up = []
    for i in range(0,NoN):
        for j in range(0,PD):
            if ENL[i,PD+j] == -1:
                DOC += 1
                Up.append(ENL[i,4*PD+j])
    Up = np.vstack([Up]).reshape(-1,1)
    return Up


def assemble_forces(ENL,NL):
    NoN = np.size(NL,0)
    PD = np.size(NL,1)
    DOF = 0
    Fp = []
    for i in range(0,NoN):
        for j in range(0,PD):
            if ENL[i, PD+j] == 1:
                DOF += 1
                Fp.append(ENL[i,5*PD+j])
    Fp = np.vstack([Fp]).reshape(-1,1)
    return Fp


def update_nodes(ENL, Uu, NL, Fu):
    NoN = np.size(NL,0)
    PD = np.size(NL,1)
    DOFs = 0
    DOCs = 0
    for i in range(0,NoN):
        for j in range(0,PD):
            if ENL[i,PD+j] == 1:
                DOFs += 1
                ENL[i,4*PD+j] = Uu[DOFs-1]
            else:
                DOCs += 1
                ENL[i,5*PD+j] = Fu[DOCs-1]
    return ENL


def dyad(u,v):
    u = u.reshape(len(v),1)
    v = v.reshape(len(v),1)
    PD = 2
    A = u @ v.T
    return A


def element_post_process(NL,EL,ENL,GPE,E,nu):
    PD = np.size(NL,1)
    NoE = np.size(EL,0)
    NPE = np.size(EL,1)
    disp = np.zeros([NoE,NPE,PD,1])
    stress = np.zeros([NoE,GPE,PD,PD])
    strain = np.zeros([NoE,GPE,PD,PD])

    for e in range(1,NoE+1):
        nl = EL[e-1,0:NPE]
        for i in range(1,NPE+1):
            for j in range(1,PD+1):
                disp[e-1,i-1,j-1,0] = ENL[nl[i-1]-1,4*PD+j-1]
        x = np.zeros([NPE, PD])
        x[0:NPE, 0:PD] = NL[nl[0:NPE] - 1, 0:PD]
        u = np.zeros([PD, NPE])
        for i in range(1, NPE + 1):
            for j in range(1, PD + 1):
                u[j - 1, i - 1] = ENL[nl[i - 1] - 1, 4 * PD + j - 1]

        coor = x.T

        for gp in range(1,GPE+1):
            epsilon = np.zeros([PD,PD])
            for i in range(1,NPE+1):
                J = np.zeros([PD,PD])
                grad = np.zeros([PD,NPE])
                (xi, eta, alpha) = GaussPoint(NPE, GPE, gp)
                grad_nat = grad_N_nat(NPE, xi, eta)
                J = coor @ grad_nat.T
                grad = np.linalg.inv(J).T @ grad_nat
                epsilon = epsilon + 1/2 * (dyad(grad[:,i-1],u[:,i-1]) + dyad(u[:,i-1],grad[:,i-1]))
            sigma = np.zeros([PD,PD])
            for a in range(1,PD+1):
                for b in range(1,PD+1):
                    for c in range(1,PD+1):
                        for d in range(1,PD+1):
                            sigma[a-1,b-1] = sigma[a-1,b-1] + constitutive(a,b,c,d,E,nu)*epsilon[c-1,d-1]
            for a in range(1,PD+1):
                for b in range(1,PD+1):
                    strain[e-1,gp-1,a-1,b-1] = epsilon[a-1,b-1]
                    stress[e-1,gp-1,a-1,b-1] = sigma[a-1,b-1]
    return disp, stress, strain


def post_process(NL,EL,ENL,scale,GPE,E,nu):
    PD = np.size(NL,1)
    NoE = np.size(EL,0)
    NPE = np.size(EL,1)

    disp, stress, strain = element_post_process(NL,EL,ENL,GPE,E,nu)

    stress_xx = np.zeros([NPE,NoE])
    stress_xy = np.zeros([NPE,NoE])
    stress_yx = np.zeros([NPE,NoE])
    stress_yy = np.zeros([NPE,NoE])

    strain_xx = np.zeros([NPE,NoE])
    strain_xy = np.zeros([NPE,NoE])
    strain_yx = np.zeros([NPE,NoE])
    strain_yy = np.zeros([NPE,NoE])

    disp_x = np.zeros([NPE,NoE])
    disp_y = np.zeros([NPE,NoE])

    X = np.zeros([NPE,NoE])
    Y = np.zeros([NPE,NoE])

    if NPE in [3,4]:
        X = ENL[EL-1,0] + scale*ENL[EL-1,4*PD]
        Y = ENL[EL-1,1] + scale*ENL[EL-1,4*PD+1]
        X = X.T
        Y = Y.T

        stress_xx[:,:] = stress[:,:,0,0].T
        stress_xy[:,:] = stress[:,:,0,1].T
        stress_yx[:,:] = stress[:,:,1,0].T
        stress_yy[:,:] = stress[:,:,1,1].T

        strain_xx[:,:] = strain[:,:,0,0].T
        strain_xy[:,:] = strain[:,:,0,1].T
        strain_yx[:,:] = strain[:,:,1,0].T
        strain_yy[:,:] = strain[:,:,1,1].T

        disp_x = disp[:,:,0,0].T
        disp_y = disp[:,:,1,0].T
    return (stress_xx,stress_xy,stress_yx,stress_yy,strain_xx,strain_xy,strain_yx,strain_yy,disp_x,disp_y,X,Y)


def truncate_colormap(cmap, minval = 0.0, maxval = 1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap

class fem(HasTraits):
    d1 = Float(0.5,auto_set=False, enter_set=True)
    d2 = Float(0.5,auto_set=False, enter_set=True)
    p = Int(6,auto_set=False, enter_set=True)
    m = Int(6,auto_set=False, enter_set=True)
    R = Float(0.2,auto_set=False, enter_set=True)
    element_type = Enum('Rectangular',auto_set=False,enter_set=True)
    defV = Float(0.1,auto_set=False, enter_set=True)
    BC_flag = Enum('extension','expansion','shear',auto_set=False,enter_set=True)
    E = Float(8/3,auto_set=False, enter_set=True)
    nu = Float(1/3,auto_set=False, enter_set=True)
    GPE = Int(4,auto_set=False, enter_set=True)
    scale = Float(1,auto_set=False, enter_set=True)
    analysis_type = Enum('Stress XX','Stress XY','Stress YX','Stress YY','Strain XX','Strain XY','Strain YX','Strain YY','Disp X',
                         'Disp Y',auto_set=False,enter_set=True)

    view = View(
        Item('d1', label='X - dimension'),
        Item('d2', label='Y - dimension'),
        Item('p', label='Partition along X and Y'),
        Item('m', label='Partition along the diagonal'),
        Item('R', label='Radius of the hole'),
        Item('element_type', label='Element Type'),
        Item('defV', label='Initial displacement'),
        Item('BC_flag', label='Boundary Conditions'),
        Item('E', label='Modulus Of Elasticity'),
        Item('nu', label='Poission Ratio'),
        Item('GPE', label='Gauss Points Per Element'),
        Item('scale', label='Scale'),
        Item('analysis_type',label='Analysis Type'),
        #show_border=True,
        resizable=True
        #buttons=[OkButton, CancelButton]
                )

    @observe('d1,d2,p,m,R,element_type,defV,BC_flag,E,nu,GPE,scale,analysis_type')
    def update_plot(self, event=None):
        D1 = self.d1
        D2 = self.d2
        P = self.p
        M = self.m
        Ri = self.R
        Element_type = self.element_type
        DefV = self.defV
        BC_Flag = self.BC_flag
        Ei = self.E
        nui = self.nu
        GPEi = self.GPE
        Scale = self.scale
        Analysis_type = self.analysis_type
        if Element_type == 'Rectangular':
            NL,EL = void_mesh(D1,D2,P,M,Ri,Element_type)
        if Element_type == 'Triangular':
            NL,EL = void_mesh_tri(D1,D2,P,M,Ri,Element_type)
        (ENL, DOFs, DOCs) = assign_BCs(NL,BC_Flag,DefV,D1,D2)
        if Element_type == 'Rectangular':
            K = assemble_stiffness(ENL,EL,NL,Ei,nui,GPEi)
        if Element_type == 'Triangular':
            K = assemble_stiffness_tri(ENL,EL,NL,E,nu,GPE)
        Fp = assemble_forces(ENL, NL)
        Up = assemble_displacements(ENL, NL)
        K_reduced = K[0:DOFs, 0:DOFs]
        K_UP = K[0:DOFs, DOFs:DOCs + DOFs]
        K_PU = K[DOFs:DOCs + DOFs, 0:DOFs]
        K_PP = K[DOFs:DOCs + DOFs, DOFs:DOCs + DOFs]
        F = Fp - (K_UP @ Up)
        Uu = np.linalg.solve(K_reduced, F)
        Fu = (K_PU @ Uu) + (K_PP @ Up)
        ENL = update_nodes(ENL, Uu, NL, Fu)
        (stress_xx, stress_xy, stress_yx, stress_yy, strain_xx, strain_xy, strain_yx, strain_yy, disp_x, disp_y, X,
         Y) = post_process(NL, EL, ENL, Scale, GPEi, Ei,nui)
        if Analysis_type == 'Stress XX':
            stress_xxNormalized = (stress_xx - stress_xx.min()) / (stress_xx.max() - stress_xx.min())
            plt.close('all')
            fig_1 = plt.figure(1)
            plt.axis('off')
            plt.show()
            plt.title('Stress XX')
            axstress_xx = Subplot(fig_1, 111)
            fig_1.add_subplot(axstress_xx)
            axstress_xx.axis['left'].set_visible(True)
            axstress_xx.axis['bottom'].set_visible(True)


            for i in range(np.size(EL, 0)):
                x = X[:, i]
                y = Y[:, i]
                c = stress_xxNormalized[:, i]
                cmap = truncate_colormap(plt.get_cmap('jet'), c.min(), c.max())
                t = axstress_xx.tripcolor(x, y, c, cmap=cmap, shading='gouraud')
                p = axstress_xx.plot(x, y, 'k-', linewidth=0.5)
                plt.show()
        if Analysis_type == 'Stress XY':
            stress_xxNormalized = (stress_xy - stress_xy.min()) / (stress_xy.max() - stress_xy.min())
            plt.close('all')
            fig_1 = plt.figure(1)
            plt.axis('off')
            plt.show()
            plt.title('Stress XY')
            axstress_xx = Subplot(fig_1, 111)
            fig_1.add_subplot(axstress_xx)
            axstress_xx.axis['left'].set_visible(True)
            axstress_xx.axis['bottom'].set_visible(True)


            for i in range(np.size(EL, 0)):
                x = X[:, i]
                y = Y[:, i]
                c = stress_xxNormalized[:, i]
                cmap = truncate_colormap(plt.get_cmap('jet'), c.min(), c.max())
                t = axstress_xx.tripcolor(x, y, c, cmap=cmap, shading='gouraud')
                p = axstress_xx.plot(x, y, 'k-', linewidth=0.5)
                plt.show()
        if Analysis_type == 'Stress YX':
            stress_xxNormalized = (stress_yx - stress_yx.min()) / (stress_yx.max() - stress_yx.min())
            plt.close('all')
            fig_1 = plt.figure(1)
            plt.axis('off')
            plt.show()
            plt.title('Stress YX')
            axstress_xx = Subplot(fig_1, 111)
            fig_1.add_subplot(axstress_xx)
            axstress_xx.axis['left'].set_visible(True)
            axstress_xx.axis['bottom'].set_visible(True)


            for i in range(np.size(EL, 0)):
                x = X[:, i]
                y = Y[:, i]
                c = stress_xxNormalized[:, i]
                cmap = truncate_colormap(plt.get_cmap('jet'), c.min(), c.max())
                t = axstress_xx.tripcolor(x, y, c, cmap=cmap, shading='gouraud')
                p = axstress_xx.plot(x, y, 'k-', linewidth=0.5)
                plt.show()
        if Analysis_type == 'Stress YY':
            stress_xxNormalized = (stress_yy - stress_yy.min()) / (stress_yy.max() - stress_yy.min())
            plt.close('all')
            fig_1 = plt.figure(1)
            plt.axis('off')
            plt.show()
            plt.title('Stress YY')
            axstress_xx = Subplot(fig_1, 111)
            fig_1.add_subplot(axstress_xx)
            axstress_xx.axis['left'].set_visible(True)
            axstress_xx.axis['bottom'].set_visible(True)


            for i in range(np.size(EL, 0)):
                x = X[:, i]
                y = Y[:, i]
                c = stress_xxNormalized[:, i]
                cmap = truncate_colormap(plt.get_cmap('jet'), c.min(), c.max())
                t = axstress_xx.tripcolor(x, y, c, cmap=cmap, shading='gouraud')
                p = axstress_xx.plot(x, y, 'k-', linewidth=0.5)
                plt.show()
        if Analysis_type == 'Strain XX':
            stress_xxNormalized = (strain_xx - strain_xx.min()) / (strain_xx.max() - strain_xx.min())
            plt.close('all')
            fig_1 = plt.figure(1)
            plt.axis('off')
            plt.show()
            plt.title('Strain XX')
            axstress_xx = Subplot(fig_1, 111)
            fig_1.add_subplot(axstress_xx)
            axstress_xx.axis['left'].set_visible(True)
            axstress_xx.axis['bottom'].set_visible(True)


            for i in range(np.size(EL, 0)):
                x = X[:, i]
                y = Y[:, i]
                c = stress_xxNormalized[:, i]
                cmap = truncate_colormap(plt.get_cmap('jet'), c.min(), c.max())
                t = axstress_xx.tripcolor(x, y, c, cmap=cmap, shading='gouraud')
                p = axstress_xx.plot(x, y, 'k-', linewidth=0.5)
                plt.show()
        if Analysis_type == 'Strain XY':
            stress_xxNormalized = (strain_xy - strain_xy.min()) / (strain_xy.max() - strain_xy.min())
            plt.close('all')
            fig_1 = plt.figure(1)
            plt.axis('off')
            plt.show()
            plt.title('Strain XY')
            axstress_xx = Subplot(fig_1, 111)
            fig_1.add_subplot(axstress_xx)
            axstress_xx.axis['left'].set_visible(True)
            axstress_xx.axis['bottom'].set_visible(True)


            for i in range(np.size(EL, 0)):
                x = X[:, i]
                y = Y[:, i]
                c = stress_xxNormalized[:, i]
                cmap = truncate_colormap(plt.get_cmap('jet'), c.min(), c.max())
                t = axstress_xx.tripcolor(x, y, c, cmap=cmap, shading='gouraud')
                p = axstress_xx.plot(x, y, 'k-', linewidth=0.5)
                plt.show()
        if Analysis_type == 'Strain YX':
            stress_xxNormalized = (strain_yx - strain_yx.min()) / (strain_yx.max() - strain_yx.min())
            plt.close('all')
            fig_1 = plt.figure(1)
            plt.axis('off')
            plt.show()
            plt.title('Strain YX')
            axstress_xx = Subplot(fig_1, 111)
            fig_1.add_subplot(axstress_xx)
            axstress_xx.axis['left'].set_visible(True)
            axstress_xx.axis['bottom'].set_visible(True)


            for i in range(np.size(EL, 0)):
                x = X[:, i]
                y = Y[:, i]
                c = stress_xxNormalized[:, i]
                cmap = truncate_colormap(plt.get_cmap('jet'), c.min(), c.max())
                t = axstress_xx.tripcolor(x, y, c, cmap=cmap, shading='gouraud')
                p = axstress_xx.plot(x, y, 'k-', linewidth=0.5)
                plt.show()
        if Analysis_type == 'Strain YY':
            stress_xxNormalized = (strain_yy - strain_yy.min()) / (strain_yy.max() - strain_yy.min())
            plt.close('all')
            fig_1 = plt.figure(1)
            plt.axis('off')
            plt.show()
            plt.title('Strain YY')
            axstress_xx = Subplot(fig_1, 111)
            fig_1.add_subplot(axstress_xx)
            axstress_xx.axis['left'].set_visible(True)
            axstress_xx.axis['bottom'].set_visible(True)


            for i in range(np.size(EL, 0)):
                x = X[:, i]
                y = Y[:, i]
                c = stress_xxNormalized[:, i]
                cmap = truncate_colormap(plt.get_cmap('jet'), c.min(), c.max())
                t = axstress_xx.tripcolor(x, y, c, cmap=cmap, shading='gouraud')
                p = axstress_xx.plot(x, y, 'k-', linewidth=0.5)
                plt.show()
        if Analysis_type == 'Disp X':
            stress_xxNormalized = (disp_x - disp_x.min()) / (disp_x.max() - disp_x.min())
            plt.close('all')
            fig_1 = plt.figure(1)
            plt.axis('off')
            plt.show()
            plt.title('Disp X')
            axstress_xx = Subplot(fig_1, 111)
            fig_1.add_subplot(axstress_xx)
            axstress_xx.axis['left'].set_visible(True)
            axstress_xx.axis['bottom'].set_visible(True)


            for i in range(np.size(EL, 0)):
                x = X[:, i]
                y = Y[:, i]
                c = stress_xxNormalized[:, i]
                cmap = truncate_colormap(plt.get_cmap('jet'), c.min(), c.max())
                t = axstress_xx.tripcolor(x, y, c, cmap=cmap, shading='gouraud')
                p = axstress_xx.plot(x, y, 'k-', linewidth=0.5)
                plt.show()
        if Analysis_type == 'Disp Y':
            stress_xxNormalized = (disp_x - disp_x.min()) / (disp_x.max() - disp_x.min())
            plt.close('all')
            fig_1 = plt.figure(1)
            plt.axis('off')
            plt.show()
            plt.title('Disp Y')
            axstress_xx = Subplot(fig_1, 111)
            fig_1.add_subplot(axstress_xx)
            axstress_xx.axis['left'].set_visible(True)
            axstress_xx.axis['bottom'].set_visible(True)


            for i in range(np.size(EL, 0)):
                x = X[:, i]
                y = Y[:, i]
                c = stress_xxNormalized[:, i]
                cmap = truncate_colormap(plt.get_cmap('jet'), c.min(), c.max())
                t = axstress_xx.tripcolor(x, y, c, cmap=cmap, shading='gouraud')
                p = axstress_xx.plot(x, y, 'k-', linewidth=0.5)
                plt.show()


mymodel = fem(d1=1,d2=1,p=6,m=6,R=0.2,element_type='Rectangular',defV=0.1,BC_flag='extension',E=8/3,nu=1/3,GPE=4,scale=1,analysis_type='Stress XX')
mymodel.configure_traits()


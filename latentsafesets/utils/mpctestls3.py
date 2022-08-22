#import scipy#HW4P1.10 in the MPC course
import numpy as np
from scipy import linalg as la
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from scipy.optimize import BFGS
import matplotlib.pyplot as plt

def mympc(A,B,Q,R,P,N,umin,umax,xmin,xmax,x,xdestin,AXE=np.zeros((1,2)),BXE=1e-6*np.ones(1)):
    x0=x-xdestin#put it inside!
    SX=np.eye(A.shape[0])
    for i in range(N):
        AIP1=np.linalg.matrix_power(A,i+1)#AIP1 means A i plus 1
        SX=np.concatenate((SX,AIP1),axis=0)#calculate the SX matrix in slides 5/6
    SU=np.zeros((B.shape[0]*N,B.shape[1]*N))#B#SUIFULL=[zeros(size(B,1),size(B,2)*N);SUI]
    for i in range(N):
        AIB=np.matmul(np.linalg.matrix_power(A,i),B)
        SUI=np.kron(np.diag(np.ones(N-i),-i),AIB)
        SU=SU+SUI#calculate the SU matrix in slides 5/6%BH = blkdiag(kron(eye(N),Q), P, kron(eye(N),R))
    SU=np.concatenate((np.zeros((B.shape[0],B.shape[1]*N)),SU),axis=0)
    QB=la.block_diag(np.kron(np.eye(N),Q), P)#QB for Q block_diagonal
    RB=la.block_diag(np.kron(np.eye(N),R))#RB for R block_diagonal
    SUT=np.transpose(SU)
    H=np.matmul(np.matmul(SUT,QB),SU)+RB#H is in slides5/6
    f=1*np.matmul(np.matmul(np.matmul(SUT,QB),SX),x0)#2.*SU'*QB*SX*x0#f is in slides5/6#page 28 of slides 6!
    AU=np.array([[1,0],[0,1],[-1,0],[0,-1]])#np.array([[1],[-1]])
    AUD=np.kron(np.eye(N),AU)#D for diagonalization%calculate the AU matrix in slides 6
    AX=np.array([[1,0],[0,1],[-1,0],[0,-1]])#4*2
    AX=np.concatenate((AX,AXE),axis=0)
    AXB=np.matmul(AX,B)
    AXD=np.zeros((AXB.shape[0]*N,AXB.shape[1]*N))#calculate the Ax matrix in slides 6
    for i in range(N):
        AI = np.linalg.matrix_power(A, i)
        AXAB=np.matmul(np.matmul(AX,AI),B)
        AXDI=np.kron(np.diag(np.ones(N-i),-i),AXAB)
        AXD=AXD+AXDI#the last AF is AX
    AXD=np.concatenate((np.zeros((AXB.shape[0]*1,AXB.shape[1]*N)),AXD),axis=0)
    G0=np.concatenate((AUD,AXD),axis=0)#THE G0 matrix in slides 6
    E01=np.zeros((AU.shape[0]*N,A.shape[1]))#E01 for E0 part 1
    AXDREAL=np.kron(np.eye(N+1),AX)#
    E02=-np.matmul(AXDREAL,SX)#E02 for E0 part 2
    E0=np.concatenate((E01,E02),axis=0)#THE E0 matrix in slides 6
    BU=np.array([umax[0],umax[1],-umin[0],-umin[1]])#([umax,-umin])#THE BU matrix in slides 6
    BX=np.array([xmax[0],xmax[1],-xmin[0],-xmin[1]])#THE BX matrix in slides 6
    BX=np.concatenate((BX,BXE),axis=0)
    W0U=np.kron(np.ones((N)),BU)#The first part of W0 matrix
    W0X=np.kron(np.ones((N+1)),BX)#the 2nd part of W0 matrix
    W0=np.concatenate((W0U,W0X),axis=0)#now it is just a row#mind the dim!#THE W0 matrix in slides 6
    #W0=W0.reshape((W0.shape[0],1))#
    W0E0x0=W0+np.matmul(E0,x0).reshape((W0.shape[0],))#it is W0+E0x0 term in slides 6 page 28
    lb=-np.inf*np.ones_like(W0E0x0)
    linear_constraint = LinearConstraint(G0, lb, W0E0x0)
    objective= lambda u: 0.5*np.matmul(np.matmul(np.transpose(u),H),u)+np.matmul(np.transpose(f),u)
    #(x[0] - 1) ** 2 + (x[1] - 2.5) ** 2
    '''
    def objective(u):
        ut=np.transpose(u)
        ft=np.transpose(f)
        return 0.5*np.matmul(np.matmul(ut,H),u)+np.matmul(ft,u)
    '''
    #u = quadprog(H,f,G0,W0E0x0)#use quadprog to find the sequence of u
    u0=np.ones(N*B.shape[1])##np.ones((N,1))#just an initial value
    res = minimize(objective, u0,constraints=linear_constraint)
    resx=res.x.reshape((N,B.shape[1]))
    return resx#the u is like the x in the quadprog function in matlab documentation


#in the code, you can follow the comments to modify the relevant parameters%to achieve different aims in different part of the problem
A=np.array([[1,0],[0,1]])#[EV1,EV2]=linalg.eigvals(A)#ACTUALLY NOT USED
B=np.array([[1,0],[0,1]])#([[0.2173],[0.0573]])#what should the dim of B be?
QV1=1#100#;#500;#change this in part 8#QV for Q value
QV2=1#100#500;#;#change this in part 8#
Q = np.diag([QV1,QV2]) #QV.*eye(2);%100*eye(2);#
RV1=1;RV2=1
R = np.diag([RV1,RV2]) #R
N = 5#1;#2;#25#10#;#prediction horizon, change this in part 6,7,8#
T=1#0.1#sample time
N2=25#10#5#2#it is the plotting horizon, 5s=50steps
#[PRIC,L,G]
PRIC= la.solve_discrete_are(A,B,Q,R)#P_ric
#PLYAP = la.solve_discrete_lyapunov(A,Q)#P_lyap
P=Q#used in part 7
bignumber=100000000#as if it is infinity
xmin=[-1,-1]#[-bignumber,-1]#0]#0#-100000#-inf
xmax=[+bignumber,+bignumber]#10000000000#inf
umin=[-3,-3]#-5#-50#-100#
umax=[3,3]#5#50#100#
x0=np.array([[6],[60]])#[0;10];%[0.10;0.10];%[0;3];%[0;2];%x0 is the initial condition, will be changed in part 8 and 10
XK=x0#
XKA=XK#[XK]#XKA is XK array
UKA=np.zeros((B.shape[1],1))#UKA is UK array
#Ncross=np.array([0,0])#cross cost term between x and u in dlqr
#[K,S,e] = dlqr(A,B,Q,R,Ncross)#calculate gain for unconstraint LQR in part 8
xdestin=np.array([[2],[20]])#
for i in range(int(N2*(1/T))):#u0=mympc(A,B,Q,R,PRIC,N,umin,umax,xmin,xmax,XK)%use P_ric
    #u0=mympc(A,B,Q,R,PLYAP,N,umin,umax,xmin,xmax,XK)#use P_lyap
    u0 = mympc(A, B, Q, R, PRIC, N, umin, umax, xmin, xmax, XK,xdestin)  # use P_lyap
    #print('u0',u0)
    #u0=mympc(A,B,Q,R,P,N,umin,umax,xmin,xmax,XK)%use P
    UK=u0[0]#FKI*XK%pick the first one from u0 to implement
    UK=UK.reshape((UK.shape[0], 1))
    #print('UK.shape', UK.shape)#finally(1,1)!
    #UK=-K*XK;used only in part 8 for unconstraint LQR
    UKA=np.concatenate((UKA,UK),axis=1)#UKA.append(UK)#np.concatenate(2,UKA,UK)#store the values
    XKP1=np.matmul(A,XK)+np.matmul(B,UK)#B*UK#+0.0*sqrt(10)*randn(2,1)#waiting to add the disturbance term!
    XKA=np.concatenate((XKA,XKP1),axis=1)#XKA.append(XKP1)#XKA=cat(2,XKA,XKP1)#store the values
    XK=XKP1#XKP1 means x_k+1
#
UKA=UKA[:,1:]
lw=2#lw means line width
sz=16#sz means size
#figure(1)
sa=1e-6#sa means small amount
t=np.arange(0,N2+sa,T)#
plt.plot(t,np.transpose(XKA),linewidth=lw,label=['x1','x2'])#,'LineWidth',lw)#
tu=np.arange(0,N2-sa,T)#=0:T:N2-T#1:N2;%0:N2-1;%1:N2;%
plt.plot(tu,np.transpose(UKA),linewidth=lw,label=['u1','u2'])#,'LineWidth',lw)#
plt.title('states and control Pric',fontsize=sz)#Pric','FontSize',sz);%
plt.xlabel('time/s',fontsize=sz)#ylabel('output','FontSize',sz)#
plt.legend(fontsize=sz)#legend('x1','Ts='+string(T),'Fontsize',sz)

info='LS3'#'HW4P1'
info1='J6'#['G'];%['H0d1'];%['I2geqx2geq0'];%['H'];%
info2='ric'#'lyap'#['ric'];%['PQ'];%['Hlyap'];%['newfric'];%['Gric'];%int2str(256)
n1=info+info1+info2+'QV1'+str(QV1)+'QV2'+str(QV2)+'N'+str(N)+'T'+str(T)+'Nt'+str(N2)#[info,info1,info2,'QV1',int2str(QV1),'QV2',int2str(QV2),'N',int2str(N)]#[info,string(QV)]#
plt.savefig('/home/jianning/Documents/ESE619THINGS/'+n1+'.png')#plt.saveas(figure(1),n1,'fig')#
plt.show()

plt.plot(XKA[0,:],XKA[1,:],linewidth=lw,label='traj')#,'LineWidth',lw)#
plt.title('trajectory PRIC',fontsize=sz)#Pric','FontSize',sz);%
plt.xlabel('x1/m',fontsize=sz)#
plt.ylabel('x2/m',fontsize=sz)#
plt.legend(fontsize=sz)#legend('x1','Ts='+string(T),'Fontsize',sz)
n2=info+info1+info2+'QV1'+str(QV1)+'QV2'+str(QV2)+'N'+str(N)+'T'+str(T)+'traj'#[info,info1,info2,'QV1',int2str(QV1),'QV2',int2str(QV2),'N',int2str(N)]#[info,string(QV)]#
plt.savefig('/home/jianning/Documents/ESE619THINGS/'+n2+'.png')#plt.saveas(figure(1),n1,'fig')#
plt.show()
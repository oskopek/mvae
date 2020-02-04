#!/usr/bin/env wolframscript
(* ::Package:: *)

p[x_,y_] := x.y;
n[x_]:=Sqrt[p[x,x]];
kplus[x_,y_,K_]:=((1-2*K*p[x,y]-K*n[y]^2)*x+(1+K*n[x]^2)*y)/(1-2*K*p[x,y]+K^2*n[x]^2*n[y]^2);
f[x_,y_,K_]:=1-(2*K*n[x-y]^2)/((1+K*n[x]^2)*(1+K*n[y]^2));
do[x_,y_,K_]:=1/Sqrt[K] ArcCos[f[x,y,K]];
dg[x_,y_,K_]:=2/Sqrt[K] ArcTan[Sqrt[K]*n[kplus[-x,y,K]]];
cf[x_,K_]:=2/(1+K*n[x]^2);
gyr[x_,y_,v_,K_]:=kplus[-kplus[x,y,K],kplus[x,kplus[y,v,K],K],K];
pt[x_,y_,v_,K_]:=cf[x,K]/cf[y,K]*gyr[y,-x,v,K];

rho[\[Xi]_, x_,R_]:=(R*x)/(R+\[Xi]);
rhoinvXi[y_,R_]:=R*(-n[y]^2+R^2)/(n[y]^2+R^2);
rhoinvX[y_,R_]:=2*(R^2*y)/(n[y]^2+R^2);


mu0d={0,0};
{rhoinvXi[mu0d, K], rhoinvX[mu0d, K]}
rho[rhoinvXi[mu0d, K], rhoinvX[mu0d, K], K]


mu0s[K_]:={K, 0, 0};
r=rho[K,{0,0}, K]
{rhoinvXi[r,K], rhoinvX[r, K]}


mu0d={1,2};
K=2;
N[{rhoinvXi[mu0d, K], rhoinvX[mu0d, K]}]
N[rho[rhoinvXi[mu0d, K], rhoinvX[mu0d, K], K]]


K=2;
xxi=Sqrt[2];
xx=Sqrt[2];
Norm[{xxi,xx,0}]
N[{xxi,xx,0}]
r=rho[xxi,{xx, 0}, K];
N[r]
N[{rhoinvXi[r, K], rhoinvX[r, K]}]




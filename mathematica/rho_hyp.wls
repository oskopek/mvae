#!/usr/bin/env wolframscript
(* ::Package:: *)

p[xs_,x_,ys_,y_] := x.y-xs*ys;
n[xs_,x_]:=Sqrt[p[xs,x,xs,x]];

rho[\[Xi]_, x_,R_]:=(R*x)/(R+\[Xi]);
rhoinvXi[y_,R_]:=R*(Norm[y]^2+R^2)/(R^2-Norm[y]^2);
rhoinvX[y_,R_]:=(2*R^2*y)/(R^2-Norm[y]^2);


Solve[(R*x)/(R+\[Xi])==y,x]
(* We know: -\[Xi]^2+x^2=-R^2 \[Equal]> x^2 = \[Xi]^2-R^2 *)
Solve[((R+\[Xi])*y/R)^2==\[Xi]^2-R^2,\[Xi]]
Solve[x==(y (R+(R (R^2+y^2))/(R^2-y^2)))/R,x]

Solve[(R*x)/(R+\[Xi])==y && \[Xi]^2+x^2==R^2,{x,\[Xi]}]
((R^3-R y^2)/(R^2+y^2))^2+((2 R^2 y)/(R^2+y^2))^2 // FullSimplify
Solve[(R*x)/(R+\[Xi])==y && -\[Xi]^2+x^2==-R^2,{x,\[Xi]}]
-((R (R^2+y^2))/(R^2-y^2))^2+((2 R^2 y)/(R^2-y^2))^2 // FullSimplify





mu0d={0,0};
{rhoinvXi[mu0d, K], rhoinvX[mu0d, K]}
rho[rhoinvXi[mu0d, K], rhoinvX[mu0d, K], K]


r=rho[K,{0,0}, K]
{rhoinvXi[r,K], rhoinvX[r, K]}


mu0d={1,2};
K=2;
N[{rhoinvXi[mu0d, K], rhoinvX[mu0d, K]}]
N[rho[rhoinvXi[mu0d, K], rhoinvX[mu0d, K], K]]


R=2;
xxi=Sqrt[5];
xx=Sqrt[3];
n[xxi*R,{1,xx}*R]^2
N[{xxi,1,xx}*R]
r=rho[xxi*R,{1,xx}*R, R];
N[r]
N[{rhoinvXi[r, R], rhoinvX[r, R]}]
N[n[rhoinvXi[r, R], rhoinvX[r, R]]^2]




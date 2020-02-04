#!/usr/bin/env wolframscript
(* ::Package:: *)

p[x_,y_] := x*y;
n[x_]:=Sqrt[p[x,x]];
kplus[x_,y_,K_]:=((1-2*K*p[x,y]-K*n[y]^2)*x+(1+K*n[x]^2)*y)/(1-2*K*p[x,y]+K^2*n[x]^2*n[y]^2);
f[x_,y_,K_]:=1-(2*K*n[x-y]^2)/((1+K*n[x]^2)*(1+K*n[y]^2));
do[x_,y_,K_]:=1/Sqrt[K] ArcCos[f[x,y,K]];
dg[x_,y_,K_]:=2/Sqrt[K] ArcTan[Sqrt[K]*n[kplus[-x,y,K]]];
cf[x_,K_]:=2/(1+K*n[x]^2);
gyr[x_,y_,v_,K_]:=kplus[-kplus[x,y,K],kplus[x,kplus[y,v,K],K],K];
pt[x_,y_,v_,K_]:=cf[x,K]/cf[y,K]*gyr[y,-x,v,K];


gyr[x,-x,v,k]


cf[x,k]/cf[y,k] * kplus[x,y,k] // Simplify


kplus[x,y,k] // Simplify
-kplus[x,y,k] // Simplify


pt[x,y,v,K]


FullSimplify[4*K*x*x <= (1+K*y*y)^2, x>y]


(k*(x-y)^2)/((1+k*x^2)*(1+k*y^2)-k*(x-y)^2)//FullSimplify


Sqrt[k]*kplus[-x,y,k] // FullSimplify





kplus[-x, y, K]-kplus[-y,x,K] // Simplify


left=-((x-y+K y x.x-2 K y x.y+K x y.y)/(1+2 K x.y+K^2 x.x y.y)) // Simplify
right=+((-x+y+K y x.x-2 K x x.y+K x y.y)/(1+2 K x.y+K^2 x.x y.y)) //Simplify

sqleft = kplus[-x, y, K].kplus[-x, y, K] //Simplify;
sqleft =(-x+y+K y x.x-2 K x x.y+K x y.y)/(1+2 K x.y+K^2 x.x y.y)//Simplify

sqright = kplus[-y,x,K].kplus[-y,x,K] // Simplify;




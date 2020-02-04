#!/usr/bin/env wolframscript
(* ::Package:: *)

Limit[ArcCos[1-(2*K*Norm[x-y]^2)/((1+K*Norm[x]^2)(1+K*Norm[y]^2))]/Sqrt[K],K->0,Direction->"FromAbove"]
Limit[ArcCosh[1-(2*K*Norm[x-y]^2)/((1+K*Norm[x]^2)(1+K*Norm[y]^2))]/Sqrt[-K],K->0,Direction->"FromBelow"]


kplus[x_,y_,K_]:=((1-2K*Dot[x,y]-K*Norm[y]^2)*x+(1+K*Norm[x]^2)*y)/(1-2K*Dot[x,y]+K^2*Norm[x]^2*Norm[y]^2);

kplus[{x1,x2},{y1,y2},0]

dpsgyr[x_,y_,K_]:=2/Sqrt[K] ArcTan[Sqrt[K]*Norm[kplus[-x, y,K]]];
dpscos[x_,y_,K_]:=1/Sqrt[K] ArcCos[1-(2*K*Norm[x-y]^2)/((1+K*Norm[x]^2)(1+K*Norm[y]^2))];

dpgyr[x_,y_,K_]:=2/Sqrt[-K] ArcTanh[Sqrt[-K]*Norm[kplus[-x, y,K]]];
dpcos[x_,y_,K_]:=1/Sqrt[-K] ArcCosh[1-(2*K*Norm[x-y]^2)/((1+K*Norm[x]^2)(1+K*Norm[y]^2))];

a={1,2,3};
b={4,5,6};

N[dpsgyr[a,b,1]]
N[dpscos[a,b,1]]

N[dpgyr[a,b,-1]]
N[dpcos[a,b,-1]]

N[dpsgyr[a,b,0.5]]
N[dpscos[a,b,0.5]]

N[dpgyr[a,b,-0.5]]
N[dpcos[a,b,-0.5]]


Simplify[kplus[x,y,K]]
FullSimplify[D[kplus[x,y,K],K]]

dpscos[x,y,1] // TrigToExp // Simplify
dpsgyr[x,y,1] // TrigToExp // Simplify


dp[x_, y_, K_]:=(1/Sqrt[-K])*ArcCosh[1-(2*K *Norm[x-y]^2)/((1+K*Norm[x]^2)(1+K*Norm[y]^2))]
Limit[dp[x,y,K],K->0, Direction->FromBelow]
(*WolframAlpha["Limit[(1/Sqrt[-K])*ArcCosh[1-2*K *Norm[x-y]^2/((1+K*Norm[x]^2)*(1+K*Norm[y]^2))],K\[Rule]0]", AppearanceElements->{"Pods", "Assumptions"},PodStates->{"More"}]*)


a={2,3};
b={4,5};

(* Long version: Expand[Simplify[D[kplus[x,y,K],K]]] *)
Simplify[D[kplus[x,y,K],K]]
Simplify[D[kplus[a,b,K],K]]
derivk[x_,y_,K_]:=(-x Norm[y]^2-K^2 y Norm[x]^4 Norm[y]^2+2 x.y (y+K^2 x Norm[x]^2 Norm[y]^2)+Norm[x]^2 (y-2 K (x+y) Norm[y]^2+K^2 x Norm[y]^4))/(1-2 K x.y+K^2 Norm[x]^2 Norm[y]^2)^2;
Simplify[derivk[a,b,K]]


kplus2[x_,y_,K_]:=((1-2K*x*y-K*y^2)*x+(1+K*x^2)*y)/(1-2K*x*y+K^2*x^2*y^2);
Simplify[kplus2[x,y,K]]
Simplify[D[kplus2[x,y,K],K]]
Integrate[Dot[x,y]/(K*Dot[x,y]-1)^2 (x+y),K]


a={2,3};
b={4,5};
Simplify[D[kplus[a,b,K],K]]
Simplify[D[kplus2[a,b,K],K]]
Simplify[Dot[a,b]/(K*Dot[a,b]-1)^2 (a+b)]


Expand[(-x y^2-K^2 y x^4 y^2+2 x.y (y+K^2 x x^2 y^2)+x^2 (y-2 K (x+y) y^2+K^2 x y^4))/(1-2 K x*y+K^2 x^2 y^2)^2]


Expand[derivk[x,y,K]]

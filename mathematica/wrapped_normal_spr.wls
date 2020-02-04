#!/usr/bin/env wolframscript
(* ::Package:: *)

p[x_,y_] := x.y;
n[x_]:=Sqrt[p[x,x]];
kplus[x_,y_,K_]:=((1+2*K*p[x,y]-K*n[y]^2)*x+(1+K*n[x]^2)*y)/(1-2*K*p[x,y]+K^2*n[x]^2*n[y]^2);
do[x_,y_,K_]:=1/Sqrt[K] ArcCos[1-(2*K*n[x-y]^2)/((1+K*n[x]^2)*(1+K*n[y]^2))];
dg[x_,y_,K_]:=2/Sqrt[K] ArcTan[Sqrt[K]*n[kplus[-x,y,K]]];
cf[x_,K_]:=2/(1+K*n[x]^2);
expmapr[x_,v_,K_]:=kplus[x,Tan[Sqrt[K]*(cf[x,K]*n[v])/2]*v/(Sqrt[K]*n[v]),K];
gyr[x_,y_,v_,K_]:=kplus[-kplus[x,y,K], kplus[x,kplus[y,v,K],K],K];
ptr[x_,y_,v_,K_] := cf[x,K]/cf[y,K]*gyr[y,-x,v,K];


(* PT with variable radius *)
ptres=Simplify[ptr[x,y,v+\[Epsilon]*\[Xi], K], v.x==0]
d=Simplify[D[ptres,\[Epsilon]]]; (*... this equals pt[{x1,x2},{y1,y2},{e1,e2}]..*)
Simplify[d /. \[Epsilon] -> 0] (*... this equals pt[{x1,x2},{y1,y2},{e1,e2}]..*)
(* hence dpt_{x\[Rule]y}(e) = pt_{x\[Rule]y}(e) for every e in the orthogonal basis *)
(* hence dpt(x\[Rule]y)(v) = pt(x\[Rule]y)(v) *) 
(* given ||pt(x\[Rule]y)(e)||_L = 1 because ||e||_L=1, hence the det is 1, because *)


(* exp, mean direction R\[NotEqual]1*)
e=Simplify[expmapr[x,u+\[Epsilon]*(u/n[u]),K]]
d=Simplify[D[e,\[Epsilon]]]
d=Simplify[d /. \[Epsilon] -> 0]
(* r = ||u||_L, radius = R *)


(* exp, other directions R\[NotEqual]1*)
e=Simplify[expmapr[x,u+\[Epsilon]*\[Xi],K]];
(* equivalent to e above, since {e1,e2} is a basis vector *)
d=Simplify[D[e,\[Epsilon]]]
d=Simplify[d /. \[Epsilon] -> 0]
d=Refine[Refine[d, Dot[u,\[Xi]]==0], Dot[\[Xi],u]== 0] // Simplify

(* r = ||u||_L *)




#!/usr/bin/env wolframscript
(* ::Package:: *)

sp[x_,y_] := Dot[x,y];
sn[x_]:=Sqrt[sp[x,x]];
expmapr[x_,u_,R_]:=Cos[sn[u]/R]x + Sin[sn[u]/R] (R*u)/sn[u];
ptr[x_,y_,v_,R_] := v - sp[y,v]/(sp[x,y]+R^2) (x+y);


(* PT with variable radius *)
p=Simplify[ptr[x,y,v+\[Epsilon]*\[Xi], R]];
Simplify[D[p,\[Epsilon]]] (*... this equals pt[{x1,x2},{y1,y2},{e1,e2}]..*)
Simplify[D[p,\[Epsilon]] /. \[Epsilon] -> 0] (*... this equals pt[{x1,x2},{y1,y2},{e1,e2}]..*)
(* hence dpt_{x\[Rule]y}(e) = pt_{x\[Rule]y}(e) for every e in the orthogonal basis *)
(* hence dpt(x\[Rule]y)(v) = pt(x\[Rule]y)(v) *)
(* given ||pt(x\[Rule]y)(e)||_L = 1 because ||e||_L=1, hence the det is 1, because *)


(* exp, mean direction R\[NotEqual]1*)
e=Simplify[expmapr[x,u+\[Epsilon]*(u/sn[u]),R]] /. (u+(u \[Epsilon])/Sqrt[u.u]).(u+(u \[Epsilon])/Sqrt[u.u])->(u.u+2*\[Epsilon]*Sqrt[u.u]+\[Epsilon]^2) //Simplify;
d=Simplify[D[e,\[Epsilon]]] /. (u+(u \[Epsilon])/Sqrt[u.u]).(u/Sqrt[u.u])-> (Sqrt[u.u]+\[Epsilon]*Sqrt[u.u]) /. (u/Sqrt[u.u]).(u+(u \[Epsilon])/Sqrt[u.u])->(Sqrt[u.u]+\[Epsilon]*Sqrt[u.u])/.(u+(u \[Epsilon])/Sqrt[u.u]).(u+(u \[Epsilon])/Sqrt[u.u])->(u.u+2*\[Epsilon]*Sqrt[u.u]+\[Epsilon]^2) // Simplify;
(-x \[Epsilon] Sqrt[u.u] Sin[Sqrt[(\[Epsilon]+Sqrt[u.u])^2]/R]-x u.u Sin[Sqrt[(\[Epsilon]+Sqrt[u.u])^2]/R])/(R Sqrt[(\[Epsilon]+Sqrt[u.u])^2] Sqrt[u.u]) // Simplify
d=Simplify[d /. \[Epsilon] -> 0] /. (u.(u/Sqrt[u.u])+(u/Sqrt[u.u]).u)-> 2Sqrt[u.u] // Simplify
dotExpand[exp_, dom : (Reals | Complexes) : Reals] :=
  Module[{res},
    res = ExpandAll[exp] /. Dot[x_, y_] :> Distribute[Dot[x, y]];
    res = res //. {
      Dot[ (a_ /; Simplify[a \[Element] dom])   b_, c_] :> a Dot[b, c],
      Dot[ a_,  (b_ /; Simplify[b \[Element] dom]) c_] :> Simplify[Conjugate[b]]  Dot[a, c],
      If[dom === Reals, Dot[x_, y_] :> Sort[Dot[x, y]], Unevaluated[Sequence[]]]};
    Collect[res, Dot[_, _]]  ]

d=dotExpand[d.d]; (* (u Cos[Sqrt[u.u]/R]).(u Cos[Sqrt[u.u]/R])/u.u-(2 (u Cos[Sqrt[u.u]/R]).((x Sin[Sqrt[u.u]/R])/R))/Sqrt[u.u]+((x Sin[Sqrt[u.u]/R])/R).((x Sin[Sqrt[u.u]/R])/R) *)
d2=Cos[Sqrt[u.u]/R]^2-(2*x.u *Cos[Sqrt[u.u]/R].Sin[Sqrt[u.u]/R])/(R*Sqrt[u.u])+Sin[Sqrt[u.u]/R]^2 // Simplify (* hence the norm is 1 *)
(* r = ||u||_L, radius = R *)


(* ::InheritFromParent:: *)
(*x Cos[Sqrt[(\[Epsilon]+Sqrt[u.u])^2]/R]+( R (u+(u \[Epsilon])/Sqrt[u.u]) Sin[Sqrt[(\[Epsilon]+Sqrt[u.u])^2]/R])/Sqrt[(\[Epsilon]+Sqrt[u.u])^2]*)


(* exp, other directions R\[NotEqual]1*)
e=Simplify[expmapr[x,u+\[Epsilon]*\[Xi],R]] /. Sqrt[(u+\[Epsilon] \[Xi]).(u+\[Epsilon] \[Xi])]->Sqrt[(u.u+\[Epsilon]^2)] //Simplify;
(* equivalent to e above, since {e1,e2} is a basis vector *)
d=Simplify[D[e,\[Epsilon]]] /.  (\[Xi].(u+\[Epsilon] \[Xi])+(u+\[Epsilon] \[Xi]).\[Xi])-> 2\[Epsilon] // Simplify //Expand;
-((R u \[Epsilon] Sin[Sqrt[\[Epsilon]^2+u.u]/R])/(\[Epsilon]^2+u.u)^(3/2))-(R \[Epsilon]^2 \[Xi] Sin[Sqrt[\[Epsilon]^2+u.u]/R])/(\[Epsilon]^2+u.u)^(3/2)-(x \[Epsilon] Sin[Sqrt[\[Epsilon]^2+u.u]/R])/(R Sqrt[\[Epsilon]^2+u.u])+(R \[Xi] Sin[Sqrt[\[Epsilon]^2+u.u]/R])/Sqrt[\[Epsilon]^2+u.u]//Simplify;
d=Simplify[d /. \[Epsilon] -> 0] /. (u.(u/Sqrt[u.u])+(u/Sqrt[u.u]).u)-> 2Sqrt[u.u] /. u.\[Xi]->0 /. \[Xi].u->0 // Simplify

d=dotExpand[d.d]; (*(R \[Xi] Sin[Sqrt[u.u]/R]).(R \[Xi] Sin[Sqrt[u.u]/R])/u.u*)
d2=(R^2 Sin[Sqrt[u.u]/R]^2)/u.u (* hence the norm is RSin[Sqrt[u.u]/R]/Sqrt[u.u]*)
(* r = ||u||_L *)







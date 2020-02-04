#!/usr/bin/env wolframscript
(* ::Package:: *)

(* H^n_1 *)
lp[x_,y_] := Dot[x,y]-2*x[[1]]*y[[1]];
ln[x_]:=Sqrt[lp[x,x]];
expmap[x_,u_]:=Cosh[ln[u]]x + Sinh[ln[u]] u/ln[u];
expmapr[x_,u_,r_]:=Cosh[ln[u]/r]x + Sinh[ln[u]/r] (r*u)/ln[u];
pt[x_,y_,v_] := v + lp[y,v]/(1-lp[x,y]) (x+y);
ptr[x_,y_,v_,r_] := v + lp[y,v]/(r^2-lp[x,y]) (x+y);
r=ln[{u1,u2}];


(* PT *)
p=Simplify[pt[{x1,x2},{y1,y2},{v1+\[Epsilon]*e1,v2+\[Epsilon]*e2}]]
Simplify[D[p,\[Epsilon]]] (*... this equals pt[{x1,x2},{y1,y2},{e1,e2}]..*)
Simplify[D[p,\[Epsilon]] /. \[Epsilon] -> 0] (*... this equals pt[{x1,x2},{y1,y2},{e1,e2}]..*)
(* hence dpt_{x\[Rule]y}(e) = pt_{x\[Rule]y}(e) for every e in the orthogonal basis *)
(* hence dpt(x\[Rule]y)(v) = pt(x\[Rule]y)(v) *)
(* given ||pt(x\[Rule]y)(e)||_L = 1 because ||e||_L=1, hence the det is 1, because *)
(* WTF?? The desired determinant is the product of the Lorentzian norms of the vectors of the set above. *)


(* PT with variable radius *)
p=Simplify[ptr[{x1,x2},{y1,y2},{v1+\[Epsilon]*e1,v2+\[Epsilon]*e2}, R]];
Simplify[D[p,\[Epsilon]]] (*... this equals pt[{x1,x2},{y1,y2},{e1,e2}]..*)
Simplify[D[p,\[Epsilon]] /. \[Epsilon] -> 0] (*... this equals pt[{x1,x2},{y1,y2},{e1,e2}]..*)
(* hence dpt_{x\[Rule]y}(e) = pt_{x\[Rule]y}(e) for every e in the orthogonal basis *)
(* hence dpt(x\[Rule]y)(v) = pt(x\[Rule]y)(v) *)
(* given ||pt(x\[Rule]y)(e)||_L = 1 because ||e||_L=1, hence the det is 1, because *)
(* WTF?? The desired determinant is the product of the Lorentzian norms of the vectors of the set above. *)


(* exp, mean direction *)
e=Simplify[expmap[{x1,x2},{u1+\[Epsilon]*(u1/r), u2+\[Epsilon]*(u2/ln[{u1,u2}])}]];
Simplify[D[e,\[Epsilon]]]
Simplify[D[e,\[Epsilon]] /. \[Epsilon] -> 0]

(* r = ||u||_L *)
(* Hence the column of the Jacobian is (\frac{\cosh(r)]}{r})u + Sinh(r)x, exactly like in the PHG. *)


(* exp, mean direction R\[NotEqual]1*)
e=Simplify[expmapr[{x1,x2},{u1+\[Epsilon]*(u1/ln[{u1,u2}]), u2+\[Epsilon]*(u2/ln[{u1,u2}])},R]]
Simplify[D[e,\[Epsilon]]]
Simplify[D[e,\[Epsilon]] /. \[Epsilon] -> 0]

(* r = ||u||_L, radius = R *)


(* exp, other directions *)
e=Simplify[expmap[{x1,x2},{u1+\[Epsilon]*e1, u2+\[Epsilon]*e2}]];
(* equivalent to e above, since {e1,e2} is a basis vector *)
e=Cosh[Sqrt[r^2 + \[Epsilon]^2]/R]*{x1, x2}+(R*Sinh[Sqrt[r^2 + \[Epsilon]^2]/R])/Sqrt[r^2 + \[Epsilon]^2] ({u1,u2}+\[Epsilon]{e1,e2});
d=Simplify[D[e,\[Epsilon]]]
Simplify[d /. \[Epsilon] -> 0]

(* r = ||u||_L *)
(* Hence the column of the Jacobian is (\frac{\cosh(r)]}{r})u + Sinh(r)x, exactly like in the PHG. *)


(* exp, other directions R\[NotEqual]1*)
e=Simplify[expmapr[{x1,x2},{u1+\[Epsilon]*e1, u2+\[Epsilon]*e2},R]];
(* equivalent to e above, since {e1,e2} is a basis vector *)
d=Simplify[D[e,\[Epsilon]]]
Simplify[d /. \[Epsilon] -> 0]

(* r = ||u||_L *)




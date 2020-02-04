#!/usr/bin/env wolframscript
(* ::Package:: *)

vp={2, 2, 2};
vph={2, 2, 1.3};
cd=ColorData["RedBlueTones"];
clr[K_]:=cd[K];

R=2;
r=1;
toruscurv[u_,v_,R_,r_]:=(2*Cos[v])/(r(R+r*Cos[v]))+0.5; (* not exactly correct, but similar *)
cft[u_,v_]:=clr[toruscurv[u,v,R,r]];
plot=ParametricPlot3D[{(R+ r*Cos[v]) Cos[u], (R + r*Cos[v]) Sin[u], r*Sin[v]}, {u, 0, 2 Pi}, {v, 0, 2 Pi}, Mesh -> Full, ColorFunction->Function[{x,y,z,u,v},cft[u,v]], ViewPoint->vp, Boxed->False, Axes->False, PlotPoints->50, ImageSize->Large, Background->None, ColorFunctionScaling->False]
Export["/home/oskopek/git/mvae-paper/figures/torus.pdf", plot, "PDF", "AllowRasterization"->False];

cfs[x_, y_, z_]:=clr[1];
plot=ParametricPlot3D[{Sin[\[Phi]] Cos[\[Theta]], Cos[\[Phi]] Cos[\[Theta]],  Sin[\[Theta]]}, {\[Theta], -2\[Pi], 2\[Pi]}, {\[Phi], -2\[Pi], 2\[Pi]}, Mesh -> Full, ColorFunction->Function[{x,y,z},cfs[x,y,z]], ViewPoint->vp, Boxed->False, Axes->False, PlotPoints->50, ImageSize->Large, Background->None, ColorFunctionScaling->False]
Export["/home/oskopek/git/mvae-paper/figures/sphere.pdf", plot, "PDF", "AllowRasterization"->False];
;
cfh[x_, y_, z_]:=clr[-1];
c=1;
a=1;
plot=ParametricPlot3D[{a*Sinh[u]*Cos[v], a*Sinh[u]*Sin[v], c*Cosh[u]}, {u, -Pi, Pi}, {v, 0, Pi}, Mesh -> Full, ColorFunction->Function[{x,y,z},cfh[x,y,z]], ViewPoint->vph,Boxed->False, Axes->False, PlotPoints->50, ImageSize->Large, Background->None, ColorFunctionScaling->False]
Export["/home/oskopek/git/mvae-paper/figures/hyperboloid.pdf", plot, "PDF", "AllowRasterization"->False];




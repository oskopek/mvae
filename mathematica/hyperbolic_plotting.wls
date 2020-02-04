#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Input:: *)
(*u = {0.5, 1};*)
(*u =u/Norm[u];*)
(*v ={2,0};*)
(*v =v/Norm[v];*)
(*uv=(u.v)*v;*)
(*vu=(u.v)*u;*)
(*alpha=1;*)
(*p1=ParametricPlot[{{Cos[t],Sin[t]}, alpha*{Cos[t],Sin[t]}},{t,0,2  Pi},PlotStyle->{Black, Dashed}];*)
(*p2=Graphics[{Red,Arrow[{{0,0},u}]}];*)
(*p3=Graphics[{Blue,Arrow[{{0,0},v}]}];*)
(*p4=Graphics[{Green,Arrow[{{0,0},uv}]}];*)
(*p5=Graphics[{Green,Arrow[{{0,0},vu}]}];*)
(*Show[p1,p2, p3, p4, p5]*)
(*angle=ArcCos[u.v]*)


(* ::Input:: *)
(*cSign[x_]:=Tanh[3*x];*)
(*lorentzProduct[x_,y_, K_] := x . y - x[[1]]*y[[1]] +cSign[K]*x[[1]]*y[[1]] ;*)
(*eps=0.001;*)
(*Kfn[K_]:=Tanh[3*K];*)
(*epsChange[K_]:=PlusMinus[Sech[3*K]^2];*)
(*rightCond[K_]:=Kfn[K]*(1/Abs[K]+eps)+epsChange[K];*)
(*space[x_, y_, K_] := Kfn[K]*x^2+y^2\[Element]rightCond[K];*)
(*space3D[x_, y_, z_,K_] := Kfn[K]*x^2+y^2+z^2\[Element]rightCond[K];*)
(*ContourPlot3D[{Sign[3*K]*x^2+y^2==Sign[3*K]*(1/Abs[K]+eps)+0.25*1/(K^2 + eps), Sign[3*K]*x^2+y^2==Sign[3*K]*(1/Abs[K]+eps)-0.25*1/(K^2 + eps)},{x,-5,5},{y,-5,5}, {K, -1, 1},PlotLegends->Automatic]*)


(* ::Input:: *)
(*K3D = 1/20;*)
(*Plot3D[coef*1/K^2,{coef, 0, 1}, {K, -1, 1},AxesLabel->Automatic]*)
(*Manipulate[ContourPlot3D[{Sign[K]*x^2+y^2+z^2==Sign[K]*(1/Abs[K]+eps), Sign[K]*x^2+y^2+z^2==Sign[K]*((1/Abs[K]+eps)-coef*1/K^2)},{x,-5,5},{y,-5,5}, {z, -5, 5},PlotLegends->Automatic, ContourStyle->{Opacity[0.5]},AxesLabel-> Automatic],{coef, 0.0, 1.0},{K, -1, 1}]*)


ContourPlot3D[Sign[3*K3D]*x^2+y^2+z^2==rightCond-eps3D,{x,-5,5},{y,-5,5}, {z, -5, 5}, PlotLegends->Automatic, ContourStyle->Opacity[0.8]]


(* 2D plots of H1, S1, E2. Look at coef 0.05 and K from 0.2 to -0.2. *)
eps=0.00000001;
Manipulate[ContourPlot[{Sign[K]*x^2+y^2==Sign[K]*(1/(Abs[K]+eps)), Sign[K]*x^2+y^2==Sign[K]*((1/(Abs[K]+eps))-coef/K^2)},{x,-5,5},{y,-5,5}, PlotLegends->Automatic, ContourStyle->{Opacity[0.5]}, AxesLabel-> {x,y}],{coef, 0.0, 0.1},{K, -0.1, 0.1}]
ContourPlot3D[{Sign[K]*x^2+y^2==Sign[K]*(1/(Abs[K]+eps)), Sign[K]*x^2+y^2==Sign[K]*((1/(Abs[K]+eps))-0.05/(K^2 + eps))},{x,-5,5},{y,-5,5}, {K, -0.1, 0.1}, PlotLegends->Automatic, AxesLabel-> {x,y,K}]


(* 2D plots of H1, S1, E2. Look at coef 0.05 and K from 0.2 to -0.2. With an interpolated Sign. *)
eps=0.00000001;
Manipulate[ContourPlot[{Tanh[10*K]*x^2+y^2==Tanh[10*K]*(1/Abs[K]+eps), Tanh[10*K]*x^2+y^2==Tanh[10*K]*((1/Abs[K]+eps)-coef/K^2)},{x,-5,5},{y,-5,5},PlotLegends->Automatic, ContourStyle->{Opacity[0.5]},AxesLabel-> Automatic],{coef, 0.0, 0.1},{K, -0.1, 0.1}]
ContourPlot3D[{Tanh[10*K]*x^2+y^2==Tanh[10*K]*(1/(Abs[K]+eps)), Tanh[10*K]*x^2+y^2==Tanh[10*K]*((1/(Abs[K]+eps))-0.05/(K^2 + eps))},{x,-5,5},{y,-5,5}, {K, -0.1, 0.1}, PlotLegends->Automatic, AxesLabel-> {x,y,K}]


vp={3, 4.5, -1.5};
p1=ContourPlot3D[{Sign[K]*x^2+y^2==Sign[K]*(1/(Abs[K]+eps))},{x,-6,6},{y,-6,6}, {K, -0.1, 0.1}, PlotLegends->{"Original"}, AxesLabel-> {x,y,K}, ViewPoint->vp]
p2=ContourPlot3D[{Sign[K]*x^2+y^2==Sign[K]*(1/(Abs[K]+eps)), Sign[K]*x^2+y^2==Sign[K]*((1/(Abs[K]+eps))-0.05/(K^2 + eps))},{x,-6,6},{y,-6,6}, {K, -0.1, 0.1}, PlotLegends->{"Original", "Lower bound"}, AxesLabel-> {x,y,K}, ViewPoint->vp]
Export["/home/oskopek/git/mvae-paper/figures/manifold_across_K=0.pdf", p1];
Export["/home/oskopek/git/mvae-paper/figures/manifold_augmented_across_K=0.pdf", p2];
(*ContourPlot3D[{Tanh[10*K]*x^2+y^2\[Equal]Tanh[10*K]*(1/(Abs[K]+eps)), Tanh[10*K]*x^2+y^2\[Equal]Tanh[10*K]*((1/(Abs[K]+eps))-0.05/(K^2 + eps))},{x,-8,8},{y,-8,8}, {K, -0.1, 0.1}, PlotLegends->{"Original", "Lower bound"}, AxesLabel-> {x,y,K},ViewPoint\[Rule]vp]*)


vp={3, -1.3, 0.3};
p1=Plot3D[(2(Pi^(n/2)))/(Sqrt[K]^(n-1 ) Gamma[n/2]),{K, 0.8, 1.3},{n,1,20}, AxesLabel->{"Curvature","Dim","Surface Area"}, ViewPoint->vp, BoxRatios -> 1]
p2=Plot[(2(Pi^(n/2)))/(Sqrt[1]^(n-1 ) Gamma[n/2]),{n,1,25}, AxesLabel->{"Dim","Surface Area"}]
Export["/home/oskopek/git/mvae-paper/figures/surface_area_of_sphere.pdf", p1];
Export["/home/oskopek/git/mvae-paper/figures/surface_area_of_sphere_K=1.pdf", p2];





(* ::InheritFromParent:: *)
(**)

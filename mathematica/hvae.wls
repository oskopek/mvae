#!/usr/bin/env wolframscript
(* ::Package:: *)

lorentzProduct[x_,y_] := x . y - 2*x[[1]]*y[[1]];
lorentzNorm[x_] := Sqrt[lorentzProduct[x,x]];
lorentzDist[a_, b_] := ArcCosh[-lorentzProduct[a, b]];


expMap[u_, mu_] := Module[{norm},
	norm = lorentzNorm[u];
	Cosh[norm]*mu + Sinh[norm]*u /norm
];

expMapR[u_, mu_, r_] := Module[{norm},
	norm = Cosh[lorentzNorm[u]/r]/r;
	norm * mu + Sinh[ArcCosh[norm]]*u/ArcCosh[norm]
];

invExpMap[z_, mu_] := Module[{alpha},
	alpha = -lorentzProduct[mu, z];
	ArcCosh[alpha]/(Sqrt[alpha^2 -1]) (z - alpha*mu)
];

mu0= {1,0,0};
mu = {2,1,Sqrt[2]}; (* has to have lorentzNorm^2 = -1 *)
mu2= {Sqrt[5],1,Sqrt[3]};
u  = {0,2,-Sqrt[2]};

(*
lorentzProduct[mu,u]
mu[[1]]
lorentzProduct[mu,mu]
lorentzProduct[mu2,mu2]

eMapped = expMap[u=u, mu=mu];
u
N[eMapped]
N[invExpMap[z=eMapped, mu=mu]]

(* this should be the same *)
N[lorentzDist[mu, eMapped]]
N[Norm[u,p=2]]
N[lorentzNorm[u]]
*)


parallelTransport[from_, to_, point_] := Module[{alpha},
	alpha = -lorentzProduct[from, to];
	point + lorentzProduct[to - alpha*from, point]/(alpha + 1) (from + to)
];

(*
ptU = parallelTransport[from=mu, to=mu2, point=u];
u
N[ptU]
ptOrig = parallelTransport[from=mu2, to=mu, point=ptU];
N[lorentzProduct[ptOrig, mu]]
*)


proj[mu_, v_] := Module[{u},
	u = parallelTransport[mu0, mu, v];
	expMap[u, mu]
];

invProj[mu_, z_] := Module[{u},
	u = invExpMap[z, mu];
	parallelTransport[mu, mu0, u]
];

v = {0, 1, 2}
p = proj[mu, v];
N[lorentzProduct[p, p]]
vhat = invProj[mu, p];
Round[N[vhat]]


pdfP[z_, mu_, d_] := Module[{v, vTilde, diff},
	v = invProj[mu, z];
	vTilde = {v[[2]], v[[3]]};
	diff = D[proj[mu, {a,b,c}], {{a,b,c},1}] /. {a -> v[[1]], b -> v[[2]], c -> v[[3]]};
	PDF[d, vTilde] / Det[diff]
];

pdfP2[z_, mu_, d_] := Module[{v, vTilde, r, alpha, det},
	u = invExpMap[z, mu];
	v = parallelTransport[mu, mu0, u];
	vTilde = {v[[2]], v[[3]]};
	r = lorentzNorm[u];
	alpha = -lorentzProduct[mu0, mu];
	det = (Sinh[r]/r)^2 * Cosh[r]* alpha;
	PDF[d, vTilde] / det
];

mu0 = {1,0,0};
mu = {2,1,Sqrt[2]};
sigma = 2 * IdentityMatrix[2];

(* Sample from G(mu, sigma)*)
pV = MultinormalDistribution[sigma];
vTilde = RandomVariate[pV, 1];
v = {0, vTilde[[1,1]], vTilde[[1,2]]};
z = proj[mu, v];

sampleG[mu_, pV_] := Module[{vTilde, v},
	vTilde = RandomVariate[pV, 1];
	v = {0, vTilde[[1,1]], vTilde[[1,2]]};
	proj[mu, v]
];

(*Probability of z*)
N[Log[pdfP[z, mu, pV]]]
N[Log[pdfP2[z, mu, pV]]]

ppdf[z_] := pdfP[z, mu0, MultinormalDistribution[IdentityMatrix[2]]];
qpdf[z_] := pdfP[z, mu, pV];



scale = 0.01;

mu0 = {1,0,0};
mu = {2,1,Sqrt[2]};

sigma = scale * IdentityMatrix[2];

(* Sample from G(mu, sigma)*)
pV = MultinormalDistribution[sigma];

PDF[pV, RandomVariate[pV,1][[1]]]

z = sampleG[mu, pV]
z = {2.01, 1.0199, Sqrt[2]}

lorentzProduct[z, z]

(*Probability of z*)
N[Log[pdfP[z, mu, pV]]]
N[Log[pdfP2[z, mu, pV]]]


(* attempt at KL *)
(*
Integrate[ppdf[{a,b,c}] * (Log[ppdf[{a,b,c}]]-Log[qpdf[{a,b,c}]]), {a, 0, 10}, {b, -10, 10}, {c, -10, 10}]
*)


(* monte carlo KL *)
mu0 = {1,0,0};
mu = {1,0,0};
mu = {2,1,Sqrt[2]};

pV0 = MultinormalDistribution[{0, 0}, IdentityMatrix[2]];
sigma = {{1.2,0},
		 {0,1.2}};
PositiveDefiniteMatrixQ[sigma]
pV = MultinormalDistribution[{0,0}, sigma];


ppdf[z_] := pdfP[z, mu0, pV0];
qpdf[z_] := pdfP[z, mu, pV];

z = sampleG[mu, pV];
klTermOrig[z_, qpdf_, ppdf_] := qpdf[z] * (Log[qpdf[z]]-Log[ppdf[z]]); 
klTerm[z_, qpdf_, ppdf_] := Log[qpdf[z]]-Log[ppdf[z]];
N[klTerm[sampleG[mu,pV], qpdf, ppdf]]


maxSamples = 500
psamples = Table[sampleG[mu0, pV0], {i, maxSamples}];
qsamples = Table[sampleG[mu0,  pV], {i, maxSamples}];
p1 = ListPointPlot3D[psamples, PlotStyle->Red];
p2 = ListPointPlot3D[qsamples, PlotStyle->Blue];
Show[p1, p2]


toPoincare[x_] := Module[{x0, x1, numerator, numerator2},
	x0 = Transpose[Take[Transpose[x], 1]];
	x1 = Transpose[Drop[Transpose[x], 1]];
	numerator = x0 + 1;
	numerator2 = Join[numerator, numerator, 2];
	x1 / numerator2
];

mu0 = {1,0,0};
mu = {2,1,Sqrt[2]};

lorentzProduct[mu, mu]
sigma = 2 * IdentityMatrix[2];

(* Sample from G(mu, sigma)*)
pV = MultinormalDistribution[sigma];
pV0 = MultinormalDistribution[IdentityMatrix[2]];

maxSamples = 500
psamples = Table[sampleG[mu, pV0], {i, maxSamples}];
qsamples = Table[sampleG[mu,  pV], {i, maxSamples}];

means = toPoincare[{mu, mu0}];
psamples2 = toPoincare[psamples];
qsamples2 = toPoincare[qsamples];

p0 = ListPlot[means, PlotStyle->Black]
p1 = ListPlot[psamples2, PlotStyle->Red]
p2 = ListPlot[qsamples2, PlotStyle->Blue]

Show[p1, p2, p0]


Mean[Table[N[klTerm[sampleG[mu,pV], qpdf, ppdf]], {i, maxSamples}]]


Mean[Table[N[klTerm[sampleG[mu,pV], qpdf, ppdf]], {i, maxSamples}]]


Total[Table[N[klTermOrig[sampleG[mu,pV], qpdf, ppdf]], {i, maxSamples}]]


dif = D[ArcCosh[1+x]^2,x]
Limit[dif,x->0]

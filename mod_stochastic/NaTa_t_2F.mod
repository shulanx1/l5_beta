:Reference :Colbert and Pan 2002
:modified with stochastic channel noise, NNa = 3500

NEURON	{
	SUFFIX NaTa_t_2F
	USEION na READ ena WRITE ina
	RANGE gNaTa_tbar, gNaTa_t, ina, NNaTa_t
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gNaTa_tbar = 0.00001 (S/cm2)
	NNaTa_t = 3500
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)
	gNaTa_t	(S/cm2)
	mInf
	mTau
	mAlpha
	mBeta
	hInf
	hTau
	hAlpha
	hBeta
	SDm
	SDh
}

STATE	{
	m
	h
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gNaTa_t = gNaTa_tbar*m*m*m*h
	ina = gNaTa_t*(v-ena)
}

DERIVATIVE states	{
	rates()
	m' = (mInf-m)/mTau  + normrand(0, SDm)
	h' = (hInf-h)/hTau  + normrand(0, SDh)
}

INITIAL{
	rates()
	m = mInf
	h = hInf
}

PROCEDURE rates(){
  LOCAL qt
  qt = 2.3^((34-21)/10)
	
  UNITSOFF
    if(v == -38){
    	v = v+0.0001
    }
		mAlpha = (0.182 * (v- -38))/(1-(exp(-(v- -38)/6)))
		mBeta  = (0.124 * (-v -38))/(1-(exp(-(-v -38)/6)))
		mTau = (1/(mAlpha + mBeta))/qt
		mInf = mAlpha/(mAlpha + mBeta)
		SDm = sqrt(fabs(mAlpha*(1-m)+mBeta*m)/(0.05*NNaTa_t*3))

    if(v == -66){
      v = v + 0.0001
    }

		hAlpha = (-0.015 * (v- -66))/(1-(exp((v- -66)/6)))
		hBeta  = (-0.015 * (-v -66))/(1-(exp((-v -66)/6)))
		hTau = (1/(hAlpha + hBeta))/qt
		hInf = hAlpha/(hAlpha + hBeta)
		SDh = sqrt(fabs(hAlpha*(1-h)+hBeta*h)/(0.05*NNaTa_t))
	UNITSON
}
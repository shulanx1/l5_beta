:Comment : The transient component of the K current
:Reference : :		Voltage-gated K+ channels in layer 5 neocortical pyramidal neurones from young rats:subtypes and gradients,Korngreen and Sakmann, J. Physiology, 2000
:Comment : shifted -10 mv to correct for junction potential
:Comment: corrected rates using q10 = 2.3, target temperature 34, orginal 21

: modifed to add stpchastic channel noise, NK = 166

NEURON	{
	SUFFIX K_Tst_2F
	USEION k READ ek WRITE ik
	RANGE gK_Tstbar, gK_Tst, ik, NK_Tst
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gK_Tstbar = 0.00001 (S/cm2)
	NK_Tst = 166
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)
	gK_Tst	(S/cm2)
	mInf
	mTau
	hInf
	hTau
	SDm_1
	SDh_1
}

STATE	{
	m
	h
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gK_Tst = gK_Tstbar*(m^4)*h
	ik = gK_Tst*(v-ek)
}

DERIVATIVE states	{
	rates()
	m' = (mInf-m)/mTau + normrand(0, SDm_1)
	h' = (hInf-h)/hTau + normrand(0, SDh_1)
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
		v = v + 10
		mInf =  1/(1 + exp(-(v+0)/19))
		mTau =  (0.34+0.92*exp(-((v+71)/59)^2))/qt
		hInf =  1/(1 + exp(-(v+66)/-10))
		hTau =  (8+49*exp(-((v+73)/23)^2))/qt
		SDm_1 = sqrt(fabs(mInf*(1-m)/mTau+(1-mInf)/mTau*m)/(0.05*NK_Tst*4))
		SDh_1 = sqrt(fabs(hInf*(1-h)/hTau+(1-hInf)/hTau*h)/(0.05*NK_Tst))
		v = v - 10
	UNITSON
}

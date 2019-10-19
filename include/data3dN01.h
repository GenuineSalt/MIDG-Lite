#ifndef DATA3dN01 

#define DATA3dN01 1

double p_r[4] = {                -1 ,                 1 ,                -1 ,                -1 };
double p_s[4] = {               -1 ,                -1 ,                 1 ,                -1 };
double p_t[4] = {               -1 ,                -1 ,                -1 ,                 1 };
double p_Dr[4][4] = {{             -0.5 ,               0.5 , 3.20493781063927e-17 , 3.20493781063927e-17 },
{             -0.5 ,               0.5 , 3.20493781063927e-17 , 3.20493781063927e-17 },
{             -0.5 ,               0.5 , 3.20493781063927e-17 , 3.20493781063927e-17 },
{             -0.5 ,               0.5 , 3.20493781063927e-17 , 3.20493781063927e-17 }};
double p_Ds[4][4] = {{             -0.5 ,                -0 ,               0.5 , 6.40987562127855e-17 },
{             -0.5 , 7.0216669371534e-17 ,               0.5 ,                 0 },
{             -0.5 ,                -0 ,               0.5 , 6.40987562127855e-17 },
{             -0.5 ,                -0 ,               0.5 , 6.40987562127855e-17 }};
double p_Dt[4][4] = {{             -0.5 , -3.5108334685767e-17 ,                -0 ,               0.5 },
{             -0.5 , 7.0216669371534e-17 ,                -0 ,               0.5 },
{             -0.5 , -7.0216669371534e-17 ,                -0 ,               0.5 },
{             -0.5 , -3.5108334685767e-17 ,                -0 ,               0.5 }};
double p_LIFT[4][12] = {{                3 , 0.499999999999999 , 0.499999999999999 ,                 3 ,               0.5 ,               0.5 ,                -2 ,                -2 ,                -2 ,                 3 ,               0.5 ,               0.5 },
{0.499999999999999 ,                 3 , 0.499999999999999 ,               0.5 ,                 3 ,               0.5 ,                 3 ,               0.5 ,               0.5 ,                -2 ,                -2 ,                -2 },
{              0.5 ,               0.5 ,                 3 ,                -2 ,                -2 ,                -2 ,               0.5 ,                 3 ,               0.5 ,               0.5 ,                 3 ,               0.5 },
{               -2 ,                -2 ,                -2 ,               0.5 ,               0.5 ,                 3 ,               0.5 ,               0.5 ,                 3 ,               0.5 ,               0.5 ,                 3 }};
int p_Fmask[4][3] = {{0 , 1 , 2 },
{0 , 1 , 3 },
{1 , 2 , 3 },
{0 , 2 , 3 }};

#endif

# Clean program: registered analysis

Status: **confirmatory**.

## line

| Claim | Result |
|---|---|
| control | True |
| A1_emergence | True |
| A2_dose | True |
| A3_structure | True |
| A4_timing | True |
| B1_route | inconclusive |
| B2_persistence | True |
| C1_compounding | True |
| D1_reach | False |
| D2_reach_timing | True |

Contrasts (mean [95% CI]; positive seeds; Holm-adjusted p where tested):

- `control_imposed_minus_baseline`: +0.420 [+0.350, +0.490]; 35/36
- `A1_emergence`: +0.418 [+0.349, +0.486]; 36/36; p_Holm=1.8e-13
- `A2_dose`: +0.297 [+0.203, +0.391]; 31/36; p_Holm=6.8e-07
- `A3_structure`: +0.234 [+0.099, +0.369]; 27/36; p_Holm=0.0024
- `A4_at_acquisition`: +0.386 [+0.320, +0.452]; 36/36; p_Holm=5e-13
- `A4_growth_after`: +0.032 [+0.015, +0.048]; 27/36
- `B1_latest_minus_first_switch`: -0.079 [-0.121, -0.037]; 8/36
- `B1_route`: inconclusive
- `B2_persistence`: +0.412 [+0.344, +0.479]; 36/36; p_Holm=1.8e-13
- `C1_compounding`: +0.255 [+0.201, +0.310]; 33/36; p_Holm=1.2e-10
- `D1_reach_threshold`: +1.047 [+0.862, +1.232]; 36/36; p_Holm=9.7e-13
- `D2_reach_growth_threshold`: +0.041 [-0.008, +0.090]; 22/36
- `D1_reach_near`: -0.149 [-0.273, -0.026]; 11/36; p_Holm=0.019
- `D2_reach_growth_near`: -0.022 [-0.120, +0.075]; 17/36

## circle

| Claim | Result |
|---|---|
| control | True |
| A1_emergence | True |
| A2_dose | True |
| A3_structure | False |
| A4_timing | False |
| B1_route | entrenchment |
| B2_persistence | True |
| C1_compounding | True |
| D1_reach | False |
| D2_reach_timing | False |
| F1_window | True |

Contrasts (mean [95% CI]; positive seeds; Holm-adjusted p where tested):

- `control_imposed_minus_baseline`: +0.393 [+0.341, +0.445]; 35/36
- `A1_emergence`: +0.324 [+0.275, +0.373]; 36/36; p_Holm=2.7e-14
- `A2_dose`: +0.225 [+0.168, +0.282]; 33/36; p_Holm=1e-08
- `A3_structure`: +0.088 [-0.059, +0.235]; 24/36; p_Holm=0.23
- `A4_at_acquisition`: +0.239 [+0.182, +0.296]; 34/36; p_Holm=2.8e-09
- `A4_growth_after`: +0.085 [+0.063, +0.107]; 34/36
- `B1_latest_minus_first_switch`: -0.164 [-0.221, -0.106]; 6/36
- `B1_route`: entrenchment
- `B2_persistence`: +0.299 [+0.250, +0.347]; 36/36; p_Holm=2e-13
- `C1_compounding`: +0.200 [+0.153, +0.247]; 33/36; p_Holm=2.6e-09
- `D1_reach_threshold`: +0.614 [+0.485, +0.742]; 34/36; p_Holm=1.4e-10
- `D2_reach_growth_threshold`: +0.031 [-0.059, +0.121]; 21/36
- `D1_reach_near`: -0.112 [-0.216, -0.008]; 10/36; p_Holm=0.071
- `D2_reach_growth_near`: +0.047 [-0.075, +0.169]; 21/36
- `F1_window_ls10`: +0.077 [+0.016, +0.138]; 28/36; p_Holm=0.044
- `F1_window_lr03`: +0.068 [+0.017, +0.120]; 23/36; p_Holm=0.043


# mini_veros vs veros: comparison matrix report

timestamp: latest -> mixed snapshots -- `20260828T232813Z` (2026-08-28 23:28:13 UTC): acc_minimal; `20260828T233138Z` (2026-08-28 23:31:38 UTC): global_biharmonic_friction, global_minimal; `20260830T064015Z` (2026-08-30 06:40:15 UTC): acc_basic, acc_biharmonic_friction, acc_biharmonic_mixing, acc_bottom_friction_var, acc_eke_isopycnal_diffusion_off, acc_eke_superbee_off, acc_explicit_vert_friction, acc_full, acc_hor_diffusion, acc_kappaH_profile_off, acc_maximal, acc_no_hor_friction, acc_no_neutral_diffusion, acc_no_skew_diffusion, acc_no_tke, acc_noslip_lateral, acc_quadratic_bottom_friction, acc_ray_friction, acc_surface_pressure, acc_tke_superbee_advection, global_biharmonic_mixing, global_default, global_hor_diffusion, global_maximal, global_no_eke, global_no_neutral_diffusion, global_no_skew_diffusion, global_surface_pressure.

3/31 variants within tolerance (rel error < 1e-06, psi scale-normalized < 1e-06).

![timing](matrix_figures/timing_summary.png)

| variant | group | status | worst error | mini ms/step | veros ms/step | speedup |
|---|---|---|---|---|---|---|
| acc_basic | acc | FAIL | 1.95e+00 | 9.61 | 39.76 | 4.1x |
| acc_biharmonic_friction | acc | FAIL | 2.00e+00 | 11.29 | 41.12 | 3.6x |
| acc_biharmonic_mixing | acc | FAIL | 1.95e+00 | 10.39 | 41.40 | 4.0x |
| acc_bottom_friction_var | acc | FAIL | 5.96e-03 | 9.32 | 39.90 | 4.3x |
| acc_eke_isopycnal_diffusion_off | acc | FAIL | 1.03e+00 | 11.69 | 43.90 | 3.8x |
| acc_eke_superbee_off | acc | FAIL | 3.54e-02 | 11.48 | 42.58 | 3.7x |
| acc_explicit_vert_friction | acc | FAIL | 4.14e-02 | 7.30 | 36.40 | 5.0x |
| acc_full | acc | FAIL | 1.44e+00 | 11.85 | 43.25 | 3.6x |
| acc_hor_diffusion | acc | FAIL | 1.62e+00 | 9.98 | 40.98 | 4.1x |
| acc_kappaH_profile_off | acc | FAIL | 1.77e+00 | 9.91 | 40.65 | 4.1x |
| acc_maximal | acc | FAIL | 1.66e+00 | 13.58 | 48.81 | 3.6x |
| acc_minimal | acc | ok | 6.37e-04 | 3.01 | 7.03 | 2.3x |
| acc_no_hor_friction | acc | FAIL | 2.00e+00 | 10.40 | 39.82 | 3.8x |
| acc_no_neutral_diffusion | acc | FAIL | 2.00e+00 | 7.27 | 27.45 | 3.8x |
| acc_no_skew_diffusion | acc | FAIL | 1.98e+00 | 9.89 | 35.83 | 3.6x |
| acc_no_tke | acc | FAIL | 4.14e-02 | 7.92 | 38.10 | 4.8x |
| acc_noslip_lateral | acc | FAIL | 2.00e+00 | 9.97 | 39.97 | 4.0x |
| acc_quadratic_bottom_friction | acc | FAIL | 1.96e+00 | 10.07 | 40.39 | 4.0x |
| acc_ray_friction | acc | FAIL | 1.78e+00 | 9.56 | 41.08 | 4.3x |
| acc_surface_pressure | acc | FAIL | 1.51e+00 | 8.45 | 39.80 | 4.7x |
| acc_tke_superbee_advection | acc | FAIL | 1.99e+00 | 10.42 | 41.83 | 4.0x |
| global_biharmonic_friction | global | ok | 4.84e-04 | 18.82 | 33.51 | 1.8x |
| global_biharmonic_mixing | global | FAIL | 1.47e+00 | 45.93 | 95.34 | 2.1x |
| global_default | global | FAIL | 1.75e+00 | 43.49 | 90.79 | 2.1x |
| global_hor_diffusion | global | FAIL | 4.41e-02 | 45.47 | 94.15 | 2.1x |
| global_maximal | global | FAIL | 1.80e+00 | 47.41 | 100.50 | 2.1x |
| global_minimal | global | ok | 1.16e-04 | 13.32 | 19.11 | 1.4x |
| global_no_eke | global | FAIL | 1.99e+00 | 41.07 | 84.94 | 2.1x |
| global_no_neutral_diffusion | global | FAIL | 1.97e+00 | 36.73 | 68.09 | 1.9x |
| global_no_skew_diffusion | global | FAIL | 1.82e+00 | 43.90 | 83.84 | 1.9x |
| global_surface_pressure | global | FAIL | 7.74e+04 | 42.00 | 90.44 | 2.2x |

## per-variant detail

### acc_basic (FAIL)
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_basic_error_evolution.png)
![acc_basic_temp_evolution](matrix_figures/acc_basic_temp_evolution.gif)
![acc_basic_psi_evolution](matrix_figures/acc_basic_psi_evolution.gif)

### acc_biharmonic_friction (FAIL)
overrides: `{'enable_hor_friction': False, 'enable_biharmonic_friction': True, 'A_hbi': 100000000000.0}`
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_biharmonic_friction_error_evolution.png)
![acc_biharmonic_friction_temp_evolution](matrix_figures/acc_biharmonic_friction_temp_evolution.gif)
![acc_biharmonic_friction_psi_evolution](matrix_figures/acc_biharmonic_friction_psi_evolution.gif)

### acc_biharmonic_mixing (FAIL)
overrides: `{'enable_biharmonic_mixing': True, 'K_hbi': 100000000000.0}`
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_biharmonic_mixing_error_evolution.png)
![acc_biharmonic_mixing_temp_evolution](matrix_figures/acc_biharmonic_mixing_temp_evolution.gif)
![acc_biharmonic_mixing_psi_evolution](matrix_figures/acc_biharmonic_mixing_psi_evolution.gif)

### acc_bottom_friction_var (FAIL)
overrides: `{'enable_bottom_friction_var': True}`
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_bottom_friction_var_error_evolution.png)
![acc_bottom_friction_var_temp_evolution](matrix_figures/acc_bottom_friction_var_temp_evolution.gif)
![acc_bottom_friction_var_psi_evolution](matrix_figures/acc_bottom_friction_var_psi_evolution.gif)

### acc_eke_isopycnal_diffusion_off (FAIL)
overrides: `{'enable_eke_isopycnal_diffusion': False}`
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_eke_isopycnal_diffusion_off_error_evolution.png)
![acc_eke_isopycnal_diffusion_off_temp_evolution](matrix_figures/acc_eke_isopycnal_diffusion_off_temp_evolution.gif)
![acc_eke_isopycnal_diffusion_off_psi_evolution](matrix_figures/acc_eke_isopycnal_diffusion_off_psi_evolution.gif)

### acc_eke_superbee_off (FAIL)
overrides: `{'enable_eke_superbee_advection': False}`
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_eke_superbee_off_error_evolution.png)
![acc_eke_superbee_off_temp_evolution](matrix_figures/acc_eke_superbee_off_temp_evolution.gif)
![acc_eke_superbee_off_psi_evolution](matrix_figures/acc_eke_superbee_off_psi_evolution.gif)

### acc_explicit_vert_friction (FAIL)
overrides: `{'enable_implicit_vert_friction': False, 'enable_explicit_vert_friction': True, 'enable_tke': False}`
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_explicit_vert_friction_error_evolution.png)
![acc_explicit_vert_friction_temp_evolution](matrix_figures/acc_explicit_vert_friction_temp_evolution.gif)
![acc_explicit_vert_friction_psi_evolution](matrix_figures/acc_explicit_vert_friction_psi_evolution.gif)

### acc_full (FAIL)
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_full_error_evolution.png)
![acc_full_temp_evolution](matrix_figures/acc_full_temp_evolution.gif)
![acc_full_psi_evolution](matrix_figures/acc_full_psi_evolution.gif)

### acc_hor_diffusion (FAIL)
overrides: `{'enable_hor_diffusion': True, 'K_h': 1000.0}`
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_hor_diffusion_error_evolution.png)
![acc_hor_diffusion_temp_evolution](matrix_figures/acc_hor_diffusion_temp_evolution.gif)
![acc_hor_diffusion_psi_evolution](matrix_figures/acc_hor_diffusion_psi_evolution.gif)

### acc_kappaH_profile_off (FAIL)
overrides: `{'enable_kappaH_profile': False}`
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_kappaH_profile_off_error_evolution.png)
![acc_kappaH_profile_off_temp_evolution](matrix_figures/acc_kappaH_profile_off_temp_evolution.gif)
![acc_kappaH_profile_off_psi_evolution](matrix_figures/acc_kappaH_profile_off_psi_evolution.gif)

### acc_maximal (FAIL)
overrides: `{'enable_biharmonic_friction': True, 'A_hbi': 100000000000.0, 'enable_noslip_lateral': True, 'enable_quadratic_bottom_friction': True, 'r_quad_bot': 0.001, 'enable_hor_diffusion': True, 'K_h': 1000.0, 'enable_biharmonic_mixing': True, 'K_hbi': 100000000000.0, 'enable_tke_superbee_advection': True}`
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_maximal_error_evolution.png)
![acc_maximal_temp_evolution](matrix_figures/acc_maximal_temp_evolution.gif)
![acc_maximal_psi_evolution](matrix_figures/acc_maximal_psi_evolution.gif)

### acc_minimal (ok)
overrides: `{'enable_hor_friction': False, 'enable_bottom_friction': False, 'enable_neutral_diffusion': False, 'enable_skew_diffusion': False, 'enable_tke': False}`
generated: `20260828T232813Z`, 5 steps @ interval 2

![errors](matrix_figures/acc_minimal_error_evolution.png)
![acc_minimal_temp_evolution](matrix_figures/acc_minimal_temp_evolution.gif)
![acc_minimal_psi_evolution](matrix_figures/acc_minimal_psi_evolution.gif)

### acc_no_hor_friction (FAIL)
overrides: `{'enable_hor_friction': False}`
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_no_hor_friction_error_evolution.png)
![acc_no_hor_friction_temp_evolution](matrix_figures/acc_no_hor_friction_temp_evolution.gif)
![acc_no_hor_friction_psi_evolution](matrix_figures/acc_no_hor_friction_psi_evolution.gif)

### acc_no_neutral_diffusion (FAIL)
overrides: `{'enable_neutral_diffusion': False, 'enable_skew_diffusion': False}`
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_no_neutral_diffusion_error_evolution.png)
![acc_no_neutral_diffusion_temp_evolution](matrix_figures/acc_no_neutral_diffusion_temp_evolution.gif)
![acc_no_neutral_diffusion_psi_evolution](matrix_figures/acc_no_neutral_diffusion_psi_evolution.gif)

### acc_no_skew_diffusion (FAIL)
overrides: `{'enable_skew_diffusion': False}`
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_no_skew_diffusion_error_evolution.png)
![acc_no_skew_diffusion_temp_evolution](matrix_figures/acc_no_skew_diffusion_temp_evolution.gif)
![acc_no_skew_diffusion_psi_evolution](matrix_figures/acc_no_skew_diffusion_psi_evolution.gif)

### acc_no_tke (FAIL)
overrides: `{'enable_tke': False}`
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_no_tke_error_evolution.png)
![acc_no_tke_temp_evolution](matrix_figures/acc_no_tke_temp_evolution.gif)
![acc_no_tke_psi_evolution](matrix_figures/acc_no_tke_psi_evolution.gif)

### acc_noslip_lateral (FAIL)
overrides: `{'enable_noslip_lateral': True}`
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_noslip_lateral_error_evolution.png)
![acc_noslip_lateral_temp_evolution](matrix_figures/acc_noslip_lateral_temp_evolution.gif)
![acc_noslip_lateral_psi_evolution](matrix_figures/acc_noslip_lateral_psi_evolution.gif)

### acc_quadratic_bottom_friction (FAIL)
overrides: `{'enable_bottom_friction': False, 'enable_quadratic_bottom_friction': True, 'r_quad_bot': 0.001}`
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_quadratic_bottom_friction_error_evolution.png)
![acc_quadratic_bottom_friction_temp_evolution](matrix_figures/acc_quadratic_bottom_friction_temp_evolution.gif)
![acc_quadratic_bottom_friction_psi_evolution](matrix_figures/acc_quadratic_bottom_friction_psi_evolution.gif)

### acc_ray_friction (FAIL)
overrides: `{'enable_ray_friction': True, 'r_ray': 1e-06}`
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_ray_friction_error_evolution.png)
![acc_ray_friction_temp_evolution](matrix_figures/acc_ray_friction_temp_evolution.gif)
![acc_ray_friction_psi_evolution](matrix_figures/acc_ray_friction_psi_evolution.gif)

### acc_surface_pressure (FAIL)
overrides: `{'enable_streamfunction': False}`
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_surface_pressure_error_evolution.png)
![acc_surface_pressure_temp_evolution](matrix_figures/acc_surface_pressure_temp_evolution.gif)
![acc_surface_pressure_psi_evolution](matrix_figures/acc_surface_pressure_psi_evolution.gif)

### acc_tke_superbee_advection (FAIL)
overrides: `{'enable_tke_superbee_advection': True}`
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_tke_superbee_advection_error_evolution.png)
![acc_tke_superbee_advection_temp_evolution](matrix_figures/acc_tke_superbee_advection_temp_evolution.gif)
![acc_tke_superbee_advection_psi_evolution](matrix_figures/acc_tke_superbee_advection_psi_evolution.gif)

### global_biharmonic_friction (ok)
overrides: `{'enable_hor_friction': False, 'enable_biharmonic_friction': True, 'A_hbi': 1000000000000.0}`
generated: `20260828T233138Z`, 5 steps @ interval 2

![errors](matrix_figures/global_biharmonic_friction_error_evolution.png)
![global_biharmonic_friction_temp_evolution](matrix_figures/global_biharmonic_friction_temp_evolution.gif)
![global_biharmonic_friction_psi_evolution](matrix_figures/global_biharmonic_friction_psi_evolution.gif)

### global_biharmonic_mixing (FAIL)
overrides: `{'enable_biharmonic_mixing': True, 'K_hbi': 1000000000000.0}`
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/global_biharmonic_mixing_error_evolution.png)
![global_biharmonic_mixing_temp_evolution](matrix_figures/global_biharmonic_mixing_temp_evolution.gif)
![global_biharmonic_mixing_psi_evolution](matrix_figures/global_biharmonic_mixing_psi_evolution.gif)

### global_default (FAIL)
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/global_default_error_evolution.png)
![global_default_temp_evolution](matrix_figures/global_default_temp_evolution.gif)
![global_default_psi_evolution](matrix_figures/global_default_psi_evolution.gif)

### global_hor_diffusion (FAIL)
overrides: `{'enable_hor_diffusion': True, 'K_h': 1000.0}`
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/global_hor_diffusion_error_evolution.png)
![global_hor_diffusion_temp_evolution](matrix_figures/global_hor_diffusion_temp_evolution.gif)
![global_hor_diffusion_psi_evolution](matrix_figures/global_hor_diffusion_psi_evolution.gif)

### global_maximal (FAIL)
overrides: `{'enable_biharmonic_friction': True, 'A_hbi': 1000000000000.0, 'enable_noslip_lateral': True, 'enable_hor_diffusion': True, 'K_h': 1000.0, 'enable_biharmonic_mixing': True, 'K_hbi': 1000000000000.0, 'enable_tke_superbee_advection': True, 'enable_eke_superbee_advection': True}`
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/global_maximal_error_evolution.png)
![global_maximal_temp_evolution](matrix_figures/global_maximal_temp_evolution.gif)
![global_maximal_psi_evolution](matrix_figures/global_maximal_psi_evolution.gif)

### global_minimal (ok)
overrides: `{'enable_hor_friction': False, 'enable_neutral_diffusion': False, 'enable_skew_diffusion': False, 'enable_eke': False}`
generated: `20260828T233138Z`, 5 steps @ interval 2

![errors](matrix_figures/global_minimal_error_evolution.png)
![global_minimal_temp_evolution](matrix_figures/global_minimal_temp_evolution.gif)
![global_minimal_psi_evolution](matrix_figures/global_minimal_psi_evolution.gif)

### global_no_eke (FAIL)
overrides: `{'enable_eke': False}`
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/global_no_eke_error_evolution.png)
![global_no_eke_temp_evolution](matrix_figures/global_no_eke_temp_evolution.gif)
![global_no_eke_psi_evolution](matrix_figures/global_no_eke_psi_evolution.gif)

### global_no_neutral_diffusion (FAIL)
overrides: `{'enable_neutral_diffusion': False, 'enable_skew_diffusion': False}`
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/global_no_neutral_diffusion_error_evolution.png)
![global_no_neutral_diffusion_temp_evolution](matrix_figures/global_no_neutral_diffusion_temp_evolution.gif)
![global_no_neutral_diffusion_psi_evolution](matrix_figures/global_no_neutral_diffusion_psi_evolution.gif)

### global_no_skew_diffusion (FAIL)
overrides: `{'enable_skew_diffusion': False}`
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/global_no_skew_diffusion_error_evolution.png)
![global_no_skew_diffusion_temp_evolution](matrix_figures/global_no_skew_diffusion_temp_evolution.gif)
![global_no_skew_diffusion_psi_evolution](matrix_figures/global_no_skew_diffusion_psi_evolution.gif)

### global_surface_pressure (FAIL)
overrides: `{'enable_streamfunction': False}`
generated: `20260830T064015Z`, 10950 steps @ interval 150

![errors](matrix_figures/global_surface_pressure_error_evolution.png)
![global_surface_pressure_temp_evolution](matrix_figures/global_surface_pressure_temp_evolution.gif)
![global_surface_pressure_psi_evolution](matrix_figures/global_surface_pressure_psi_evolution.gif)

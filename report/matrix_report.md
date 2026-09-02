# mini_veros vs veros: comparison matrix report

timestamp: 20260902T072723Z -> `20260902T072723Z` (2026-09-02 07:27:23 UTC).

A 30-year run cannot agree point-wise -- roundoff-level differences grow to the size of the flow's own variability, and real veros does the same against itself (see `divergence_report.md`).


## Metrics

**<u>Columns:</u>** 

- **horizon** is the last step at which every field still agreed to a scale-normalized max error below 1e-06 (`>` means it never stopped agreeing); 
- **rel L2** is the relative L2 distance at the final step
- **corr** is the worst field's pattern correlation there
- **clim ratio** is the climatology difference divided by veros's own -- below 1.0 means the models are closer to each other than veros is to itself. ( We assume that we reach an equilibrium after the first half of the run ~15 years)

### Formulas 

For one field at one recorded step, let $m$ and $v$ be the mini_veros and veros fields flattened over the grid, $N$ their length, and $\overline{x}$ a spatial mean. Cells where either side is not finite are dropped before any of this.

**Scale-normalized max error.** The largest point-wise disagreement, divided by the reference field's own magnitude :

$$
\mathrm{max\_norm} \;=\; \frac{\max_i \left| m_i - v_i \right|}{\mathrm{rms}(v)}, \qquad \mathrm{rms}(x) = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}
$$


**Relative $L_2$ error.** The whole-field distance :

$$
\mathrm{rel\_L2} \;=\; \frac{\| m - v \|_2}{\| v \|_2}
$$


**Pattern correlation.** The Pearson correlation of the two fields' anomalies : 1 if same structure even shift by an offset :

$$
\mathrm{corr} \;=\; \frac{\sum_i (m_i - \overline{m})(v_i - \overline{v})}{\| m - \overline{m} \|_2 \; \| v - \overline{v} \|_2}
$$

**Agreement horizon.** The first recorded step at which any field's $\mathrm{max\_norm}$ crosses $\varepsilon = 10^{-6}$. 

$$
T_{\mathrm{agree}} = \min \{ t \text{s.t} \mathrm{max\_norm}(t) > \varepsilon \}
$$


**Climatology ratio : ** compare average fields agains veros time variability

Assuming that we reached an equilibrium at the second part of the run : 

$\langle x \rangle_{A}$  : time mean over a window $A$ of records,

 We split a run of $R$ records into its second half $H$ and its third and fourth quarters $Q_3, Q_4$.

$$D \;=\; \langle m \rangle_{H} - \langle v \rangle_{H}, \qquad S \;=\; \langle v \rangle_{Q_3} - \langle v \rangle_{Q_4}, \qquad \mathrm{clim\ ratio} \;=\; \frac{\mathrm{rms}(D)}{\mathrm{rms}(S)}$$

.The ratio is only reported where veros actually varies, $\mathrm{rms}(S) > 10^{-8} \cdot \mathrm{rms}(v)$.

![timing](matrix_figures/timing_summary.png)

| variant | group | horizon | rel L2 | corr | clim ratio |
|---|---|---|---|---|---|
| acc_basic | acc | 900 | 7.93e-03 | 1.0000 | 0.20 |
| acc_biharmonic_friction | acc | 450 | 4.45e-01 | 0.8992 | 0.35 |
| acc_biharmonic_mixing | acc | 900 | 4.12e-02 | 0.9992 | 0.04 |
| acc_bottom_friction_var | acc | 600 | 3.55e-07 | 1.0000 | 0.00 |
| acc_eke_isopycnal_diffusion_off | acc | 1200 | 2.84e-05 | 1.0000 | 0.00 |
| acc_eke_superbee_off | acc | 900 | 6.73e-07 | 1.0000 | 0.00 |
| acc_explicit_vert_friction | acc | 1650 | 5.82e-06 | 1.0000 | 0.00 |
| acc_full | acc | 900 | 8.49e-07 | 1.0000 | 0.00 |
| acc_hor_diffusion | acc | 1350 | 1.54e-04 | 1.0000 | 0.11 |
| acc_kappaH_profile_off | acc | 600 | 1.89e-03 | 1.0000 | 0.01 |
| acc_maximal | acc | 150 | 1.89e-05 | 1.0000 | 0.00 |
| acc_minimal | acc | 1950 | - | - | - |
| acc_no_hor_friction | acc | 300 | 4.33e-01 | 0.9068 | 0.27 |
| acc_no_neutral_diffusion | acc | 900 | 8.08e-03 | 1.0000 | 0.00 |
| acc_no_skew_diffusion | acc | 1050 | 3.00e-03 | 1.0000 | 0.00 |
| acc_no_tke | acc | 1650 | 5.82e-06 | 1.0000 | 0.00 |
| acc_noslip_lateral | acc | 750 | 7.27e-02 | 0.9974 | 0.08 |
| acc_quadratic_bottom_friction | acc | 600 | 1.10e-02 | 0.9999 | 0.07 |
| acc_ray_friction | acc | 600 | 8.54e-07 | 1.0000 | 0.00 |
| acc_surface_pressure | acc | 150 | 2.72e-03 | 1.0000 | 0.00 |
| acc_tke_superbee_advection | acc | 150 | 1.92e-02 | 0.9998 | 0.19 |
| global_biharmonic_friction | global | 600 | - | - | - |
| global_biharmonic_mixing | global | 150 | 3.51e-05 | 1.0000 | 0.00 |
| global_default | global | 150 | 2.17e-05 | 1.0000 | 0.00 |
| global_hor_diffusion | global | 150 | 1.24e-04 | 1.0000 | 0.00 |
| global_maximal | global | 150 | 5.79e-04 | 1.0000 | 0.00 |
| global_minimal | global | 300 | - | - | - |
| global_no_eke | global | 150 | 2.39e-04 | 1.0000 | 0.00 |
| global_no_neutral_diffusion | global | 150 | 1.59e-03 | 1.0000 | 0.00 |
| global_no_skew_diffusion | global | 150 | 1.73e-04 | 1.0000 | 0.00 |
| global_surface_pressure | global | 150 | 2.94e-05 | 1.0000 | 0.00 |


## per-variant detail

### acc_basic 
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_basic_error_evolution.png)
![acc_basic_temp_evolution](matrix_figures/acc_basic_temp_evolution.gif)
![acc_basic_psi_evolution](matrix_figures/acc_basic_psi_evolution.gif)

### acc_biharmonic_friction
overrides: `{'enable_hor_friction': False, 'enable_biharmonic_friction': True, 'A_hbi': 100000000000.0}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_biharmonic_friction_error_evolution.png)
![acc_biharmonic_friction_temp_evolution](matrix_figures/acc_biharmonic_friction_temp_evolution.gif)
![acc_biharmonic_friction_psi_evolution](matrix_figures/acc_biharmonic_friction_psi_evolution.gif)

### acc_biharmonic_mixing
overrides: `{'enable_biharmonic_mixing': True, 'K_hbi': 100000000000.0}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_biharmonic_mixing_error_evolution.png)
![acc_biharmonic_mixing_temp_evolution](matrix_figures/acc_biharmonic_mixing_temp_evolution.gif)
![acc_biharmonic_mixing_psi_evolution](matrix_figures/acc_biharmonic_mixing_psi_evolution.gif)

### acc_bottom_friction_var
overrides: `{'enable_bottom_friction_var': True}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_bottom_friction_var_error_evolution.png)
![acc_bottom_friction_var_temp_evolution](matrix_figures/acc_bottom_friction_var_temp_evolution.gif)
![acc_bottom_friction_var_psi_evolution](matrix_figures/acc_bottom_friction_var_psi_evolution.gif)

### acc_eke_isopycnal_diffusion_off
overrides: `{'enable_eke_isopycnal_diffusion': False}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_eke_isopycnal_diffusion_off_error_evolution.png)
![acc_eke_isopycnal_diffusion_off_temp_evolution](matrix_figures/acc_eke_isopycnal_diffusion_off_temp_evolution.gif)
![acc_eke_isopycnal_diffusion_off_psi_evolution](matrix_figures/acc_eke_isopycnal_diffusion_off_psi_evolution.gif)

### acc_eke_superbee_off
overrides: `{'enable_eke_superbee_advection': False}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_eke_superbee_off_error_evolution.png)
![acc_eke_superbee_off_temp_evolution](matrix_figures/acc_eke_superbee_off_temp_evolution.gif)
![acc_eke_superbee_off_psi_evolution](matrix_figures/acc_eke_superbee_off_psi_evolution.gif)

### acc_explicit_vert_friction
overrides: `{'enable_implicit_vert_friction': False, 'enable_explicit_vert_friction': True, 'enable_tke': False}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_explicit_vert_friction_error_evolution.png)
![acc_explicit_vert_friction_temp_evolution](matrix_figures/acc_explicit_vert_friction_temp_evolution.gif)
![acc_explicit_vert_friction_psi_evolution](matrix_figures/acc_explicit_vert_friction_psi_evolution.gif)

### acc_full
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_full_error_evolution.png)
![acc_full_temp_evolution](matrix_figures/acc_full_temp_evolution.gif)
![acc_full_psi_evolution](matrix_figures/acc_full_psi_evolution.gif)

### acc_hor_diffusion
overrides: `{'enable_hor_diffusion': True, 'K_h': 1000.0}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_hor_diffusion_error_evolution.png)
![acc_hor_diffusion_temp_evolution](matrix_figures/acc_hor_diffusion_temp_evolution.gif)
![acc_hor_diffusion_psi_evolution](matrix_figures/acc_hor_diffusion_psi_evolution.gif)

### acc_kappaH_profile_off
overrides: `{'enable_kappaH_profile': False}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_kappaH_profile_off_error_evolution.png)
![acc_kappaH_profile_off_temp_evolution](matrix_figures/acc_kappaH_profile_off_temp_evolution.gif)
![acc_kappaH_profile_off_psi_evolution](matrix_figures/acc_kappaH_profile_off_psi_evolution.gif)

### acc_maximal
overrides: `{'enable_biharmonic_friction': True, 'A_hbi': 100000000000.0, 'enable_noslip_lateral': True, 'enable_quadratic_bottom_friction': True, 'r_quad_bot': 0.001, 'enable_hor_diffusion': True, 'K_h': 1000.0, 'enable_biharmonic_mixing': True, 'K_hbi': 100000000000.0, 'enable_tke_superbee_advection': True}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_maximal_error_evolution.png)
![acc_maximal_temp_evolution](matrix_figures/acc_maximal_temp_evolution.gif)
![acc_maximal_psi_evolution](matrix_figures/acc_maximal_psi_evolution.gif)

### acc_minimal
overrides: `{'enable_hor_friction': False, 'enable_bottom_friction': False, 'enable_neutral_diffusion': False, 'enable_skew_diffusion': False, 'enable_tke': False}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_minimal_error_evolution.png)
![acc_minimal_temp_evolution](matrix_figures/acc_minimal_temp_evolution.gif)
![acc_minimal_psi_evolution](matrix_figures/acc_minimal_psi_evolution.gif)

### acc_no_hor_friction
overrides: `{'enable_hor_friction': False}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_no_hor_friction_error_evolution.png)
![acc_no_hor_friction_temp_evolution](matrix_figures/acc_no_hor_friction_temp_evolution.gif)
![acc_no_hor_friction_psi_evolution](matrix_figures/acc_no_hor_friction_psi_evolution.gif)

### acc_no_neutral_diffusion
overrides: `{'enable_neutral_diffusion': False, 'enable_skew_diffusion': False}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_no_neutral_diffusion_error_evolution.png)
![acc_no_neutral_diffusion_temp_evolution](matrix_figures/acc_no_neutral_diffusion_temp_evolution.gif)
![acc_no_neutral_diffusion_psi_evolution](matrix_figures/acc_no_neutral_diffusion_psi_evolution.gif)

### acc_no_skew_diffusion
overrides: `{'enable_skew_diffusion': False}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_no_skew_diffusion_error_evolution.png)
![acc_no_skew_diffusion_temp_evolution](matrix_figures/acc_no_skew_diffusion_temp_evolution.gif)
![acc_no_skew_diffusion_psi_evolution](matrix_figures/acc_no_skew_diffusion_psi_evolution.gif)

### acc_no_tke 
overrides: `{'enable_tke': False}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_no_tke_error_evolution.png)
![acc_no_tke_temp_evolution](matrix_figures/acc_no_tke_temp_evolution.gif)
![acc_no_tke_psi_evolution](matrix_figures/acc_no_tke_psi_evolution.gif)

### acc_noslip_lateral
overrides: `{'enable_noslip_lateral': True}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_noslip_lateral_error_evolution.png)
![acc_noslip_lateral_temp_evolution](matrix_figures/acc_noslip_lateral_temp_evolution.gif)
![acc_noslip_lateral_psi_evolution](matrix_figures/acc_noslip_lateral_psi_evolution.gif)

### acc_quadratic_bottom_friction 
overrides: `{'enable_bottom_friction': False, 'enable_quadratic_bottom_friction': True, 'r_quad_bot': 0.001}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_quadratic_bottom_friction_error_evolution.png)
![acc_quadratic_bottom_friction_temp_evolution](matrix_figures/acc_quadratic_bottom_friction_temp_evolution.gif)
![acc_quadratic_bottom_friction_psi_evolution](matrix_figures/acc_quadratic_bottom_friction_psi_evolution.gif)

### acc_ray_friction
overrides: `{'enable_ray_friction': True, 'r_ray': 1e-06}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_ray_friction_error_evolution.png)
![acc_ray_friction_temp_evolution](matrix_figures/acc_ray_friction_temp_evolution.gif)
![acc_ray_friction_psi_evolution](matrix_figures/acc_ray_friction_psi_evolution.gif)

### acc_surface_pressure
overrides: `{'enable_streamfunction': False}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_surface_pressure_error_evolution.png)
![acc_surface_pressure_temp_evolution](matrix_figures/acc_surface_pressure_temp_evolution.gif)
![acc_surface_pressure_psi_evolution](matrix_figures/acc_surface_pressure_psi_evolution.gif)

### acc_tke_superbee_advection
overrides: `{'enable_tke_superbee_advection': True}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/acc_tke_superbee_advection_error_evolution.png)
![acc_tke_superbee_advection_temp_evolution](matrix_figures/acc_tke_superbee_advection_temp_evolution.gif)
![acc_tke_superbee_advection_psi_evolution](matrix_figures/acc_tke_superbee_advection_psi_evolution.gif)

### global_biharmonic_friction
overrides: `{'enable_hor_friction': False, 'enable_biharmonic_friction': True, 'A_hbi': 1000000000000.0}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/global_biharmonic_friction_error_evolution.png)
![global_biharmonic_friction_temp_evolution](matrix_figures/global_biharmonic_friction_temp_evolution.gif)
![global_biharmonic_friction_psi_evolution](matrix_figures/global_biharmonic_friction_psi_evolution.gif)

### global_biharmonic_mixing
overrides: `{'enable_biharmonic_mixing': True, 'K_hbi': 1000000000000.0}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/global_biharmonic_mixing_error_evolution.png)
![global_biharmonic_mixing_temp_evolution](matrix_figures/global_biharmonic_mixing_temp_evolution.gif)
![global_biharmonic_mixing_psi_evolution](matrix_figures/global_biharmonic_mixing_psi_evolution.gif)

### global_default
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/global_default_error_evolution.png)
![global_default_temp_evolution](matrix_figures/global_default_temp_evolution.gif)
![global_default_psi_evolution](matrix_figures/global_default_psi_evolution.gif)

### global_hor_diffusion 
overrides: `{'enable_hor_diffusion': True, 'K_h': 1000.0}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/global_hor_diffusion_error_evolution.png)
![global_hor_diffusion_temp_evolution](matrix_figures/global_hor_diffusion_temp_evolution.gif)
![global_hor_diffusion_psi_evolution](matrix_figures/global_hor_diffusion_psi_evolution.gif)

### global_maximal
overrides: `{'enable_biharmonic_friction': True, 'A_hbi': 1000000000000.0, 'enable_noslip_lateral': True, 'enable_hor_diffusion': True, 'K_h': 1000.0, 'enable_biharmonic_mixing': True, 'K_hbi': 1000000000000.0, 'enable_tke_superbee_advection': True, 'enable_eke_superbee_advection': True}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/global_maximal_error_evolution.png)
![global_maximal_temp_evolution](matrix_figures/global_maximal_temp_evolution.gif)
![global_maximal_psi_evolution](matrix_figures/global_maximal_psi_evolution.gif)

### global_minimal 
overrides: `{'enable_hor_friction': False, 'enable_neutral_diffusion': False, 'enable_skew_diffusion': False, 'enable_eke': False}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/global_minimal_error_evolution.png)
![global_minimal_temp_evolution](matrix_figures/global_minimal_temp_evolution.gif)
![global_minimal_psi_evolution](matrix_figures/global_minimal_psi_evolution.gif)

### global_no_eke
overrides: `{'enable_eke': False}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/global_no_eke_error_evolution.png)
![global_no_eke_temp_evolution](matrix_figures/global_no_eke_temp_evolution.gif)
![global_no_eke_psi_evolution](matrix_figures/global_no_eke_psi_evolution.gif)

### global_no_neutral_diffusion
overrides: `{'enable_neutral_diffusion': False, 'enable_skew_diffusion': False}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/global_no_neutral_diffusion_error_evolution.png)
![global_no_neutral_diffusion_temp_evolution](matrix_figures/global_no_neutral_diffusion_temp_evolution.gif)
![global_no_neutral_diffusion_psi_evolution](matrix_figures/global_no_neutral_diffusion_psi_evolution.gif)

### global_no_skew_diffusion
overrides: `{'enable_skew_diffusion': False}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/global_no_skew_diffusion_error_evolution.png)
![global_no_skew_diffusion_temp_evolution](matrix_figures/global_no_skew_diffusion_temp_evolution.gif)
![global_no_skew_diffusion_psi_evolution](matrix_figures/global_no_skew_diffusion_psi_evolution.gif)

### global_surface_pressure
overrides: `{'enable_streamfunction': False}`
generated: `20260902T072723Z`, 10950 steps @ interval 150

![errors](matrix_figures/global_surface_pressure_error_evolution.png)
![global_surface_pressure_temp_evolution](matrix_figures/global_surface_pressure_temp_evolution.gif)
![global_surface_pressure_psi_evolution](matrix_figures/global_surface_pressure_psi_evolution.gif)

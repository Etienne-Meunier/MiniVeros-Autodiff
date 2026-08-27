#!/usr/bin/env python3
"""
Registry of the mini_veros vs veros comparison matrix (see ../program.md).

A "family" is a base geometry + baseline physics (one of the three setups
that already exist on both sides: acc_basic, acc/full, global_4deg). A
"variant" is a family plus a dict of settings overrides -- e.g. flipping one
enable_* flag on/off, or retuning a coefficient -- applied identically to
both mini_veros (StaticConfig/Parameters, via dataclasses.replace) and real
veros (VerosSetup's `override=` dict) by variant_util.build_*_variant.

This works because mini_veros's StaticConfig/Parameters field names were
deliberately chosen to match real veros's settings names 1:1 (see
mini_veros/model.py's docstrings) -- one overrides dict routes to both
without translation. Field names not present on either side raise in
variant_util._route_overrides / real veros's own override mechanism.

Flags intentionally never toggled here because mini_veros's port raises on
True (see mini_veros/model.py's StaticConfig comments): enable_TEM_friction,
enable_superbee_advection (tracer advection scheme, not the EKE/TKE
superbee flags), enable_eke_upwind_advection, enable_idemix,
enable_tke_hor_diffusion, enable_tke_upwind_advection.

Flags never toggled because they need source fields the setups don't
provide (enable_momentum_sources, enable_tempsalt_sources -- would be a
silent no-op either way, not worth a matrix row).
"""

FAMILIES = {
    "acc_basic": dict(
        mini_module="mini_veros.setups.acc.basic",
        real_module="veros.setups.acc_basic.acc_basic",
        real_class="ACCBasicSetup",
        group="acc",
    ),
    "acc_full": dict(
        mini_module="mini_veros.setups.acc.full",
        real_module="veros.setups.acc.acc",
        real_class="ACCSetup",
        group="acc",
    ),
    "global_default": dict(
        mini_module="mini_veros.setups.global_4deg.default",
        real_module="veros.setups.global_4deg.global_4deg",
        real_class="GlobalFourDegreeSetup",
        group="global",
    ),
}

# Coefficients used when a variant turns on a mixing/friction term whose
# baseline coefficient is 0 (a literal no-op otherwise). Same values go to
# both mini_veros and real veros via the shared overrides dict -- picked to
# be the right order of magnitude for these grids, not tuned for realism.
_A_HBI_ACC = 1e11
_A_HBI_GLOBAL = 1e12
_K_HBI_ACC = 1e11
_K_HBI_GLOBAL = 1e12
_K_H = 1000.0
_R_RAY = 1e-6
_R_QUAD_BOT = 1e-3

# `enable_tke` requires `enable_implicit_vert_friction` on real veros (see
# veros/settings.py's post-init check) -- so any variant that switches to
# explicit vertical friction turns tke off on both sides too, rather than
# tripping that check on the real-veros side only.
_EXPLICIT_VERT_FRICTION = dict(enable_implicit_vert_friction=False, enable_explicit_vert_friction=True, enable_tke=False)

VARIANTS = [
    # --- acc: channel geometry, 2deg / 15 levels -----------------------
    dict(name="acc_basic", family="acc_basic", overrides={}),
    dict(name="acc_full", family="acc_full", overrides={}),
    dict(name="acc_explicit_vert_friction", family="acc_basic", overrides=_EXPLICIT_VERT_FRICTION),
    dict(name="acc_no_hor_friction", family="acc_basic", overrides=dict(enable_hor_friction=False)),
    dict(
        name="acc_biharmonic_friction", family="acc_basic",
        overrides=dict(enable_hor_friction=False, enable_biharmonic_friction=True, A_hbi=_A_HBI_ACC),
    ),
    dict(name="acc_noslip_lateral", family="acc_basic", overrides=dict(enable_noslip_lateral=True)),
    dict(name="acc_ray_friction", family="acc_basic", overrides=dict(enable_ray_friction=True, r_ray=_R_RAY)),
    dict(name="acc_bottom_friction_var", family="acc_basic", overrides=dict(enable_bottom_friction_var=True)),
    dict(
        name="acc_quadratic_bottom_friction", family="acc_basic",
        overrides=dict(enable_bottom_friction=False, enable_quadratic_bottom_friction=True, r_quad_bot=_R_QUAD_BOT),
    ),
    dict(name="acc_surface_pressure", family="acc_basic", overrides=dict(enable_streamfunction=False)),
    dict(name="acc_hor_diffusion", family="acc_basic", overrides=dict(enable_hor_diffusion=True, K_h=_K_H)),
    dict(name="acc_biharmonic_mixing", family="acc_basic", overrides=dict(enable_biharmonic_mixing=True, K_hbi=_K_HBI_ACC)),
    dict(
        name="acc_no_neutral_diffusion", family="acc_basic",
        overrides=dict(enable_neutral_diffusion=False, enable_skew_diffusion=False),
    ),
    dict(name="acc_no_skew_diffusion", family="acc_basic", overrides=dict(enable_skew_diffusion=False)),
    dict(name="acc_no_tke", family="acc_basic", overrides=dict(enable_tke=False)),
    dict(name="acc_tke_superbee_advection", family="acc_basic", overrides=dict(enable_tke_superbee_advection=True)),
    dict(name="acc_eke_isopycnal_diffusion_off", family="acc_full", overrides=dict(enable_eke_isopycnal_diffusion=False)),
    dict(name="acc_eke_superbee_off", family="acc_full", overrides=dict(enable_eke_superbee_advection=False)),
    dict(name="acc_kappaH_profile_off", family="acc_basic", overrides=dict(enable_kappaH_profile=False)),
    dict(
        name="acc_minimal", family="acc_basic",
        overrides=dict(
            enable_hor_friction=False, enable_bottom_friction=False,
            enable_neutral_diffusion=False, enable_skew_diffusion=False, enable_tke=False,
        ),
    ),
    dict(
        name="acc_maximal", family="acc_full",
        overrides=dict(
            enable_biharmonic_friction=True, A_hbi=_A_HBI_ACC,
            enable_noslip_lateral=True,
            enable_quadratic_bottom_friction=True, r_quad_bot=_R_QUAD_BOT,
            enable_hor_diffusion=True, K_h=_K_H,
            enable_biharmonic_mixing=True, K_hbi=_K_HBI_ACC,
            enable_tke_superbee_advection=True,
        ),
    ),
    # --- global_4deg: real bathymetry, 4deg / 15 levels -----------------
    # No enable_tke=False variant here (unlike acc): real veros's own
    # veros/setups/global_4deg/global_4deg.py reads vs.forc_tke_surface
    # unconditionally in set_forcing_kernel, with no enable_tke guard
    # (the acc/acc_basic setups do guard it) -- so enable_tke=False raises
    # "Variable forc_tke_surface is not active in this configuration"
    # against the real reference setup itself, before mini_veros even
    # enters the picture. Upstream veros example-setup limitation, not a
    # mini_veros port gap or a matrix-harness issue.
    dict(name="global_default", family="global_default", overrides={}),
    dict(name="global_no_eke", family="global_default", overrides=dict(enable_eke=False)),
    dict(
        name="global_biharmonic_friction", family="global_default",
        overrides=dict(enable_hor_friction=False, enable_biharmonic_friction=True, A_hbi=_A_HBI_GLOBAL),
    ),
    dict(name="global_surface_pressure", family="global_default", overrides=dict(enable_streamfunction=False)),
    dict(name="global_hor_diffusion", family="global_default", overrides=dict(enable_hor_diffusion=True, K_h=_K_H)),
    dict(
        name="global_biharmonic_mixing", family="global_default",
        overrides=dict(enable_biharmonic_mixing=True, K_hbi=_K_HBI_GLOBAL),
    ),
    dict(
        name="global_no_neutral_diffusion", family="global_default",
        overrides=dict(enable_neutral_diffusion=False, enable_skew_diffusion=False),
    ),
    dict(name="global_no_skew_diffusion", family="global_default", overrides=dict(enable_skew_diffusion=False)),
    dict(
        name="global_minimal", family="global_default",
        overrides=dict(
            enable_hor_friction=False, enable_neutral_diffusion=False,
            enable_skew_diffusion=False, enable_eke=False,
        ),
    ),
    dict(
        name="global_maximal", family="global_default",
        overrides=dict(
            enable_biharmonic_friction=True, A_hbi=_A_HBI_GLOBAL,
            enable_noslip_lateral=True,
            enable_hor_diffusion=True, K_h=_K_H,
            enable_biharmonic_mixing=True, K_hbi=_K_HBI_GLOBAL,
            enable_tke_superbee_advection=True, enable_eke_superbee_advection=True,
        ),
    ),
]

VARIANTS_BY_NAME = {v["name"]: v for v in VARIANTS}
assert len(VARIANTS_BY_NAME) == len(VARIANTS), "duplicate variant name in VARIANTS"

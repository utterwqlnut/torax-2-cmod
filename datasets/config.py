import numpy as np

dilution_factor = 0.9
Zeff = 1.5
Zimp = (Zeff - dilution_factor) / (1 - dilution_factor)

dilution_factor = 0.9
Zeff = 1.5
Zimp = (Zeff - dilution_factor) / (1 - dilution_factor)

CONFIG = {
    'plasma_composition': {
        'A_i_override': 2.0,   
        'Z_eff': Zeff,
        'Z_impurity_override': Zimp, 
    },
    'profile_conditions': {
        'Ip': 0,
        'T_i': 0,
        'T_e': 0,
        'n_e': 0,  # Initial electron density profile
        'nbar': 0,
    },
    'numerics': {
        't_final': 0,
        't_initial': 0,
        'fixed_dt': 0.02,
        'evolve_ion_heat': True,
        'evolve_electron_heat': True,
        'evolve_current': True,
        'evolve_density': True,
        'dt_reduction_factor': 3,
        'adaptive_T_source_prefactor': 1.0e10,
        'adaptive_n_source_prefactor': 1.0e8,
    },
    'geometry': { # very important need actual geometry
        'geometry_type': 'eqdsk',
        'geometry_file': 'test.eqdsk',
        'n_surfaces': 50,
    },
    'sources': {
        'generic_current': {
            'fraction_of_total_current': 0.15,
            'gaussian_width': 0.075,
            'gaussian_location': 0.36,
        },
        'pellet': {
            'S_total': 0.0e22,
            'pellet_width': 0.1,
            'pellet_deposition_location': 0.85,
        },
        'generic_heat': {
            'gaussian_location': 0.12741589640723575,
            'gaussian_width': 0.07280908366127758,
            # total heating (with a rough assumption of radiation reduction)
            'P_total': 0,
            'electron_heat_fraction': 1.0,
        },
        'fusion': {},
        'ei_exchange': {},
    },
    'transport': {
        'model_name': 'qlknn',
        'apply_inner_patch': True,
        'chi_i_inner': 0.2,
        'chi_e_inner': 0.2,
        'rho_inner': 0.2,
        'apply_outer_patch': True,
        'chi_i_outer': 0.2,
        'chi_e_outer': 0.2,
        'rho_outer': 0.95,
        'chi_min': 0.2,
        'chi_max': 100.0,
        'D_e_min': 0.05,
        'D_e_max': 100.0,
        'V_e_min': -50.0,
        'V_e_max': 50.0,
        'smoothing_width': 0.1*np.sqrt(np.log(2))/2,
        'smooth_everywhere': True,
        'include_ITG': True,
        'include_TEM': True,
        'include_ETG': True,
        'DV_effective': True,
        'An_min': 0.05,
        'avoid_big_negative_s': True,
        'smag_alpha_correction': True,
        'q_sawtooth_proxy': True,
        'ITG_flux_ratio_correction': 1.0,
        'ETG_correction_factor': 1.0,
    },
    # Pedestal is assumed to be same as edge profile
    'pedestal': {
        'model_name': 'set_T_ped_n_ped',
        'set_pedestal': True,
        'T_i_ped': 0,
        'T_e_ped': 0,
        'n_e_ped': 0,
        'rho_norm_ped_top': 0.95,
    },
    'solver': {
        'solver_type': 'newton_raphson',
        'use_predictor_corrector': True,
        'n_corrector_steps': 10,
        'chi_pereverzev': 30,
        'D_pereverzev': 15,
        'use_pereverzev': True,
        'log_iterations': False,
    },
    'time_step_calculator': {
        'calculator_type': 'fixed',
    },
}
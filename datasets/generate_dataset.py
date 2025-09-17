import pandas as pd
from tqdm import tqdm
import eqtools
from eqtools.CModEFIT import CModEFITTree
from eqdsk import EQDSKInterface
import torax
import xarray as xr
import numpy as np
from scipy.optimize import least_squares

from config import CONFIG
from pull_shot import ShotPuller

all_data = []

df = pd.read_csv('all_shots.csv')
shots = df['SHOT'].tolist()

puller = ShotPuller()

for shot in tqdm(shots):
    try:
        shot_data = puller.pull_shot_full(shot)
        
        eq = CModEFITTree(shot)
        eq.gfile(time=eq.getTimeBase()[0],name='torax/data/third_party/geo/test.eqdsk',shot=shot,nh=100,nw=100,nbbbs=100)

        eqdsk_read = EQDSKInterface.from_file('torax/data/third_party/geo/test.eqdsk', no_cocos=True)
        eqdsk_read.identify(as_cocos=8)
        eqdsk_read = eqdsk_read.to_cocos(2)

        eqdsk_read.write("torax/data/third_party/geo/test.eqdsk",file_format='eqdsk')

        for column in shot_data.data_vars:
            if column[-7:] == 'profile' or column == 'commit_hash':
                continue
            print(column)
            shot_data[column] = shot_data[column].interpolate_na(dim="index",method='linear').bfill(dim="index")
        print('Here')
        # Get rhos array
        rhos = np.linspace(0,1,len(shot_data['ne_profile'][0]))

        times = shot_data['time'].values.astype(float)
        rhos = np.linspace(0, 1, shot_data['ne_profile'].shape[1])

        ip_vals = np.abs(shot_data['ip'].values)
        bt_vals = np.abs(shot_data['bt'].values)
        n_e_vals = np.abs(shot_data['n_e'].values)
        p_input_vals = np.abs(shot_data['p_input'].values)
        te_arr = shot_data['te_profile'].values/(1000*11600) # Convert from eV to keV
        ne_arr = shot_data['ne_profile'].values

        # Build dictionaries in one line each
        ip = dict(zip(times, ip_vals))
        bt = dict(zip(times, bt_vals))
        nbar = dict(zip(times, n_e_vals))
        ptot = dict(zip(times, p_input_vals))

        te = {t: dict(zip(rhos, row)) for t, row in zip(times, te_arr)}
        ti = te.copy()
        ne = {t: dict(zip(rhos, row)) for t, row in zip(times, ne_arr)}

        CONFIG["profile_conditions"]["Ip"] = ip
        CONFIG["profile_conditions"]["T_i"] = ti
        CONFIG["profile_conditions"]["T_e"] = te

        # Sometimes ne profile data is majority missing/NaNs in which case it has unphysical values
        if ne[times[0]][rhos[0]]<1e10:
            CONFIG["profile_conditions"].pop("n_e",None)
        else:
            CONFIG["profile_conditions"]["n_e"] = ne

        CONFIG["profile_conditions"]["nbar"] = nbar
        CONFIG["numerics"]["t_initial"] = shot_data['time'][0]
        CONFIG["numerics"]["t_final"] = shot_data['time'][len(shot_data['time'])-1]

        # Ensure all keys are floats, values are floats
        CONFIG["sources"]["generic_heat"]["P_total"] = ptot

        CONFIG["pedestal"]["T_i_ped"] = {float(t): float(val[rhos[-2]]) for t, val in ti.items()}
        CONFIG["pedestal"]["T_e_ped"] = {float(t): float(val[rhos[-2]]) for t, val in te.items()}
        CONFIG["pedestal"]["n_e_ped"] = {float(t): float(val[rhos[-2]]) for t, val in ne.items()}

        

        config = torax.ToraxConfig.from_dict(CONFIG)

        data_tree, state_history = torax.run_simulation(config)

        # Check that the simulation completed successfully.
        if state_history.sim_error != torax.SimError.NO_ERROR:
            raise ValueError(
                f'TORAX failed to run the simulation with error: {state_history.sim_error}.'
            )

        column_conversion = {
            'beta_N':'beta_n',
            'beta_pol':'beta_p',
            'fgw_n_e_volume_avg': 'greenwald_fraction',
            'li3': 'li',
            'q95': 'q95',
            'v_loop_lcfs': 'v_loop',
            #'elongation': 'kappa', elongation is essentially an input doesnt get evolved
            # 'upper_gap/lower_gap' may be able to derive will have to check
            # So all diagnostics in disruption prediction varaibles except ip_error/ip (ip is given as input), upper_gap/lower_gap (may be able to derive), n_equal_1_mode (again may be able to derive from outputs) 
        }
        simulated_data = {}
        for column in column_conversion.keys():
            xarray_data = np.interp(shot_data['time'],data_tree.scalars.time,data_tree.scalars[column].to_numpy())
            simulated_data["sim "+column_conversion[column]] = xarray_data
        
        # TODO get profiles too
        df = pd.concat((shot_data.drop_vars(["te_profile","ne_profile"]).to_dataframe(),pd.DataFrame(simulated_data)),axis=1)
        all_data.append(df)

    except Exception as e:
        print('Invalid Shot '+str(shot)+' '+str(e))

pd.concat(all_data).to_csv('dispy-torax-data.csv')

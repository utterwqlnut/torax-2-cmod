from disruption_py.settings.retrieval_settings import RetrievalSettings
from disruption_py.workflow import get_shots_data
from disruption_py.machine.cmod.physics import CmodPhysicsMethods
from disruption_py.core.physics_method.params import PhysicsMethodParams
from disruption_py.core.retrieval_manager import RetrievalManager
from disruption_py.workflow import _get_database_instance, _get_mds_instance, resolve_tokamak_from_environment
from disruption_py.core.physics_method.errors import CalculationError
from disruption_py.machine.cmod.thomson import CmodThomsonDensityMeasure
from disruption_py.core.utils.math import (
    gaussian_fit,
    gauss,
    gaussian_fit_with_fixed_mean,
    interp1,
    smooth,
)
import xarray as xr
from disruption_py.machine.cmod.efit import CmodEfitMethods
from MDSplus import mdsExceptions
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


class ShotPuller:
    
    def __init__(self,rho_resolution=25):
        self.retrieval_settings = RetrievalSettings(
            # Use the efit timebase when returning data
            time_setting="efit",
        )
        tokamak = resolve_tokamak_from_environment()
        database = _get_database_instance(tokamak, None)
        mds_conn = _get_mds_instance(tokamak, None)

        self.retrieval_manager = RetrievalManager(
            tokamak=tokamak,
            process_database=database,
            process_mds_conn=mds_conn,
        )
        
        self.rho_resolution = rho_resolution

    def safe_gaussian_fit(xdata,ydata):
        i = ydata.argmax()
        guess = [ydata[i], xdata[i], (xdata.max() - xdata.min()) / 3]
        try:
            pother, pmean, psigma = gaussian_fit(
                xdata, ydata, guess
            )
        except RuntimeError as exc:
            return None, None, None

        if pmean < -0.35 or pmean > 0.35:
            return None, None, None

        return pother, pmean, psigma
    
    def fill_nans_interp(data):
        nans = np.isnan(data)
        x = np.arange(len(data))
        data[nans] = np.interp(x[nans], x[~nans], data[~nans])

        return data
    
    def pull_shot_basic(self, shot_number):
        shot_data = get_shots_data(
            tokamak="cmod",
            # Retrieve data for the desired shots
            shotlist_setting=[shot_number],
            # Use the created retrieval_settings
            retrieval_settings=self.retrieval_settings,
            # Automatically stream retrieved data to a csv file by passing in a file path ending in .csv
            output_setting="dataframe",
            # Use a single process to retrieve the data
            num_processes=1,
        )

        return shot_data


    def pull_shot_ts_extra(self, shot_number, n_e):
        params = self.retrieval_manager.shot_setup(shot_number, self.retrieval_settings)

        # Check if shot is blacklisted
        if CmodPhysicsMethods.is_on_blacklist(params.shot_id):
            raise CalculationError("Shot is on blacklist")
        
        # Fetch data
        # Get EFIT geometry data
        z0 = params.mds_conn.get_data(
            r"\efit_aeqdsk:zmagx/100", tree_name="_efit_tree"
        )  # [m]
        kappa = params.mds_conn.get_data(
            r"\efit_aeqdsk:kappa", tree_name="_efit_tree"
        )  # [dimensionless]
        aminor, efit_time = params.mds_conn.get_data_with_dims(
            r"\efit_aeqdsk:aout/100", tree_name="_efit_tree"
        )  # [m], [s]
        bminor = aminor * kappa

        # Get Te data and TS time basis
        node_ext = ".yag_new.results.profiles"
        ts_te_core, ts_time = params.mds_conn.get_data_with_dims(
            f"{node_ext}:te_rz", tree_name="electrons"
        )  # [keV], [s]
        ts_te_core = ts_te_core * 1000  # [keV] -> [eV]
        ts_te_edge = params.mds_conn.get_data(r"\ts_te")  # [eV]
        ts_te = np.concatenate((ts_te_core, ts_te_edge)) * 11600  # [eV] -> [K]

        # Get ne data
        ts_ne_core = params.mds_conn.get_data(
            f"{node_ext}:ne_rz", tree_name="electrons"
        )  # [m^-3]
        ts_ne_edge = params.mds_conn.get_data(r"\ts_ne")  # [m^-3]
        ts_ne = np.concatenate((ts_ne_core, ts_ne_edge))

        # Get TS chord positions
        ts_z_core = params.mds_conn.get_data(
            f"{node_ext}:z_sorted", tree_name="electrons"
        )  # [m]
        ts_z_edge = params.mds_conn.get_data(r"\fiber_z", tree_name="electrons")  # [m]
        ts_z = np.concatenate((ts_z_core, ts_z_edge))

        # Make sure that there are equal numbers of edge position and edge temperature points
        if len(ts_z_edge) != ts_te_edge.shape[0]:
            raise CalculationError(
                "TS edge data and z positions are not the same length for shot"
            )
        times = params.times
        ts_pressure = ts_te * ts_ne * 1.38e-23

        # Interpolate EFIT signals to TS time basis
        bminor = interp1(efit_time, bminor, ts_time)
        z0 = interp1(efit_time, z0, ts_time)

        # Calculate Te, ne, & pressure peaking factors
        te_profile = np.full((len(ts_time), self.rho_resolution), np.nan)
        ne_profile = np.full((len(ts_time), self.rho_resolution), np.nan)


        (itimes,) = np.where((ts_time > 0) & (ts_time < times[-1]))
    
        for itime in itimes:
            ts_te_arr = ts_te[:, itime]
            ts_ne_arr = ts_ne[:, itime]
            ts_pressure_arr = ts_pressure[:, itime]
            # This gives identical results using either ts_te_arr or ts_ne_arr
            (indx,) = np.where(ts_ne_arr > 0)
            if len(indx) < 10:
                continue
            ts_te_arr = ts_te_arr[indx]
            ts_ne_arr = ts_ne_arr[indx]
            ts_pressure_arr = ts_pressure_arr[indx]
            ts_z_arr = ts_z[indx]
            sorted_indx = np.argsort(ts_z_arr)
            ts_z_arr = ts_z_arr[sorted_indx]
            ts_te_arr = ts_te_arr[sorted_indx]
            
            ts_ne_arr = ts_ne_arr[sorted_indx]
            ts_pressure_arr = ts_pressure_arr[sorted_indx]
            # Create equal-spacing array of ts_z_arr and interpolate TS profile on it
            # Skip if there's no EFIT zmagx data

            if np.isnan(z0[itime]):
                continue

            z_arr_equal_spacing = np.linspace(z0[itime], ts_z_arr[-1], len(ts_z_arr))
            te_arr_equal_spacing = interp1(ts_z_arr, ts_te_arr, z_arr_equal_spacing)
            ne_arr_equal_spacing = interp1(ts_z_arr, ts_ne_arr, z_arr_equal_spacing)

            pressure_arr_equal_spacing = interp1(
                ts_z_arr, ts_pressure_arr, z_arr_equal_spacing
            )
            # Fit gaussian to profiles
            adjusted_z_rho = (z_arr_equal_spacing - z0[itime])/(max(abs(z_arr_equal_spacing - z0[itime])))

            coeffs_te = ShotPuller.safe_gaussian_fit(adjusted_z_rho,te_arr_equal_spacing)
            coeffs_ne = ShotPuller.safe_gaussian_fit(adjusted_z_rho,ne_arr_equal_spacing)

            if coeffs_te[0] == None or coeffs_ne[0] == None:
                continue
            
            rhos_new = np.linspace(0,1,self.rho_resolution)
            te_profile[itime] = gauss(rhos_new,*coeffs_te)
            ne_profile[itime] = gauss(rhos_new,*coeffs_ne)


        ne_new_profile = np.full((len(times), self.rho_resolution), np.nan)
        te_new_profile = np.full((len(times), self.rho_resolution), np.nan)
        
        for rho_idx in range(ne_profile.shape[1]):
            # Set unphysical values to NaN and then interpolate over them
            ne_profile[:,rho_idx] = np.where(ne_profile[:,rho_idx]<1e10,np.nan,ne_profile[:,rho_idx])
            ne_profile[:,rho_idx] = ShotPuller.fill_nans_interp(ne_profile[:,rho_idx])
            te_profile[:,rho_idx] = ShotPuller.fill_nans_interp(te_profile[:,rho_idx])
            # Interpolate to new time basis
            ne_new_profile[:,rho_idx] = interp1(ts_time, ne_profile[:,rho_idx], times, "linear")
            te_new_profile[:,rho_idx] = interp1(ts_time, te_profile[:,rho_idx], times, "linear")

        ne_new_profile = ne_new_profile/ne_new_profile.mean(axis=1)[:, np.newaxis] * np.abs(n_e[:, np.newaxis]) # Normalize to line averaged density
        return ne_new_profile, te_new_profile
        

    def pull_shot_full(self,shot_number):
        # Pull disruption py data
        xr_full = self.pull_shot_basic(shot_number).to_xarray()

        try:
            # Clear unphysical n_e values
            xr_full['n_e'] = xr_full['n_e'].where(xr_full['n_e']>=1e10).interpolate_na(dim='index', method='linear', fill_value="extrapolate")  

            # Pull TS profiles
            ne_profile, te_profile = self.pull_shot_ts_extra(shot_number, np.asarray(xr_full['n_e']))

            xr_full["ne_profile"] = (("time_idx","profile"),ne_profile)
            xr_full["te_profile"] = (("time_idx","profile"),te_profile)

        except Exception as e:
            print(e)
            pass 
        
        return xr_full
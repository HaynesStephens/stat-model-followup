import xarray as xr
import numpy as np
import pandas as pd

##########################
###### FUNCTIONS #########
##########################

def loadWeatherDaily(climate_model, climate_name, clim_var, coords=[(48.75, 36.25), (-103.75, -80.25)], ssp='ssp585'):
    clim_dir = '/project2/ggcmi/AgMIP.input/phase3/ISIMIP3/climate_land_only_v2/climate3b'
    years = ['1971_1980',
             '1981_1990',
             '1991_2000',
             '2001_2010',
             '2011_2014',
             '2015_2020',
             '2021_2030',
             '2031_2040',
             '2041_2050',
             '2051_2060',
             '2061_2070',
             '2071_2080',
             '2081_2090',
             '2091_2100']
    scen = ['historical']*5 + [ssp]*9
    scen = dict(zip(years, scen))

    file_names = ['{0}/{1}/{2}/{3}_{1}_{4}_global_daily_{5}.nc'.format(clim_dir,
                                                                       scen[y],
                                                                       climate_model,
                                                                       climate_name,
                                                                       clim_var,
                                                                       y) for y in years]
    if coords != None:
        lat_N, lat_S = coords[0]
        lon_W, lon_E = coords[1]
        da = xr.concat(
            [xr.open_dataarray(fname).sel(lat=slice(lat_N, lat_S), lon=slice(lon_W, lon_E)) for fname in file_names],
            dim='time')
    else:
        da = xr.concat([xr.open_dataarray(fname) for fname in file_names], dim='time')
    return da

def loadPBM(climate_name, crop_model, output_var, coords=[(48.75, 36.25), (-103.75, -80.25)]):
    short_name = climate_name.split('_')[0]
    crop_lower = crop_model.lower()
    output_dir = '/project2/ggcmi/AgMIP.output/{0}/phase3b'.format(crop_model)
    lat_N, lat_S = coords[0]
    lon_W, lon_E = coords[1]

    da_his = '{0}/{1}/historical/mai/{2}_{1}_w5e5_historical_2015soc_default_{3}-mai-noirr_global_annual_1850_2014.nc'.format(output_dir, short_name, crop_lower, output_var)
    da_his = xr.open_mfdataset(da_his, decode_times=False).sel(lat=slice(lat_N, lat_S), lon=slice(lon_W, lon_E))
    da_his['time'] = pd.date_range('1850', '2015', freq='1A')

    da_fut = '{0}/{1}/ssp585/mai/{2}_{1}_w5e5_ssp585_2015soc_2015co2_{3}-mai-noirr_global_annual_2015_2100.nc'.format(output_dir, short_name, crop_lower, output_var)
    da_fut = xr.open_mfdataset(da_fut, decode_times=False).sel(lat=slice(lat_N, lat_S), lon=slice(lon_W, lon_E))
    da_fut['time'] = pd.date_range('2015', '2101', freq='1A')

    da_out = xr.merge([da_his, da_fut])
    da_out = da_out.sel(time=slice('1981', '2100'))
    return da_out

def loadSoil():
    soils = '/project2/ggcmi/AgMIP.input/phase3/ISIMIP3/soil/HWSD_soil_data_on_cropland_v2.3.nc'
    soils = xr.open_dataset(soils).sel(lat=slice(48.75, 36.25), lon=slice(-103.8, -80.75))
    soilvars = ['texture_class', 'soil_ph', 'soil_caco3']#, 'bulk_density', 'cec_soil', 'oc', 'awc', 'sand', 'silt', 'clay', 'gravel', 'ece', 'bs_soil',
            #'issoil', 'root_obstacles', 'impermeable_layer', 'mu_global', 'lon', 'lat']
    soils = soils[soilvars]
    return soils

if __name__=='__main__':

    clim_models = ['GFDL-ESM4', 'IPSL-CM6A-LR', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'UKESM1-0-LL']
    clim_names = ['gfdl-esm4_r1i1p1f1_w5e5',
                'ipsl-cm6a-lr_r1i1p1f1_w5e5',
                'mpi-esm1-2-hr_r1i1p1f1_w5e5',
                'mri-esm2-0_r1i1p1f1_w5e5',
                'ukesm1-0-ll_r1i1p1f2_w5e5']
    clim_names = dict(zip(clim_models, clim_names))

    crop_models = crop_models = ['ACEA', 'CROVER', 'CYGMA1p74', 'DSSAT-Pythia', 
                                'EPIC-IIASA', 'ISAM', 'LDNDC', 'LPJmL',
                                'pDSSAT', 'PEPIC', 'PROMET', 'SIMPLACE-LINTUL5']
    output_vars = ['yield', 'plantday']

    soils = loadSoil()
    tas = loadWeatherDaily(clim_models[0], clim_names[clim_models[0]], 'tas', coords=[(48.75, 36.25), (-103.75, -80.25)], ssp='ssp585')
    yields = loadPBM(clim_names[clim_models[0]], crop_models[7], output_vars[0], coords=[(48.75, 36.25), (-103.75, -80.25)])

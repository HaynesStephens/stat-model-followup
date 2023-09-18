import xarray as xr
import numpy as np
import pandas as pd

##########################
###### FUNCTIONS #########
##########################

def loadDailyData(model, name, clim_var, coords=[(48.75, 36.25), (-103.75, -80.25)]):
    clim_dir = '/project2/ggcmi/AgMIP.input/phase3/ISIMIP3/climate_land_only/climate3b'
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
    scen = ['historical',
            'historical',
            'historical',
            'historical',
            'historical',
            'ssp585',
            'ssp585',
            'ssp585',
            'ssp585',
            'ssp585',
            'ssp585',
            'ssp585',
            'ssp585',
            'ssp585']
    scen = dict(zip(years, scen))

    file_names = ['{0}/{1}/{2}/{3}_{1}_{4}_global_daily_{5}.nc'.format(clim_dir,
                                                                       scen[y],
                                                                       model,
                                                                       name,
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

def compilePhase3(climate_model, crop_model, output_var, coords=[(48.75, 36.25), (-103.75, -80.25)]):
    short_name = climate_model.split('_')[0]
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

    savedir = '/project2/moyer/ag_data/ggcmi_phase3/new_ssps'
    da_out.to_netcdf('{0}/{1}_{2}_w5e5_ssp585_2015soc_2015co2_{3}-mai-noirr_global_annual_1981_2100.nc'.format(savedir, crop_lower, short_name, output_var))
    return True

##########################
###### EXECUTION #########
##########################

clim_models = ['GFDL-ESM4', 'IPSL-CM6A-LR', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'UKESM1-0-LL']
clim_names = ['gfdl-esm4_r1i1p1f1_w5e5',
              'ipsl-cm6a-lr_r1i1p1f1_w5e5',
              'mpi-esm1-2-hr_r1i1p1f1_w5e5',
              'mri-esm2-0_r1i1p1f1_w5e5',
              'ukesm1-0-ll_r1i1p1f2_w5e5']
clim_names = dict(zip(clim_models, clim_names))

def compile_new_GGCMI_phase3(): # Ran on 23.09.18
    output_vars = ['yield']
    crop_models = ['PEPIC', 'pDSSAT', 'LPJmL']
    for crop_model in crop_models:
        print('CROP MODEL: {0}'.format(crop_model))
        for output_var in output_vars:
            print('VARIABLE: {0}'.format(output_var))
            for model in clim_models:
                name = clim_names[model]
                print('MODEL: {0}'.format(model))
                try:
                    output_loaded = compilePhase3(name, crop_model, output_var)
                    print('COMPLETE: {0}'.format(output_loaded))
                except:
                    print('NOT FOUND')
                    continue
            print('\n')


#####################
###### MAIN #########
#####################
if __name__ == '__main__':
    getBinsTrueSeason()


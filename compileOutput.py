import xarray as xr
import numpy as np
import pandas as pd

clim_models = ['GFDL-ESM4', 'IPSL-CM6A-LR', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'UKESM1-0-LL']
clim_names = ['gfdl-esm4_r1i1p1f1_w5e5',
              'ipsl-cm6a-lr_r1i1p1f1_w5e5',
              'mpi-esm1-2-hr_r1i1p1f1_w5e5',
              'mri-esm2-0_r1i1p1f1_w5e5',
              'ukesm1-0-ll_r1i1p1f2_w5e5']
clim_names = dict(zip(clim_models, clim_names))

##########################
###### FUNCTIONS #########
##########################

def compileOutput(name, crop_model, output_var, coords=[(48.75, 36.25), (-103.75, -80.25)]):
    short_name = name.split('_')[0]
    crop_lower = crop_model.lower()
    output_dir = '/project2/ggcmi/AgMIP.output/{0}/phase3b'.format(crop_model)
    years = ['1850_2014',
             '2015_2100']
    scen = ['historical',
            'ssp585']
    scen = dict(zip(years, scen))
    file_names = ['{0}/{1}/{2}/mai/{3}_{1}_w5e5_{2}_2015soc_2015co2_{4}-mai-noirr_global_annual_{5}.nc'.format(output_dir,
                                                                                                               short_name,
                                                                                                               scen[y],
                                                                                                               crop_lower,
                                                                                                               output_var,
                                                                                                               y) for y in years]
    if coords != None:
        lat_N, lat_S = coords[0]
        lon_W, lon_E = coords[1]
        da = xr.concat(
            [xr.open_dataarray(fname, decode_times=False).sel(lat=slice(lat_N, lat_S), lon=slice(lon_W, lon_E)) for fname in file_names],
            dim='time')
    else:
        da = xr.concat([xr.open_dataarray(fname, decode_times=False) for fname in file_names], dim='time')
    da['time'] = pd.date_range('1850', '2101', freq='1A')
    da = da.sel(time=slice('1971', '2100'))
    savedir = '/project2/geos39650/ag_data/ggcmi_phase3'
    da.to_netcdf('{0}/{1}_{2}_w5e5_ssp585_2015soc_2015co2_{3}-mai-noirr_global_annual_1971_2100.nc'.format(savedir,
                                                                                                           crop_lower,
                                                                                                           short_name,
                                                                                                           output_var))
    return True

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


def loadPR(model, name):
    pr = loadDailyData(model, name, 'pr')
    save_dir = '/project2/geos39650/ag_data/climate_projections/phase3'
    pr.to_netcdf('{0}/{1}_{2}_{3}_MDW_daily_1971_2100.nc'.format(save_dir, name, 'ssp585', 'pr'))
    return True


def loadGDDandHDD(model, name):
    tasmin = loadDailyData(model, name, 'tasmin')
    tasmax = loadDailyData(model, name, 'tasmax')

    gdd = xr.DataArray(np.zeros(tasmin.shape),
                       coords={'time': tasmin.time, 'lat': tasmin.lat, 'lon': tasmin.lon},
                       dims=['time', 'lat', 'lon'])
    hdd = xr.DataArray(np.zeros(tasmin.shape),
                       coords={'time': tasmin.time, 'lat': tasmin.lat, 'lon': tasmin.lon},
                       dims=['time', 'lat', 'lon'])

    for y in np.sort(np.unique(tasmin['time.year'].values)):
        print(y)
        tasmini = tasmin.sel(time=str(y))
        tasmaxi = tasmax.sel(time=str(y))

        dt, dx, dy = tasmini.shape

        tasmini_vals = np.ma.masked_array(np.reshape(np.repeat(tasmini.values, 24, axis=np.newaxis), (dt, dx, dy, 24)))
        tasmini_vals = np.ma.masked_where(np.isnan(tasmini_vals), tasmini_vals)
        tasmini_vals = np.ma.harden_mask(tasmini_vals)

        tasmaxi_vals = np.ma.masked_array(np.reshape(np.repeat(tasmaxi.values, 24, axis=np.newaxis), (dt, dx, dy, 24)))
        tasmaxi_vals = np.ma.masked_where(np.isnan(tasmaxi_vals), tasmaxi_vals)
        tasmaxi_vals = np.ma.harden_mask(tasmaxi_vals)

        hrs = np.reshape(np.tile(np.arange(24), (dt, dx, dy)), (dt, dx, dy, 24))

        cos_hrs = np.cos(hrs * np.pi / 12)
        amplitude = (-1) * (tasmaxi_vals - tasmini_vals) / 2
        offset = (tasmaxi_vals + tasmini_vals) / 2
        t_hrs = (amplitude * cos_hrs) + offset
        t_hrs = t_hrs - 273.15
        gdd_hrs = t_hrs.copy()
        hdd_hrs = t_hrs.copy()

        t_high = 29
        t_low = 10
        ## GDD CALCULATION
        gdd_hrs[gdd_hrs > t_high] = t_high
        gdd_hrs = gdd_hrs - t_low
        gdd_hrs[gdd_hrs < 0] = 0
        gdd.loc[dict(time=tasmini.time)] = np.sum(gdd_hrs * (1 / 24), axis=3)

        ## HDD CALCULATION
        hdd_hrs = hdd_hrs - t_high
        hdd_hrs[hdd_hrs < 0] = 0
        hdd.loc[dict(time=tasmini.time)] = np.sum(hdd_hrs * (1 / 24), axis=3)

    save_dir = '/project2/geos39650/ag_data/climate_projections/phase3'
    gdd.to_netcdf('{0}/{1}_{2}_{3}_MDW_daily_1971_2100.nc'.format(save_dir, name, 'ssp585', 'gdd'))
    hdd.to_netcdf('{0}/{1}_{2}_{3}_MDW_daily_1971_2100.nc'.format(save_dir, name, 'ssp585', 'hdd'))
    return True


def loadBins(model, name):
    tasmin = loadDailyData(model, name, 'tasmin')
    tasmax = loadDailyData(model, name, 'tasmax')

    tbins = xr.DataArray(np.zeros((tasmin.time.size, tasmin.lat.size, tasmin.lon.size, 14)),
                         coords={'time': tasmin.time, 'lat': tasmin.lat, 'lon': tasmin.lon, 'tbin': np.arange(14)},
                         dims=['time', 'lat', 'lon', 'tbin'])

    for y in np.sort(np.unique(tasmin['time.year'].values)):
        print(y)
        tasmini = tasmin.sel(time=str(y))
        tasmaxi = tasmax.sel(time=str(y))

        dt, dx, dy = tasmini.shape

        tasmini_vals = np.ma.masked_array(np.reshape(np.repeat(tasmini.values, 24, axis=np.newaxis), (dt, dx, dy, 24)))
        tasmini_vals = np.ma.masked_where(np.isnan(tasmini_vals), tasmini_vals)
        tasmini_vals = np.ma.harden_mask(tasmini_vals)

        tasmaxi_vals = np.ma.masked_array(np.reshape(np.repeat(tasmaxi.values, 24, axis=np.newaxis), (dt, dx, dy, 24)))
        tasmaxi_vals = np.ma.masked_where(np.isnan(tasmaxi_vals), tasmaxi_vals)
        tasmaxi_vals = np.ma.harden_mask(tasmaxi_vals)

        hrs = np.reshape(np.tile(np.arange(24), (dt, dx, dy)), (dt, dx, dy, 24))

        cos_hrs = np.cos(hrs * np.pi / 12)
        amplitude = (-1) * (tasmaxi_vals - tasmini_vals) / 2
        offset = (tasmaxi_vals + tasmini_vals) / 2
        t_hrs = (amplitude * cos_hrs) + offset
        t_hrs = t_hrs - 273.15
        t_hrs = np.clip(t_hrs, a_min=t_hrs.min(), a_max=42)
        bins = np.arange(0, 43, 3)
        tbins.loc[dict(time=tasmini.time)] = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 3, t_hrs) / 24
    save_dir = '/project2/geos39650/ag_data/climate_projections/phase3'
    tbins.to_netcdf('{0}/{1}_{2}_{3}_MDW_daily_1971_2100.nc'.format(save_dir, name, 'ssp585', 'tbins'))
    return True


def fixedGS_ssp(name, clim_var):
    """
    A function to compile the fixed-season climate variables
    :param name: name of climate model
    :param clim_var: name of wanted variable
    :return: True after making fixed-season file
    """
    plant_date = '-03-01'
    har_date = '-08-28'

    clim_dir = '/project2/geos39650/ag_data/climate_projections/phase3'
    clim_var_path = '{0}/{1}_{2}_{3}_MDW_daily_1971_2100.nc'.format(clim_dir, name, 'ssp585', clim_var)
    clim_arr = xr.open_dataarray(clim_var_path)

    years = clim_arr['time.year'].resample(time='1A').max().values

    if clim_var == 'tbins':
        dims = ['time', 'lat', 'lon', 'tbin']
        coords = {'time': pd.date_range(str(years[0]), str(years[-1] + 1), freq='1A'),
                  'lat': clim_arr.lat,
                  'lon': clim_arr.lon,
                  'tbin': np.arange(14)}
    else:
        dims = ['time', 'lat', 'lon']
        coords = {'time': pd.date_range(str(years[0]), str(years[-1] + 1), freq='1A'),
                  'lat': clim_arr.lat,
                  'lon': clim_arr.lon}

    out = xr.DataArray(dims=dims, coords=coords)

    for year in years:
        print(year)
        clim_year = clim_arr.sel(time=slice(str(year) + plant_date, str(year) + har_date)).values
        clim_year = np.ma.masked_where(np.isnan(clim_year), clim_year)
        out.loc[dict(time=str(year))] = np.ma.sum(clim_year, axis=0)

    save_dir = '/project2/geos39650/ag_data/growseasons/phase3'
    out.to_netcdf('{0}/{1}_{2}_{3}_MDW_fixedGS_1971_2100.nc'.format(save_dir, name, 'ssp585', clim_var))
    return True

def trueGS_ssp(name, crop_model, clim_var, include_partial=False):
    """
    Get the true-season values for GDD, HDD, Pr
    :param name: climate model
    :param crop_model: crop model
    :param clim_var: variable of climate to calculate
    :param include_partial: whether you want to include partially grown yields or not
    :return:
    """
    crop_lower = crop_model.lower()
    startyear = 1971
    endyear   = 2100
    short_name = name.split('_')[0]

    clim_dir = '/project2/geos39650/ag_data/climate_projections/phase3'
    clim_var_path = '{0}/{1}_ssp585_{2}_MDW_daily_1971_2100.nc'.format(clim_dir, name, clim_var)
    clim_arr = xr.open_dataarray(clim_var_path)
    if clim_var == 'tbins':
        dims = ['time', 'lat', 'lon', 'tbin']
        extrayear_coords={'time': pd.date_range(str(endyear+1)+'-01-01', str(endyear+1)+'-12-31', freq='1D'),
                          'lat': clim_arr.lat,
                          'lon': clim_arr.lon,
                          'tbin': np.arange(14)}
        coords={'time':pd.date_range(str(startyear), str(endyear+1), freq='1A'),
                'lat':clim_arr.lat,
                'lon':clim_arr.lon,
                'tbin': np.arange(14)}
    else:
        dims = ['time', 'lat', 'lon']
        extrayear_coords = {'time': pd.date_range(str(endyear + 1) + '-01-01', str(endyear + 1) + '-12-31', freq='1D'),
                            'lat': clim_arr.lat,
                            'lon': clim_arr.lon}
        coords = {'time': pd.date_range(str(startyear), str(endyear + 1), freq='1A'),
                  'lat': clim_arr.lat,
                  'lon': clim_arr.lon}

    extra_year = xr.DataArray(dims=dims, coords=extrayear_coords)
    clim_arr = xr.concat([clim_arr, extra_year], dim='time')
    out = xr.DataArray(dims=dims, coords=coords)

    PlHa_dir = '/project2/geos39650/ag_data/ggcmi_phase3'
    plant_file  = '{0}/{1}_{2}_w5e5_ssp585_2015soc_2015co2_{3}-mai-noirr_global_annual_1971_2100.nc'.format(PlHa_dir,
                                                                                                            crop_lower,
                                                                                                            short_name,
                                                                                                            'plantday')
    plant       = xr.open_dataarray(plant_file).values
    plant       = np.append(plant, np.empty(plant[0].shape)[np.newaxis,:, :]*np.NaN, axis=0)
    plant       = np.ma.masked_array(plant)
    plant       = np.ma.masked_where(np.isnan(plant), plant)
    plant       = np.ma.harden_mask(plant).astype(int)
    har_file    = '{0}/{1}_{2}_w5e5_ssp585_2015soc_2015co2_{3}-mai-noirr_global_annual_1971_2100.nc'.format(PlHa_dir,
                                                                                                            crop_lower,
                                                                                                            short_name,
                                                                                                            'matyday')
    har         = xr.open_dataarray(har_file).values
    har         = np.append(har, np.empty(har[0].shape)[np.newaxis, :, :]*np.NaN, axis=0)
    har         = np.ma.masked_array(har)
    har         = np.ma.masked_where(np.isnan(har), har)
    har         = np.ma.harden_mask(har).astype(int)
    if include_partial: # If you want to include yields that weren't fully grown
        har     = np.abs(har)

    for i in range(0, endyear - startyear + 1):
        curryear = startyear + i
        print(curryear, '-', curryear + 1)
        clim_year       = clim_arr.sel(time=slice(str(curryear), str(curryear + 1)))
        lenyear1        = clim_year.sel(time=str(curryear)).time.size
        lenyear2        = clim_year.time.size
        dx              = clim_year.lat.size
        dy              = clim_year.lon.size
        clim_year       = clim_year.values

        pl = plant[i:i + 2].copy()
        pl = np.ma.masked_where(np.isnan(pl), pl)
        # Mask PLANTING dates WHERE ZERO (never planted)
        pl = np.ma.masked_where(pl == 0, pl)
        pl[1] = pl[1] + lenyear1

        ha = har[i:i + 2].copy()
        ha = np.ma.masked_where(np.isnan(ha), ha)
        # Mask HARVEST dates WHERE ZERO (failure)
        ha = np.ma.masked_where(ha == 0, ha)
        ha = pl + ha

        dayids = np.reshape(np.repeat(np.repeat(np.arange(1, lenyear2+1, 1), dx, axis=np.newaxis), dy, axis=np.newaxis),
                            (lenyear2, dx, dy))

        # Mask PLANTING dates AFTER Year-1 ended
        pl = np.ma.masked_where(pl > lenyear1, pl)
        ha = np.ma.masked_where(pl > lenyear1, ha)

        # Mask HARVEST dates AFTER Year-2 ended
        pl = np.ma.masked_where(ha > lenyear2, pl)
        ha = np.ma.masked_where(ha > lenyear2, ha)

        if i == dy-1: # If it's the last year of calculation
            # Cut HARVEST dates AFTER the last REAL year (Year-1)
            pl = np.ma.masked_where(ha > lenyear1, pl)
            ha = np.ma.masked_where(ha > lenyear1, ha)

        # Check if you've got any double-dates
        plsum = np.ma.sum(pl / pl, axis=0)
        plwhere = np.where(plsum > 1)
        hasum = np.ma.sum(ha / ha, axis=0)
        hawhere = np.where(hasum > 1)
        if not ((plsum[plwhere].size == 0) and (hasum[hawhere].size == 0)):
            print('Size', plwhere[0].size)
            for j in range(plwhere[0].size):
                assert plwhere[0][j] == hawhere[0][j], "Double-dates don't match for PL and HA"
                assert plwhere[1][j] == hawhere[1][j], "Double-dates don't match for PL and HA"
                lat_i = plwhere[0][j]
                lon_i = plwhere[1][j]
                print("[{0}, {1}]".format(clim_arr.lat.values[lat_i], clim_arr.lon.values[lon_i]))
                pl.mask[1, lat_i, lon_i] = True     # Mask the earlier harvest, only show the latter
                ha.mask[1, lat_i, lon_i] = True     # Mask the earlier harvest, only show the latter


        # Collapse the 2 1-years into a single 2-year 2D array
        pl = np.sum(pl, axis=0)
        ha = np.sum(ha, axis=0)

        # Repeat out to fit daily size
        pl = np.repeat(pl[np.newaxis, :, :], lenyear2, axis=0)
        ha = np.repeat(ha[np.newaxis, :, :], lenyear2, axis=0)
        if clim_var == 'tbins':
            pl = np.repeat(pl[:, :, :, np.newaxis], 14, axis=3)
            ha = np.repeat(ha[:, :, :, np.newaxis], 14, axis=3)
            dayids = np.repeat(dayids[:, :, :, np.newaxis], 14, axis=3)

        # Assert that same grid cells are masked in PL and HA arrays
        assert np.array_equal(ha.mask, pl.mask), "Masks of dates don't match for PL and HA"

        clim_year = np.ma.masked_where(dayids < pl, clim_year)
        clim_year = np.ma.masked_where(dayids > ha, clim_year)
        clim_year = np.ma.masked_where(clim_year >= 1e20, clim_year)
        clim_year = np.ma.harden_mask(clim_year)

        out.loc[dict(time=str(curryear))] = np.ma.sum(clim_year, axis=0)

    save_dir = '/project2/geos39650/ag_data/growseasons/phase3'
    if include_partial:
        out.to_netcdf('{0}/{1}_{2}_{3}_{4}_MDW_TRUEgs_partial_1971_2100.nc'.format(save_dir,
                                                                                   crop_lower,
                                                                                   name,
                                                                                   'ssp585',
                                                                                   clim_var))
    else:
        out.to_netcdf('{0}/{1}_{2}_{3}_{4}_MDW_TRUEgs_1971_2100.nc'.format(save_dir,
                                                                                   crop_lower,
                                                                                   name,
                                                                                   'ssp585',
                                                                                   clim_var))
    return True


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

def getDailyValues(): # Ran on Nov. 12
    for model in clim_models:
        name = clim_names[model]
        print('MODEL')
        print(model)
        print('loading PR')
        pr_loaded = loadPR(model, name)
        print('LOADED PR')
        print('loading TEMP')
        temp_loaded = loadGDDandHDD(model, name)
        print('LOADED TEMP')
    return True

def getSeasonSums(): # Ran on Nov. 12
    clim_vars = ['gdd', 'hdd', 'pr']
    for clim_var in clim_vars:
        print('VARIABLE: {0}'.format(clim_var))
        print('\n')
        for model in clim_models:
            name = clim_names[model]
            print('MODEL: {0}'.format(model))
            clim_loaded = fixedGS_ssp(name, clim_var)
            print('COMPLETE: {0}'.format(clim_loaded))

def get_GSmean_tas(): # Ran on Nov. 12
    for model in clim_models:
        name = clim_names[model]
        print('MODEL')
        print(model)
        print('loading TAS')
        tas = loadDailyData(model, name, 'tas')
        tas = tas.sel(time=tas.time.dt.month.isin([3,4,5,6,7,8])).resample(time="1A").mean()
        save_dir = '/project2/geos39650/ag_data/climate_projections/phase3'
        tas.to_netcdf('{0}/{1}_{2}_{3}_MDW_GSmean_1971_2100.nc'.format(save_dir, name, 'ssp585', 'tas'))
        print('LOADED TAS')
    return True

def compile_GGCMI_files(): # Ran on Nov. 12
    output_vars = ['harvyear', 'matyday', 'plantday', 'plantyear', 'yield']
    crop_model = 'LPJmL'
    for output_var in output_vars:
        print('VARIABLE: {0}'.format(output_var))
        print('\n')
        for model in clim_models:
            name = clim_names[model]
            print('MODEL: {0}'.format(model))
            output_loaded = compileOutput(name, crop_model, output_var)
            print('COMPLETE: {0}'.format(output_loaded))

def get_trueGS_Sums(): # Ran on Nov. 12
    clim_vars = ['gdd', 'hdd', 'pr']
    for clim_var in clim_vars:
        print('VARIABLE: {0}'.format(clim_var))
        for model in clim_models:
            name = clim_names[model]
            print('MODEL: {0}'.format(model))
            clim_loaded = trueGS_ssp(name, clim_var)
            print('COMPLETE: {0}'.format(clim_loaded))
        print('\n')

def get_trueGS_Sums_wPartial(): # Ran on Nov. 17
    clim_vars = ['gdd', 'hdd', 'pr']
    for clim_var in clim_vars:
        print('VARIABLE: {0}'.format(clim_var))
        for model in clim_models:
            name = clim_names[model]
            print('MODEL: {0}'.format(model))
            clim_loaded = trueGS_ssp(name, clim_var, include_partial=True)
            print('COMPLETE: {0}'.format(clim_loaded))
        print('\n')

def compile_all_GGCMI_phase3(): # Ran on Nov. 23
    output_vars = ['harvyear', 'matyday', 'plantday', 'plantyear', 'yield']
    crop_models = ['LPJmL', 'EPIC-IIASA', 'PROMET', 'PEPIC', 'CROVER', 'CYGMA1p74', 'ACEA', 'SIMPLACE-LINTUL5']
    for crop_model in crop_models:
        print('CROP MODEL: {0}'.format(crop_model))
        for output_var in output_vars:
            print('VARIABLE: {0}'.format(output_var))
            for model in clim_models:
                name = clim_names[model]
                print('MODEL: {0}'.format(model))
                try:
                    output_loaded = compileOutput(name, crop_model, output_var)
                    print('COMPLETE: {0}'.format(output_loaded))
                except:
                    print('NOT FOUND')
                    continue
            print('\n')

def get_trueGS_SumsPartial_byModel(): # Ran on Nov. 23
    crop_models = ['LPJmL', 'CYGMA1p74']
    clim_vars = ['gdd', 'hdd', 'pr']
    for crop_model in crop_models:
        print('CROP MODEL: {0}'.format(crop_model))
        for clim_var in clim_vars:
            print('VARIABLE: {0}'.format(clim_var))
            for model in clim_models:
                name = clim_names[model]
                print('MODEL: {0}'.format(model))
                clim_loaded = trueGS_ssp(name, crop_model, clim_var, include_partial=True)
                print('COMPLETE: {0}'.format(clim_loaded))
            print('\n')

def getDailyBins(): # Ran on Dec. 7
    for model in clim_models:
        name = clim_names[model]
        print('MODEL')
        print(model)
        print('loading Bins')
        bins = loadBins(model, name)
        print('LOADED')
        print('\n')
    return True

def getBinsFixedSeason(): # Ran on Dec. 8
    clim_var = 'tbins'
    print('VARIABLE: {0}'.format(clim_var))
    print('FIXED SEASON')
    print('\n')
    for model in clim_models:
        name = clim_names[model]
        print('MODEL: {0}'.format(model))
        clim_loaded = fixedGS_ssp(name, clim_var)
        print('COMPLETE: {0}'.format(clim_loaded))
        print('\n')

def getBinsTrueSeason(): # Ran on Dec. 8
    clim_var = 'tbins'
    print('TRUE SEASONS')
    crop_models = ['LPJmL', 'CYGMA1p74']
    for crop_model in crop_models:
        print('CROP MODEL: {0}'.format(crop_model))
        for model in clim_models:
            name = clim_names[model]
            print('MODEL: {0}'.format(model))
            clim_loaded = trueGS_ssp(name, crop_model, clim_var, include_partial=True)
            print('COMPLETE: {0}'.format(clim_loaded))
        print('\n')


def compile_new_GGCMI_phase3(): # Ran on Jun. 3
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


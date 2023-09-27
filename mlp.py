# Create an MPL model based on the Colab notebook
# imports
# import sys
# pbm = sys.argv[1]
# cmodel = int(sys.argv[2])
import argparse

# imports
from build_dataset import *
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt

#intel patch to accelerate ml algorithms
from sklearnex import patch_sklearn
patch_sklearn()

import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
print('SKLEARN', sklearn.__version__)
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV
# import xgboost as xgb
import xarray as xr, numpy as np, pandas as pd
print('IMPORTS complete.')

#################
# VARIABLE INPUTS
#################
clim_model = 'GFDL-ESM4'
clim_name = 'gfdl-esm4_r1i1p1f1_w5e5'
crop_model = 'LPJmL'
weather_vars = ['tas']
weather_agg = ['mean']
soil_vars = ['texture_class', 'soil_ph', 'soil_caco3']
output_vars = ['yield', 'plantday']
ssp = 'ssp585'


if __name__ == '__main__':

    """# Load climate model"""
    # area_mask = xr.open_dataset('/content/drive/Shareddrives/GEOS39650/yields/data/ag_district_mask_coordinates.nc')
    # mdw       = area_mask['Midwest']
    # loaddir = '/content/drive/Shareddrives/GEOS39650/yields/data/rcp_crop_runs/phase3/'

    weather = []
    for var, agg_method in zip(weather_vars, weather_agg):
        varout = loadWeatherDaily(clim_model, clim_name, var, coords=[(48.75, 36.25), (-103.75, -80.25)], ssp=ssp)
        varout = varout.sel(time=slice('1981','2100'))
        varout = varout.sel(time=varout.time.dt.month.isin(np.arange(3,9)))
        if agg_method=='mean':
            varout = varout.resample(time='1M').mean()
        elif agg_method=='sum':
            varout = varout.resample(time='1M').sum()
        else:
            print('WRONG AGG METHOD.')
        if 'tas' in var: varout = varout - 273.15
        weather.append(varout)

    # def getWeather(cmodel, var, aggmethod='mean'):
    #     if var=='pr':
    #         out = '{0}_r1i1p1f1_w5e5_ssp585_{1}_MDW_daily_1971_2100.nc'.format(cmodel.lower(), var)
    #     else:
    #         out = '{0}_r1i1p1f1_w5e5_ssp585_{1}_MDW_daily_1981_2100.nc'.format(cmodel.lower(), var)
    #     out = xr.open_dataarray(loaddir+out).sel(lat=slice(48.75, 36.25), lon=slice(-103.8, -80.75), time=slice('1981','2100')).where(mdw==1)
    #     out = out.sel(time=out.time.dt.month.isin(np.arange(3,9))).rename(var)
    #     if aggmethod=='mean':
    #         out = out.resample(time='1M').mean()
    #     elif aggmethod=='sum':
    #         out = out.resample(time='1M').sum()
    #     else:
    #         print('WRONG AGG METHOD.')
    #     return out

    # vpd = getWeather(cmodel, 'vpd', aggmethod='sum')
    # pr = getWeather(cmodel, 'pr', aggmethod='sum')
    # hurs = getWeather(cmodel, 'hurs', aggmethod='mean')
    # tasmax = getWeather(cmodel, 'tasmax', aggmethod='mean') - 273.15
    # tasmin = getWeather(cmodel, 'tasmin', aggmethod='mean') - 273.15
    # tas = ((tasmin + tasmax) / 2).rename('tas')

    # weather = xr.merge([vpd,tas,pr,hurs])
    # # weather = weather - weather.sel(time=slice('1981','2010')).mean(dim=['time'])

    # """## Soils"""
    # soilvars = ['texture_class', 'soil_ph', 'soil_caco3']#, 'bulk_density', 'cec_soil', 'oc', 'awc', 'sand', 'silt', 'clay', 'gravel', 'ece', 'bs_soil',
    #             #'issoil', 'root_obstacles', 'impermeable_layer', 'mu_global', 'lon', 'lat']
    # soils = loaddir+'soils/HWSD_soil_data_on_cropland_v2.3.nc'
    # soils = xr.open_dataset(soils).sel(lat=slice(48.75, 36.25), lon=slice(-103.8, -80.75)).where(mdw==1)
    # soils = soils[soilvars]

    # """# Load and format historical data"""
    # ag = '{0}_{1}_w5e5_ssp585_2015soc_2015co2_yield-mai-noirr_global_annual_1981_2100.nc'.format(pbm, cmodel.lower())
    # ag = '{0}ggcmi/{1}/{2}'.format(loaddir, pbm, ag)
    # ag = xr.open_dataarray(ag).sel(lat=slice(48.75, 36.25), lon=slice(-103.8, -80.75)).where(mdw==1).rename('yield_rf')
    # ag = ag.where((ag > 0))# | (np.isnan(ag)), 0.1)

    # # # Can log yields
    # # ag = np.log(ag)

    # # # OR take the anomalies
    # # ag = ag - ag.sel(time=slice('1981','2010')).mean(dim=['time'])

    # ag = ag.to_dataframe().reset_index()
    # ag['year'] = ag.time.dt.year
    # fe = ag.groupby(['lat','lon'])['yield_rf'].mean().reset_index().rename(columns={'yield_rf':'FE'})
    # ag = ag.merge(fe, on=['lat','lon'])
    # ag = ag[['lat','lon','year','yield_rf','FE']]
    # ag.loc[ag.yield_rf==0] = 0

    # df = weather.to_dataframe().reset_index().dropna()
    # df['year'] = df.time.dt.year
    # df['month'] = df.time.dt.month
    # df = df.drop('time', axis=1)
    # df = df.pivot(index=['year','lat','lon'], columns='month')

    # # Reset column names
    # df.columns = [f'{col[0]}_{col[1]}' for col in df.columns]

    # # Reset the index
    # df.reset_index(inplace=True)
    # df = df.merge(soils.to_dataframe().reset_index(), on=['lat','lon'])
    # df = ag.merge(df, on=['year','lat','lon']).dropna()

    # df_train = df[df.year.isin(np.arange(1981,2011))]
    # df_pred = df[df.year.isin(np.arange(2011,2101))]

    # var_combos = [['tas_'],['pr_'],['hurs_']]

    # for combo in var_combos:

    #     # Split data into labels & features -- and convert to numpy arrays
    #     # CUSTOM VARIABLES
    #     labels = df_train['yield_rf'].values.flatten()
    #     months_incl = np.arange(3,9)
    #     months_excl = np.array([month for month in np.arange(1,13) if month not in months_incl])
    #     weather_vars = ['tas_','pr_','hurs_']#,'vpd_']
    #     weather_vars = [var+str(month).zfill(1) for var in weather_vars for month in months_incl]
    #     other_vars = soilvars
    #     df_features = df_train[other_vars+weather_vars]
    #     print(df_features.columns)
    #     feature_list=list(df_features.columns)
    #     features=np.array(df_features)

    #     """## Train-test split"""

    #     random_state = 4
    #     # RANDOM SPLIT
    #     train_features, test_features, train_labels, test_labels = train_test_split(features,
    #                                                                                 labels,
    #                                                                                 test_size=0.20,
    #                                                                                 random_state = random_state,
    #                                                                                 shuffle = True)

    #     """# Tune model hyperparameters"""

    #     # Create a reference model to be tuned.
    #     mpl = make_pipeline(
    #         StandardScaler(),
    #         MLPRegressor(random_state = random_state)
    #     )

    #     def getTunedModel( baseModel, random_state ):
    #         max_iter = sp_randInt(50, 500)
    #         hidden_layer_sizes = [(10,), (20,), (50,), (100,), (250,),
    #                             (10,10), (20,20), (50,50), (100,100), (250,250),
    #                             (10,10,10), (20,20,20), (50,50,50), (100,100,100), (250,250,250),]

    #         random_grid = {
    #             'mlpregressor__max_iter': max_iter,
    #             'mlpregressor__hidden_layer_sizes':hidden_layer_sizes
    #             }
    #         print(random_grid)

    #         model_tuned = RandomizedSearchCV(cv=5, estimator = baseModel, param_distributions = random_grid,
    #                                         n_iter = 4, verbose=1, random_state=random_state , n_jobs = -1)
    #         return model_tuned

    #     # Run tuning to find optimal hyperparameters.
    #     mpl_tuned = getTunedModel(mpl, random_state)
    #     mpl_tuned.fit(train_features,train_labels)

    #     result = pd.DataFrame.from_dict(mpl_tuned.cv_results_)

    #     """# Select best parameters and fit model"""

    #     # Choose the best hyperparameters from the random CV search.
    #     best = result[result.rank_test_score == 1]
    #     key = list(best.params.keys())[0]
    #     best_params = dict(best.params)[key]
    #     max_iter = best_params['mlpregressor__max_iter']
    #     hidden_layer_sizes = best_params['mlpregressor__hidden_layer_sizes']

    #     # Create a new model instance with the optimal hyperparameters.
    #     # mpl_opt = MLPRegressor(random_state = random_state,
    #     #                        hidden_layer_sizes = hidden_layer_sizes,
    #     #                        max_iter = max_iter)
    #     mpl_opt = make_pipeline(
    #         StandardScaler(),
    #         MLPRegressor(random_state = random_state,
    #                     hidden_layer_sizes = hidden_layer_sizes,
    #                     max_iter = max_iter)
    #     )

    #     # Re-split the dataset (may be unnecessary) and fit with the optimal model.
    #     mpl_opt.fit(features, labels)

    #     # Make historical predictions and plot residuals.
    #     y_pred = mpl_opt.predict(features)
    #     residuals = y_pred - labels

    #     # Add performance metrics to the blurb output.
    #     blurb = 'RF model (split train-test): 10-iter CV.'
    #     blurb = blurb + '\nGoodness of Fit (R2): {0}'.format(metrics.r2_score(labels, y_pred))
    #     blurb = blurb + '\nMean Absolute Error (MAE): {0}'.format(metrics.mean_absolute_error(labels, y_pred))
    #     blurb = blurb + '\nMean Squared Error (MSE): {0}'.format(metrics.mean_squared_error(labels, y_pred))
    #     blurb = blurb + '\nRoot Mean Squared Error (RMSE): {0}'.format(np.sqrt(metrics.mean_squared_error(labels, y_pred)))
    #     mape = np.mean(np.abs((labels - y_pred) / np.abs(labels+0.001)))
    #     blurb = blurb + '\nMean Absolute Percentage Error (MAPE): {0}'.format(round(mape * 100, 2))
    #     blurb = blurb + '\nAccuracy: {0}'.format(round(100*(1 - mape), 2))
    #     print(blurb)

    #     df_train['pred'] = mpl_opt.predict(features)
    #     df_train.loc[df_train["pred"] <= 0.0, "pred"] = 0.0

    #     df_pred['pred'] = mpl_opt.predict(df_pred[feature_list].values)
    #     df_pred.loc[df_pred["pred"] <= 0.0, "pred"] = 0.0

    #     ds = xr.merge([df_train.set_index(['year','lat','lon']).to_xarray(),
    #                 df_pred.set_index(['year','lat','lon']).to_xarray()])
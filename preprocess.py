# coding=utf-8
import turtledemo.chaos

import matplotlib.pyplot as plt
import tweepy.utils

from __init__ import *
import xarray as xr
import climate_indices
from climate_indices import compute
from climate_indices import indices
from meta_info import *

class GIMMS_NDVI:

    def __init__(self):
        self.datadir = join(data_root, 'GIMMS_NDVI')
        pass

    def run(self):
        # self.resample()
        # self.monthly_compose()
        # self.per_pix()
        # self.per_pix_anomaly()
        self.per_pix_anomaly_detrend()
        pass

    def resample(self):
        fdir = join(self.datadir,'tif_8km_bi_weekly')
        outdir = join(self.datadir,'bi_weekly_05')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            ToRaster().resample_reproj(fpath,outpath,0.5)

    def monthly_compose(self):
        fdir = join(self.datadir,'bi_weekly_05')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir)
        Pre_Process().monthly_compose(fdir,outdir,method='max')
        pass

    def per_pix(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'per_pix')
        T.mk_dir(outdir)
        Pre_Process().data_transform(fdir,outdir)

    def per_pix_anomaly(self):
        fdir = join(self.datadir,'per_pix')
        outdir = join(self.datadir,'per_pix_anomaly')
        T.mk_dir(outdir)
        Pre_Process().cal_anomaly(fdir,outdir)

    def per_pix_anomaly_detrend(self):
        fdir = join(self.datadir,'per_pix_anomaly')
        outdir = join(self.datadir,'per_pix_anomaly_detrend')
        T.mk_dir(outdir)
        Pre_Process().detrend(fdir,outdir)
        pass

class SPEI:

    def __init__(self):
        self.datadir = join(data_root, 'SPEI')
        pass

    def run(self):
        # self.nc_to_tif()
        # self.per_pix()
        # self.clean()
        # self.every_month()
        self.pick_year_range()
        pass

    def nc_to_tif(self):
        fdir = join(self.datadir,'nc')
        outdir = join(self.datadir,'tif')
        params = []
        for f in T.listdir(fdir):
            scale = f.split('.')[0]
            outdir_i = join(outdir,scale)
            T.mk_dir(outdir_i,force=True)
            fpath = join(fdir,f)
            param = [fpath,'spei',outdir_i]
            # self.kernel_nc_to_tif(param)
            # exit()
            params.append(param)
        MULTIPROCESS(self.kernel_nc_to_tif,params).run(process=7)

    def kernel_nc_to_tif(self,param):
        fpath, var, outdir_i = param
        self.nc_to_tif_func(fpath, var, outdir_i)
        pass


    def per_pix(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'per_pix')
        T.mk_dir(outdir)
        for folder in T.listdir(fdir):
            print(folder)
            fdir_i = join(fdir,folder)
            outdir_i = join(outdir,global_year_range,folder)
            T.mk_dir(outdir_i,force=True)
            Pre_Process().data_transform(fdir_i, outdir_i)

    def clean(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        outdir = join(self.datadir,'per_pix_clean',global_year_range)
        T.mk_dir(outdir,force=True)

        for scale in T.listdir(fdir):
            outf = join(outdir,scale)
            fdir_i = join(fdir,scale)
            spatial_dict = T.load_npy_dir(fdir_i)
            spatial_dict_out = {}
            for pix in tqdm(spatial_dict,desc=scale):
                r,c = pix
                if r > 180:
                    continue
                vals = spatial_dict[pix]
                vals = np.array(vals)
                vals[vals<-999] = np.nan
                vals[vals>999] = np.nan
                if T.is_all_nan(vals):
                    continue
                spatial_dict_out[pix] = vals
            T.save_npy(spatial_dict_out, outf)

    def every_month(self):
        fdir = join(self.datadir,'per_pix_clean',global_year_range)
        outdir = join(self.datadir,'every_month',global_year_range)
        params_list = []
        for f in T.listdir(fdir):
            scale = f.split('.')[0]
            outdir_i = join(outdir, scale)
            T.mkdir(outdir_i, force=True)
            param = [fdir,f,outdir_i]
            # self.kernel_every_month(param)
            params_list.append(param)
        MULTIPROCESS(self.kernel_every_month,params_list).run(process=7)

    def kernel_every_month(self,params):
        fdir,f,outdir_i = params
        fpath = join(fdir, f)
        spatial_dict = T.load_npy(fpath)
        month_list = range(1, 13)
        for mon in month_list:
            spatial_dict_mon = {}
            for pix in tqdm(spatial_dict, desc=f'{mon}'):
                r, c = pix
                if r > 180:
                    continue
                vals = spatial_dict[pix]
                val_mon = T.monthly_vals_to_annual_val(vals, [mon])
                val_mon[val_mon < -10] = -999999
                num = T.count_num(val_mon, -999999)
                if num > 10:
                    continue
                val_mon[val_mon < -10] = np.nan
                if T.is_all_nan(val_mon):
                    continue
                spatial_dict_mon[pix] = val_mon
            outf = join(outdir_i, f'{mon:02d}')
            T.save_npy(spatial_dict_mon, outf)
        pass

    def nc_to_tif_func(self, fname, var_name, outdir):
        try:
            ncin = Dataset(fname, 'r')
            print(ncin.variables.keys())

        except:
            raise UserWarning('File not supported: ' + fname)
        try:
            lat = ncin.variables['lat'][:]
            lon = ncin.variables['lon'][:]
        except:
            try:
                lat = ncin.variables['latitude'][:]
                lon = ncin.variables['longitude'][:]
            except:
                try:
                    lat = ncin.variables['lat_FULL'][:]
                    lon = ncin.variables['lon_FULL'][:]
                except:
                    raise UserWarning('lat or lon not found')
        shape = np.shape(lat)
        try:
            time = ncin.variables['time_counter'][:]
            basetime_str = ncin.variables['time_counter'].units
        except:
            time = ncin.variables['time'][:]
            basetime_str = ncin.variables['time'].units

        basetime_unit = basetime_str.split('since')[0]
        basetime_unit = basetime_unit.strip()
        print(basetime_unit)
        print(basetime_str)
        if basetime_unit == 'days':
            timedelta_unit = 'days'
        elif basetime_unit == 'years':
            timedelta_unit = 'years'
        elif basetime_unit == 'month':
            timedelta_unit = 'month'
        elif basetime_unit == 'months':
            timedelta_unit = 'month'
        elif basetime_unit == 'seconds':
            timedelta_unit = 'seconds'
        elif basetime_unit == 'hours':
            timedelta_unit = 'hours'
        else:
            raise Exception('basetime unit not supported')
        basetime = basetime_str.strip(f'{timedelta_unit} since ')
        try:
            basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d')
        except:
            try:
                basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M:%S')
            except:
                try:
                    basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M:%S.%f')
                except:
                    try:
                        basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M')
                    except:
                        try:
                            basetime = datetime.datetime.strptime(basetime, '%Y-%m')
                        except:
                            raise UserWarning('basetime format not supported')
        data = ncin.variables[var_name]
        if len(shape) == 2:
            xx, yy = lon, lat
        else:
            xx, yy = np.meshgrid(lon, lat)
        for time_i in tqdm(range(len(time))):
            if basetime_unit == 'days':
                date = basetime + datetime.timedelta(days=int(time[time_i]))
            elif basetime_unit == 'years':
                date1 = basetime.strftime('%Y-%m-%d')
                base_year = basetime.year
                date2 = f'{int(base_year + time[time_i])}-01-01'
                delta_days = Tools().count_days_of_two_dates(date1, date2)
                date = basetime + datetime.timedelta(days=delta_days)
            elif basetime_unit == 'month' or basetime_unit == 'months':
                date1 = basetime.strftime('%Y-%m-%d')
                base_year = basetime.year
                base_month = basetime.month
                date2 = f'{int(base_year + time[time_i] // 12)}-{int(base_month + time[time_i] % 12)}-01'
                delta_days = Tools().count_days_of_two_dates(date1, date2)
                date = basetime + datetime.timedelta(days=delta_days)
            elif basetime_unit == 'seconds':
                date = basetime + datetime.timedelta(seconds=int(time[time_i]))
            elif basetime_unit == 'hours':
                date = basetime + datetime.timedelta(hours=int(time[time_i]))
            else:
                raise Exception('basetime unit not supported')
            time_str = time[time_i]
            mon = date.month
            year = date.year
            if year < 1982:
                continue
            # print(year)
            # exit()
            day = date.day
            outf_name = f'{year}{mon:02d}{day:02d}.tif'
            outpath = join(outdir, outf_name)
            if isfile(outpath):
                continue
            arr = data[time_i]
            arr = np.array(arr)
            lon_list = xx.flatten()
            lat_list = yy.flatten()
            val_list = arr.flatten()
            lon_list[lon_list > 180] = lon_list[lon_list > 180] - 360
            df = pd.DataFrame()
            df['lon'] = lon_list
            df['lat'] = lat_list
            df['val'] = val_list
            lon_list_new = df['lon'].tolist()
            lat_list_new = df['lat'].tolist()
            val_list_new = df['val'].tolist()
            DIC_and_TIF().lon_lat_val_to_tif(lon_list_new, lat_list_new, val_list_new, outpath)

    def pick_year_range(self):

        fdir = join(self.datadir,'per_pix_clean','1982-2015')
        year_range_list = []
        for VI in global_VIs_year_range_dict:
            year_range = global_VIs_year_range_dict[VI]
            if year_range == '1982-2015':
                continue
            outdir = join(self.datadir,'per_pix_clean',year_range)
            T.mk_dir(outdir)
            start_year = int(year_range.split('-')[0])
            end_year = int(year_range.split('-')[1])
            date_list = []
            for y in range(1982,2015 + 1):
                for m in range(1,13):
                    date = f'{y}-{m:02d}'
                    date_list.append(date)
            pick_date_list = []
            for y in range(start_year, end_year + 1):
                for m in range(1, 13):
                    date = f'{y}-{m:02d}'
                    pick_date_list.append(date)
            for f in T.listdir(fdir):
                fpath = join(fdir,f)
                outf = join(outdir,f)
                dic = T.load_npy(fpath)
                picked_vals_dic = {}
                for pix in tqdm(dic):
                    vals = dic[pix]
                    dic_i = dict(zip(date_list,vals))
                    picked_vals = []
                    for date in pick_date_list:
                        val = dic_i[date]
                        picked_vals.append(val)
                    picked_vals = np.array(picked_vals)
                    picked_vals_dic[pix] = picked_vals
                T.save_npy(picked_vals_dic,outf)


class SPI:
    def __init__(self):
        self.datadir = join(data_root,'SPI')
        pass

    def run(self):
        # self.cal_spi()
        # self.pick_SPI_year_range()
        self.every_month()
        # self.check_spi()
        pass

    def cal_spi(self):
        date_range = '1930-2020'
        data_start_year = 1930
        # P_dir = CRU().data_dir + 'pre/per_pix/'
        P_dir = join(Precipitation().datadir,'per_pix',date_range)
        # P_dic = T.load_npy_dir(P_dir,condition='005')
        P_dic = T.load_npy_dir(P_dir)
        # scale_list = [1,3,6,9,12]
        scale_list = range(1,25)
        for scale in scale_list:
            outdir = join(self.datadir,'per_pix',date_range)
            T.mk_dir(outdir,force=True)
            outf = join(outdir,f'spi{scale:02d}')
            # distrib = indices.Distribution('pearson')
            distrib = indices.Distribution('gamma')
            Periodicity = compute.Periodicity(12)
            spatial_dic = {}
            for pix in tqdm(P_dic,desc=f'scale {scale}'):
                r,c = pix
                if r > 180:
                    continue
                vals = P_dic[pix]
                vals = np.array(vals)
                vals = T.mask_999999_arr(vals,warning=False)
                if np.isnan(np.nanmean(vals)):
                    continue
                # zscore = Pre_Process().z_score_climatology(vals)
                spi = climate_indices.indices.spi(
                values=vals,
                scale=scale,
                distribution=distrib,
                data_start_year=data_start_year,
                calibration_year_initial=1960,
                calibration_year_final=2000,
                periodicity=Periodicity,
                # fitting_params: Dict = None,
                )
                spatial_dic[pix] = spi
                # plt.plot(spi)
                # plt.show()
            T.save_npy(spatial_dic,outf)

    def pick_SPI_year_range(self):
        fdir = join(self.datadir,'per_pix','1930-2020')
        outdir = join(self.datadir,'per_pix',global_year_range)
        T.mk_dir(outdir)
        start_year = 1930
        end_year = 2020
        date_list = []
        for y in range(start_year,end_year + 1):
            for m in range(1,13):
                date = f'{y}-{m:02d}'
                date_list.append(date)
        pick_date_list = []
        for y in range(1982, 2015 + 1):
            for m in range(1, 13):
                date = f'{y}-{m:02d}'
                pick_date_list.append(date)
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            outf = join(outdir,f)
            dic = T.load_npy(fpath)
            picked_vals_dic = {}
            for pix in tqdm(dic):
                vals = dic[pix]
                dic_i = dict(zip(date_list,vals))
                picked_vals = []
                for date in pick_date_list:
                    val = dic_i[date]
                    picked_vals.append(val)
                picked_vals = np.array(picked_vals)
                picked_vals_dic[pix] = picked_vals
            T.save_npy(picked_vals_dic,outf)

    def every_month(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        outdir = join(self.datadir,'every_month',global_year_range)
        params_list = []
        for f in T.listdir(fdir):
            scale = f.split('.')[0]
            outdir_i = join(outdir, scale)
            T.mkdir(outdir_i, force=True)
            param = [fdir,f,outdir_i]
            # self.kernel_every_month(param)
            params_list.append(param)
        MULTIPROCESS(self.kernel_every_month,params_list).run(process=7)

    def kernel_every_month(self,params):
        fdir,f,outdir_i = params
        fpath = join(fdir, f)
        spatial_dict = T.load_npy(fpath)
        month_list = range(1, 13)
        for mon in month_list:
            spatial_dict_mon = {}
            for pix in spatial_dict:
                r, c = pix
                if r > 180:
                    continue
                vals = spatial_dict[pix]
                val_mon = T.monthly_vals_to_annual_val(vals, [mon])
                val_mon[val_mon < -10] = -999999
                num = T.count_num(val_mon, -999999)
                if num > 10:
                    continue
                val_mon[val_mon < -10] = np.nan
                if T.is_all_nan(val_mon):
                    continue
                spatial_dict_mon[pix] = val_mon
            outf = join(outdir_i, f'{mon:02d}')
            T.save_npy(spatial_dict_mon, outf)
        pass

    def check_spi(self):
        fdir = join(self.datadir,'SPI',global_year_range)
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            spatial_dict = T.load_npy(fpath)
            spatial_dict1 = {}
            for pix in spatial_dict:
                vals = spatial_dict[pix]
                spatial_dict1[pix] = len(vals)
                # spatial_dict1[pix] = np.mean(vals)
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict1)
            plt.imshow(arr)
            plt.show()


class Precipitation:

    def __init__(self):
        self.datadir = join(data_root,'CRU_precip')
        pass

    def run(self):
        self.pick_year_range()
        # self.anomaly()
        self.detrend()
        # self.check_per_pix()
        pass

    def pick_year_range(self):
        fdir = join(self.datadir,'per_pix','1930-2020')
        outdir = join(self.datadir,'per_pix',global_year_range)
        T.mk_dir(outdir)
        outf = join(outdir,'precip')
        start_year = 1930
        end_year = 2020
        date_list = []
        for y in range(start_year, end_year + 1):
            for m in range(1, 13):
                date = f'{y}-{m:02d}'
                date_list.append(date)

        pick_date_list = []
        for y in range(1982, 2015 + 1):
            for m in range(1, 13):
                date = f'{y}-{m:02d}'
                pick_date_list.append(date)

        dic = T.load_npy_dir(fdir)
        picked_vals_dic = {}
        for pix in tqdm(dic):
            vals = dic[pix]
            vals = np.array(vals,dtype=np.float32)
            vals[vals < 0] = np.nan
            if T.is_all_nan(vals):
                continue
            dic_i = dict(zip(date_list, vals))
            picked_vals = []
            for date in pick_date_list:
                val = dic_i[date]
                picked_vals.append(val)
            picked_vals = np.array(picked_vals)
            picked_vals_dic[pix] = picked_vals
        T.save_npy(picked_vals_dic, outf)

    def anomaly(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        outdir = join(self.datadir,'anomaly',global_year_range)
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)

    def detrend(self):
        fdir = join(self.datadir,'anomaly/1982-2015')
        outdir = join(self.datadir,'detrend/1982-2015')
        T.mk_dir(outdir)
        outf = join(outdir,'precip.npy')
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_detrend = T.detrend_dic(spatial_dict)
        T.save_npy(spatial_dict_detrend,outf)

    def check_per_pix(self):
        # fdir = join(self.datadir, 'per_pix', year_range)
        fdir = join(self.datadir, 'anomaly', global_year_range)
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict1 = {}
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            # a,b,r,p = T.nan_line_fit(list(range(len(vals))),vals)
            vals = np.array(vals,dtype=np.float)
            if type(vals) == float:
                continue
            vals[vals<-999] = np.nan
            if T.is_all_nan(vals):
                continue
            # spatial_dict1[pix] = np.mean(vals)
            spatial_dict1[pix] = len(vals)
            # spatial_dict1[pix] = a
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict1)
        plt.imshow(arr)
        plt.show()
        pass

class TMP:
    def __init__(self):
        self.datadir = join(data_root,'CRU_tmp')
        pass

    def run(self):
        # self.check_per_pix()
        self.detrend()
        pass

    def detrend(self):
        fdir = join(self.datadir,'anomaly/1982-2015')
        outdir = join(self.datadir,'detrend')
        T.mk_dir(outdir)
        outf = join(outdir,'detrend.npy')
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_detrend = T.detrend_dic(spatial_dict)
        T.save_npy(spatial_dict_detrend,outf)


    def check_per_pix(self):
        # fdir = join(self.datadir, 'per_pix', year_range)
        fdir = join(self.datadir, 'anomaly', global_year_range)
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict1 = {}
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            a,b,r,p = T.nan_line_fit(list(range(len(vals))),vals)
            # vals = np.array(vals)
            # vals[vals<-999] = np.nan
            # if T.is_all_nan(vals):
            #     continue
            # spatial_dict1[pix] = np.mean(vals)
            spatial_dict1[pix] = a
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict1)
        plt.imshow(arr)
        plt.show()
        pass


class VPD:
    '''
    calculate from CRU
    '''
    def __init__(self):
        self.datadir = join(data_root, 'VPD')
        pass

    def run(self):
        # self.anomaly()
        self.detrend()
        # self.check_per_pix()
        pass

    def anomaly(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        outdir = join(self.datadir,'anomaly',global_year_range)
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)
        pass

    def detrend(self):
        fdir = join(self.datadir,'anomaly',global_year_range)
        outdir = join(self.datadir,'detrend',global_year_range)
        T.mk_dir(outdir,force=True)
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_detrend = T.detrend_dic(spatial_dict)
        outf = join(outdir,'VPD.npy')
        T.save_npy(spatial_dict_detrend,outf)

    def check_per_pix(self):
        fdir = join(self.datadir, 'per_pix',global_year_range)
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict1 = {}
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            # a,b,r,p = T.nan_line_fit(list(range(len(vals))),vals)
            vals = np.array(vals)
            vals[vals<0] = np.nan
            if T.is_all_nan(vals):
                continue
            plt.plot(vals)
            plt.show()
            # spatial_dict1[pix] = np.mean(vals)
            # spatial_dict1[pix] = a
            spatial_dict1[pix] = len(vals)
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict1)
        plt.imshow(arr)
        plt.show()
        pass

class GLC2000:

    def __init__(self):
        self.datadir = join(data_root,'GLC2000')
        pass

    def run(self):
        self.resample()
        # self.unify()
        self.reclass_lc()
        self.lc_dict_with_number()
        self.show_reclass_lc()
        self.show_lc_dict_with_number()
        pass

    def resample(self):

        tif = join(self.datadir,'glc2000_v1_1.tif')
        outtif = join(self.datadir,'glc2000_v1_1_05_deg.tif')
        ToRaster().resample_reproj(tif,outtif,res=0.5)

    def unify(self):
        tif = join(self.datadir,'glc2000_v1_1_05_deg.tif')
        outtif = join(self.datadir,'glc2000_v1_1_05_deg_unify.tif')
        DIC_and_TIF().unify_raster(tif,outtif)

    def reclass_lc(self):
        outf = join(self.datadir,'reclass_lc_dic2')
        excel = join(self.datadir,'glc2000_Global_Legend.xls')
        tif = join(self.datadir,'glc2000_v1_1_05_deg_unify.tif')
        legend_df = pd.read_excel(excel)
        val_dic = T.df_to_dic(legend_df,'VALUE')
        spatial_dic = DIC_and_TIF().spatial_tif_to_dic(tif)
        reclass_dic = {}
        for pix in spatial_dic:
            val = spatial_dic[pix]
            if np.isnan(val):
                continue
            val = int(val)
            # lc = val_dic[val]['reclass_1']
            lc = val_dic[val]['reclass_2']
            if type(lc) == float:
                continue
            reclass_dic[pix] = lc
        T.save_npy(reclass_dic,outf)

    def lc_dict_with_number(self):
        outf = join(self.datadir,'lc_dict_with_number.npy')
        tif = join(self.datadir,'glc2000_v1_1_05_deg_unify.tif')
        spatial_dic = DIC_and_TIF().spatial_tif_to_dic(tif)
        T.save_npy(spatial_dic,outf)

    def show_reclass_lc(self):
        lc_dict_f = join(self.datadir,'reclass_lc_dic.npy')
        lc_dict = T.load_npy(lc_dict_f)
        lc_list = []
        for pix in lc_dict:
            lc = lc_dict[pix]
            lc_list.append(lc)
        lc_list = list(set(lc_list))
        print(lc_list)

    def show_lc_dict_with_number(self):
        lc_dict_f = join(self.datadir,'lc_dict_with_number.npy')
        lc_dict = T.load_npy(lc_dict_f)
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(lc_dict)
        arr[np.isnan(arr)]=20
        dict_new = DIC_and_TIF().spatial_arr_to_dic(arr)
        T.save_npy(dict_new,lc_dict_f)


class CCI_SM:

    def __init__(self):
        self.datadir = join(data_root,'CCI-SM')
        pass

    def run(self):
        # self.per_pix()
        # self.per_pix_no_nan()
        # self.anomaly()
        self.detrend()
        # self.check_cci_sm()
        pass

    def per_pix(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'per_pix')
        T.mkdir(outdir)
        Pre_Process().data_transform(fdir,outdir)

    def per_pix_no_nan(self):
        fdir = join(self.datadir, 'per_pix')
        outdir = join(self.datadir, 'per_pix_no_nan')
        T.mk_dir(outdir)
        outf = join(outdir,'CCI-SM.npy')
        spatial_dic = T.load_npy_dir(fdir)
        spatial_dic1 = {}
        for pix in tqdm(spatial_dic):
            vals = spatial_dic[pix]
            vals = np.array(vals)
            vals[vals < -999] = np.nan
            if T.is_all_nan(vals):
                continue
            spatial_dic1[pix] = vals
        T.save_npy(spatial_dic1,outf)

    def anomaly(self):
        fdir = join(self.datadir,'per_pix_no_nan')
        outdir = join(self.datadir,'anomaly')
        T.mk_dir(outdir)
        Pre_Process().cal_anomaly(fdir,outdir)


    def detrend(self):
        f = join(self.datadir,'anomaly','CCI-SM.npy')
        outdir = join(self.datadir,'detrend')
        outf = join(outdir,'CCI-SM.npy')
        T.mk_dir(outdir)
        spatial_dict = T.load_npy(f)
        detrend_spatial_dict = T.detrend_dic(spatial_dict)
        T.save_npy(detrend_spatial_dict,outf)
        pass

    def check_cci_sm(self):
        fdir = join(self.datadir, 'anomaly')
        spatial_dic = T.load_npy_dir(fdir)
        spatial_dic1 = {}
        for pix in tqdm(spatial_dic):
            vals = spatial_dic[pix]
            vals = np.array(vals)
            vals[vals<-999] = np.nan
            if T.is_all_nan(vals):
                continue
            # mean = np.nanmean(vals)
            a,b,r,p = T.nan_line_fit(np.arange(len(vals)),vals)
            mean = len(vals)
            spatial_dic1[pix] = a
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic1)
        plt.imshow(arr)
        plt.show()


class VOD_AMSRU:
    '''
    X-band AMSR-E/AMSR2
    '''
    def __init__(self):
        self.datadir = join(data_root,'VOD_Xband')
        pass

    def run(self):
        # self.per_pix()
        # self.anomaly()
        self.detrend()
        pass

    def per_pix(self):
        fdir = join(self.datadir, 'tif_monthly/D')
        outdir = join(self.datadir, 'per_pix/2003-2015')
        T.mk_dir(outdir, force=True)
        Pre_Process().data_transform(fdir, outdir)

    def anomaly(self):
        fdir = join(self.datadir, 'per_pix/2003-2015')
        outdir = join(self.datadir, 'per_pix_anomaly/2003-2015')
        T.mk_dir(outdir, force=True)
        Pre_Process().cal_anomaly(fdir, outdir)

    def detrend(self):
        fdir = join(self.datadir, 'per_pix_anomaly/2003-2015')
        outdir = join(self.datadir, 'per_pix_anomaly_detrend/2003-2015')
        T.mk_dir(outdir, force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir, f)
            outf = join(outdir, f)
            dic = T.load_npy(fpath)
            dic_detrend = T.detrend_dic(dic)
            T.save_npy(dic_detrend, outf)
        pass

class VOD_Kband:
    def __init__(self):
        self.datadir = join(data_root,'VOD_Kband')
        pass

    def run(self):
        # self.unzip()
        # self.nc_to_tif()
        # self.resample()
        # self.monthly_compose()
        # self.move_annual_tif()
        # self.per_pix()
        # self.anomaly()
        self.detrend()
        pass

    def unzip(self):
        '''
        unzip all the zip files in the zips folder
        half hour to finish
        :return:
        '''
        fdir = join(self.datadir,'zips')
        outdir = join(self.datadir,'nc')
        T.unzip(fdir,outdir)

    def nc_to_tif(self):
        fdir = join(self.datadir,'nc')
        outdir = '/Volumes/SSD1T/VOD_K_band/tif'
        T.mk_dir(outdir)
        for year in T.listdir(fdir):
            fdir_i = join(fdir,year,year)
            outdir_i = join(outdir,year)
            T.mk_dir(outdir_i)
            for f in tqdm(T.listdir(fdir_i)):
                fpath = join(fdir_i,f)
                # outf = join(outdir_i,f.replace('.nc','.tif'))
                T.nc_to_tif(fpath,'vod',outdir_i)

    def resample(self):
        fdir = '/Volumes/SSD1T/VOD_K_band/tif'
        outdir = join(self.datadir,'daily_tif_05')
        T.mk_dir(outdir)
        for year in T.listdir(fdir):
            fdir_i = join(fdir,year)
            outdir_i = join(outdir,year)
            T.mk_dir(outdir_i)
            for f in tqdm(T.listdir(fdir_i)):
                fpath = join(fdir_i,f)
                outf = join(outdir_i,f)
                ToRaster().resample_reproj(fpath,outf,0.5)

    def monthly_compose(self):
        fdir = join(self.datadir,'daily_tif_05')
        outdir = join(self.datadir,'monthly_tif_05')
        T.mk_dir(outdir)
        for year in T.listdir(fdir):
            fdir_i = join(fdir,year)
            outdir_i = join(outdir,year)
            T.mk_dir(outdir_i)
            Pre_Process().monthly_compose(fdir_i,outdir_i,method='max')

    def move_annual_tif(self):
        fdir = join(self.datadir,'monthly_tif_05')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir)
        for year in T.listdir(fdir):
            fdir_i = join(fdir,year)
            for f in T.listdir(fdir_i):
                fpath = join(fdir_i,f)
                outf = join(outdir,f)
                shutil.copy(fpath,outf)

    def per_pix(self):
        fdir = join(self.datadir,'tif/1988-2015')
        outdir = join(self.datadir,'per_pix/1988-2015')
        T.mk_dir(outdir,force=True)
        Pre_Process().data_transform(fdir,outdir)

    def anomaly(self):
        fdir = join(self.datadir,'per_pix/1988-2015')
        outdir = join(self.datadir,'anomaly/1988-2015')
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)

    def detrend(self):
        fdir = join(self.datadir,'anomaly/1988-2015')
        outdir = join(self.datadir,'detrend/1988-2015')
        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outf = join(outdir,f)
            dic = T.load_npy(fpath)
            dic_detrend = T.detrend_dic(dic)
            T.save_npy(dic_detrend,outf)
        pass

class CSIF:

    def __init__(self):
        self.datadir = join(data_root,'CSIF')
        pass

    def run(self):
        self.detrend()
        pass

    def detrend(self):
        fdir = join(self.datadir,'per_pix_anomaly/2001-2015')
        outdir = join(self.datadir,'per_pix_anomaly_detrend/2001-2015')
        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outf = join(outdir,f)
            dic = T.load_npy(fpath)
            dic_detrend = T.detrend_dic(dic)
            T.save_npy(dic_detrend,outf)

class Terraclimate:
    def __init__(self):
        self.datadir = join(data_root,'Terraclimate')
        pass

    def run(self):
        # self.nc_to_tif_srad()
        # self.nc_to_tif_aet()
        # self.resample()
        # self.per_pix()
        # self.anomaly()
        self.detrend()
        # self.download_all()
        pass

    def nc_to_tif_srad(self):
        outdir = self.datadir + '/srad/tif/'
        T.mk_dir(outdir,force=True)
        fdir = self.datadir + '/srad/nc11/'
        for fi in T.listdir(fdir):
            print(fi)
            if fi.startswith('.'):
                continue
            f = fdir + fi
            year = fi.split('.')[-2].split('_')[-1]
            # print(year)
            # exit()
            ncin = Dataset(f, 'r')
            ncin_xarr = xr.open_dataset(f)
            # print(ncin.variables)
            # exit()
            lat = ncin['lat']
            lon = ncin['lon']
            pixelWidth = lon[1] - lon[0]
            pixelHeight = lat[1] - lat[0]
            longitude_start = lon[0]
            latitude_start = lat[0]
            time = ncin.variables['time']

            start = datetime.datetime(1900, 1, 1)
            # print(time)
            # for t in time:
            #     print(t)
            # exit()
            flag = 0
            for i in tqdm(range(len(time))):
                # print(i)
                flag += 1
                # print(time[i])
                date = start + datetime.timedelta(days=int(time[i]))
                year = str(date.year)
                # exit()
                month = '%02d' % date.month
                day = '%02d'%date.day
                date_str = year + month
                newRasterfn = outdir + date_str + '.tif'
                if os.path.isfile(newRasterfn):
                    continue
                # print(date_str)
                # exit()
                # if not date_str[:4] in valid_year:
                #     continue
                # print(date_str)
                # exit()
                # arr = ncin.variables['tmax'][i]
                arr = ncin_xarr['srad'][i]
                arr = np.array(arr)
                # print(arr)
                # grid = arr < 99999
                # arr[np.logical_not(grid)] = -999999
                newRasterfn = outdir + date_str + '.tif'
                ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, arr)
                # grid = np.ma.masked_where(grid>1000,grid)
                # DIC_and_TIF().arr_to_tif(arr,newRasterfn)
                # plt.imshow(arr,'RdBu')
                # plt.colorbar()
                # plt.show()
                # nc_dic[date_str] = arr
                # exit()

    def nc_to_tif_aet(self):
        outdir = self.datadir + '/aet/tif/'
        T.mk_dir(outdir,force=True)
        fdir = self.datadir + '/aet/nc/'
        for fi in T.listdir(fdir):
            print(fi)
            if fi.startswith('.'):
                continue
            f = fdir + fi
            year = fi.split('.')[-2].split('_')[-1]
            # print(year)
            # exit()
            ncin = Dataset(f, 'r')
            ncin_xarr = xr.open_dataset(f)
            # print(ncin.variables)
            # exit()
            lat = ncin['lat']
            lon = ncin['lon']
            pixelWidth = lon[1] - lon[0]
            pixelHeight = lat[1] - lat[0]
            longitude_start = lon[0]
            latitude_start = lat[0]
            time = ncin.variables['time']

            start = datetime.datetime(1900, 1, 1)
            # print(time)
            # for t in time:
            #     print(t)
            # exit()
            flag = 0
            for i in tqdm(range(len(time))):
                # print(i)
                flag += 1
                # print(time[i])
                date = start + datetime.timedelta(days=int(time[i]))
                year = str(date.year)
                # exit()
                month = '%02d' % date.month
                day = '%02d'%date.day
                date_str = year + month
                newRasterfn = outdir + date_str + '.tif'
                if os.path.isfile(newRasterfn):
                    continue
                # print(date_str)
                # exit()
                # if not date_str[:4] in valid_year:
                #     continue
                # print(date_str)
                # exit()
                # arr = ncin.variables['tmax'][i]
                arr = ncin_xarr['aet'][i]
                arr = np.array(arr)
                # print(arr)
                # grid = arr < 99999
                # arr[np.logical_not(grid)] = -999999
                newRasterfn = outdir + date_str + '.tif'
                ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, arr)
                # grid = np.ma.masked_where(grid>1000,grid)
                # DIC_and_TIF().arr_to_tif(arr,newRasterfn)
                # plt.imshow(arr,'RdBu')
                # plt.colorbar()
                # plt.show()
                # nc_dic[date_str] = arr
                # exit()

    def resample(self):
        var_i = 'aet'
        # var_i = 'srad'
        fdir = join(self.datadir, f'{var_i}/tif')
        outdir = join(self.datadir, f'{var_i}/tif_05')
        T.mk_dir(outdir, force=True)
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            outpath = join(outdir, f)
            ToRaster().resample_reproj(fpath, outpath, res=0.5)
        pass

    def per_pix(self):
        # var_i = 'aet'
        var_i = 'srad'
        fdir = join(self.datadir, f'{var_i}/tif_05')
        outdir = join(self.datadir, f'{var_i}/per_pix_05')
        T.mk_dir(outdir, force=True)
        Pre_Process().data_transform(fdir, outdir)

    def anomaly(self):
        # var_i = 'aet'
        var_i = 'srad'
        fdir = join(self.datadir, f'{var_i}/per_pix_05')
        outdir = join(self.datadir, f'{var_i}/anomaly')
        T.mk_dir(outdir, force=True)
        Pre_Process().cal_anomaly(fdir, outdir)

    def detrend(self):
        # var_i = 'aet'
        var_i = 'srad'
        fdir = join(self.datadir,f'{var_i}/anomaly')
        outdir = join(self.datadir,f'{var_i}/detrend')
        outf = join(outdir,f'{var_i}.npy')
        T.mk_dir(outdir)
        spatial_dict = T.load_npy_dir(fdir)
        detrend_spatial_dict = T.detrend_dic(spatial_dict)
        T.save_npy(detrend_spatial_dict,outf)
        pass

    def download_all(self):
        param_list = []
        # product_list = ['def','ws','vap','pdsi','pet','ppt','soil','tmax','vpd']
        product_list = ['aet']
        # product_list = ['vpd']
        for product in product_list:
            for y in range(1982, 2021):
                param_list.append([product,str(y)])
                params = [product,str(y)]
                self.download(params)
        # MULTIPROCESS(self.download, param_list).run(process=8, process_or_thread='t')

    def download(self,params):
        product, y = params
        outdir = join(self.datadir,product,'nc')
        T.mk_dir(outdir, force=True)
        url = 'https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_{}_{}.nc'.format(product, y)
        print(url)
        # while 1:
        #     try:
        #         outf = join(outdir, '{}_{}.nc'.format(product, y))
        #         if os.path.isfile(outf):
        #             return None
        #         req = requests.request('GET', url)
        #         content = req.content
        #         fw = open(outf, 'wb')
        #         fw.write(content)
        #         return None
        #     except Exception as e:
        #         print(url, 'error sleep 5s')
        #         time.sleep(5)


class GLEAM_ET:

    def __init__(self):
        self.datadir = data_root + 'GLEAM_ET/'
        T.mk_dir(self.datadir)
        pass


    def run(self):
        # self.nc_to_tif()
        # self.resample()
        # self.tif_to_perpix_1982_2015()
        # self.anomaly()
        self.detrend()
        pass


    def nc_to_tif(self):
        f = join(self.datadir,'nc/Et_1980-2020_GLEAM_v3.5a_MO.nc')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir)
        ncin = Dataset(f, 'r')
        ncin_xarr = xr.open_dataset(f)
        # print(ncin.variables)
        # exit()
        lat = ncin['lat']
        lon = ncin['lon']
        pixelWidth = lon[1] - lon[0]
        pixelHeight = lat[1] - lat[0]
        longitude_start = lon[0]
        latitude_start = lat[0]
        time_obj = ncin.variables['time']
        start = datetime.datetime(1900, 1, 1)
        # print(time)
        # for t in time:
        #     print(t)
        # exit()
        flag = 0
        for i in tqdm(range(len(time_obj))):
            # print(i)
            flag += 1
            # print(time[i])
            date = start + datetime.timedelta(days=int(time_obj[i]))
            year = str(date.year)
            # exit()
            month = '%02d' % date.month
            day = '%02d' % date.day
            date_str = year + month
            newRasterfn = join(outdir,date_str + '.tif')
            if os.path.isfile(newRasterfn):
                continue
            # print(date_str)
            # exit()
            # if not date_str[:4] in valid_year:
            #     continue
            # print(date_str)
            # exit()
            # arr = ncin.variables['pet'][i]
            arr = ncin_xarr.variables['Et'][i]
            arr = np.array(arr)
            arr[arr<0] = np.nan
            arr = arr.T
            # plt.imshow(arr)
            # plt.show()
            # print(arr)
            # grid = arr < 99999
            # arr[np.logical_not(grid)] = -999999
            ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, arr)
            # grid = np.ma.masked_where(grid>1000,grid)
            # DIC_and_TIF().arr_to_tif(arr,newRasterfn)
            # plt.imshow(arr,'RdBu')
            # plt.colorbar()
            # plt.show()
            # nc_dic[date_str] = arr
            # exit()

        pass

    def resample(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_05_deg')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            ToRaster().resample_reproj(fpath,outpath,res=0.5)
        pass

    def tif_to_perpix_1982_2015(self):
        fdir = join(self.datadir,'tif_05_deg')
        outdir = join(self.datadir,'perpix/1982-2015')
        T.mk_dir(outdir,force=True)
        selected_tif_list = []
        for y in range(1982,2016):
            for m in range(1,13):
                f = '{}{:02d}.tif'.format(y,m)
                selected_tif_list.append(f)
        Pre_Process().data_transform_with_date_list(fdir,outdir,selected_tif_list)

    def anomaly(self):
        fdir = join(self.datadir, 'perpix/1982-2015')
        outdir = join(self.datadir, 'anomaly/1982-2015')
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)
        pass

    def detrend(self):
        fdir = join(self.datadir, 'anomaly/1982-2015')
        outdir = join(self.datadir, 'detrend/1982-2015')
        T.mk_dir(outdir,force=True)
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_detrend = T.detrend_dic(spatial_dict)
        outf = join(outdir,'GLEAM_ET.npy')
        T.save_npy(spatial_dict_detrend,outf)
        pass

class ERA_SM:

    def __init__(self):
        self.datadir = data_root + 'ERA-SM/'

    def run(self):
        # self.download_sm()
        # self.nc_to_tif()
        # self.resample()
        # self.clean()
        # self.tif_to_perpix_1982_2015()
        # self.anomaly()
        # self.detrend()
        self.check_cci_sm()
        pass

    def download_sm(self):
        from ecmwfapi import ECMWFDataServer
        server = ECMWFDataServer()
        outdir = join(self.datadir,'nc')
        outf = join(outdir,'ERA_SM.nc')
        date_list = []
        for y in range(1982, 2016):
            for m in range(1, 13):
                date = '{}{:02d}{:02d}'.format(y, m, 1)
                date_list.append(date)
        date_str = '/'.join(date_list)
        # print date_str
        # exit()
        server.retrieve({
            "class": "ei",
            "dataset": "interim",
            "date": date_str,
            "expver": "1",
            "grid": "0.25/0.25",
            "levtype": "sfc",
            "param": "39.128",
            "stream": "moda",
            "type": "an",
            "target": outf,
            "format": "netcdf",
        })

        pass

    def nc_to_tif(self):
        f = join(self.datadir, 'nc/ERA_025.nc')
        outdir = join(self.datadir, 'tif')
        T.mk_dir(outdir)
        T.nc_to_tif(f,'swvl1',outdir)

    def resample(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_05')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            ToRaster().resample_reproj(fpath,outpath,res=0.5)

    def clean(self):
        fdir = join(self.datadir,'tif_05')
        outdir = join(self.datadir,'tif_05_clean')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            if not f.endswith('.tif'):
                continue
            outpath = join(outdir,f)
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array[array<=0] = np.nan
            ToRaster().array2raster(outpath, originX, originY, pixelWidth, pixelHeight, array)

    def tif_to_perpix_1982_2015(self):
        fdir = join(self.datadir,'tif_05_clean')
        outdir = join(self.datadir,'perpix/1982-2015')
        T.mk_dir(outdir,force=True)
        selected_tif_list = []
        for y in range(1982,2016):
            for m in range(1,13):
                f = '{}{:02d}01.tif'.format(y,m)
                selected_tif_list.append(f)
        Pre_Process().data_transform_with_date_list(fdir,outdir,selected_tif_list)

    def anomaly(self):
        fdir = join(self.datadir, 'perpix/1982-2015')
        outdir = join(self.datadir, 'anomaly/1982-2015')
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)
        pass

    def detrend(self):
        fdir = join(self.datadir,'anomaly/1982-2015')
        outdir = join(self.datadir,'detrend/1982-2015')
        T.mk_dir(outdir,force=True)
        outf = join(outdir,'ERA-SM.npy')
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_detrend = T.detrend_dic(spatial_dict)
        T.save_npy(spatial_dict_detrend,outf)

    def check_cci_sm(self):
        # fdir = join(self.datadir, 'anomaly','1982-2015')
        fdir = join(self.datadir, 'detrend','1982-2015')
        spatial_dic = T.load_npy_dir(fdir)
        spatial_dic1 = {}
        for pix in tqdm(spatial_dic):
            vals = spatial_dic[pix]
            vals = np.array(vals)
            vals[vals<-999] = np.nan
            if T.is_all_nan(vals):
                continue
            mean = np.nanmean(vals)
            # a,b,r,p = T.nan_line_fit(np.arange(len(vals)),vals)
            # mean = len(vals)
            # spatial_dic1[pix] = a
            spatial_dic1[pix] = mean
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic1)
        plt.imshow(arr)
        plt.show()

class GLEAM_SMRoot:

    def __init__(self):
        self.datadir = data_root + 'GLEAM_SMRoot/'
        T.mk_dir(self.datadir)
        pass


    def run(self):
        # self.nc_to_tif()
        # self.resample()
        # self.tif_to_perpix_1982_2015()
        # self.anomaly()
        self.detrend()
        pass


    def nc_to_tif(self):
        f = join(self.datadir,'nc/SMroot_1980-2020_GLEAM_v3.5a_MO.nc')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir)
        ncin = Dataset(f, 'r')
        ncin_xarr = xr.open_dataset(f)
        # print(ncin.variables)
        # exit()
        lat = ncin['lat']
        lon = ncin['lon']
        pixelWidth = lon[1] - lon[0]
        pixelHeight = lat[1] - lat[0]
        longitude_start = lon[0]
        latitude_start = lat[0]
        time_obj = ncin.variables['time']
        start = datetime.datetime(1900, 1, 1)
        # print(time)
        # for t in time:
        #     print(t)
        # exit()
        flag = 0
        for i in tqdm(range(len(time_obj))):
            # print(i)
            flag += 1
            # print(time[i])
            date = start + datetime.timedelta(days=int(time_obj[i]))
            year = str(date.year)
            # exit()
            month = '%02d' % date.month
            day = '%02d' % date.day
            date_str = year + month
            newRasterfn = join(outdir,date_str + '.tif')
            if os.path.isfile(newRasterfn):
                continue
            # print(date_str)
            # exit()
            # if not date_str[:4] in valid_year:
            #     continue
            # print(date_str)
            # exit()
            # arr = ncin.variables['pet'][i]
            arr = ncin_xarr.variables['SMroot'][i]
            arr = np.array(arr)
            arr[arr<0] = np.nan
            arr = arr.T
            # plt.imshow(arr)
            # plt.show()
            # print(arr)
            # grid = arr < 99999
            # arr[np.logical_not(grid)] = -999999
            ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, arr)
            # grid = np.ma.masked_where(grid>1000,grid)
            # DIC_and_TIF().arr_to_tif(arr,newRasterfn)
            # plt.imshow(arr,'RdBu')
            # plt.colorbar()
            # plt.show()
            # nc_dic[date_str] = arr
            # exit()

        pass

    def resample(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_05_deg')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            ToRaster().resample_reproj(fpath,outpath,res=0.5)
        pass

    def tif_to_perpix_1982_2015(self):
        fdir = join(self.datadir,'tif_05_deg')
        outdir = join(self.datadir,'perpix/1982-2015')
        T.mk_dir(outdir,force=True)
        selected_tif_list = []
        for y in range(1982,2016):
            for m in range(1,13):
                f = '{}{:02d}.tif'.format(y,m)
                selected_tif_list.append(f)
        Pre_Process().data_transform_with_date_list(fdir,outdir,selected_tif_list)

    def anomaly(self):
        fdir = join(self.datadir, 'perpix/1982-2015')
        outdir = join(self.datadir, 'anomaly/1982-2015')
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)
        pass

    def detrend(self):
        fdir = join(self.datadir, 'anomaly/1982-2015')
        outdir = join(self.datadir, 'detrend/1982-2015')
        T.mk_dir(outdir,force=True)
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_detrend = T.detrend_dic(spatial_dict)
        outf = join(outdir,'GLEAM_ET.npy')
        T.save_npy(spatial_dict_detrend,outf)
        pass

class ERA_2m_T:

    def __init__(self):
        self.datadir = join(data_root,'ERA_2m_T')
        pass

    def run(self):
        # self.download_data()
        # self.download_monthly()
        # self.nc_to_tif()
        # self.resample()
        # self.perpix()
        self.anomaly()
        pass

    def download_data(self):
        from ecmwfapi import ECMWFDataServer
        server = ECMWFDataServer()
        outdir = join(self.datadir,'nc')
        T.mk_dir(outdir,force=True)
        date_list = []
        init_date = datetime.datetime(1982,1,1)
        flag = 1
        for y in range(1982, 2019):
            for m in range(1, 13):
                # date = '{}{:02d}{:02d}'.format(y, m, 1)
                # date = "1982-01-01/to/1982-01-31"
                start_date_obj = T.month_index_to_date_obj(flag-1, init_date) - datetime.timedelta(days=1)
                end_date_obj = T.month_index_to_date_obj(flag, init_date) - datetime.timedelta(days=1)
                flag += 1
                start_date = start_date_obj.strftime('%Y-%m-%d')
                end_date = end_date_obj.strftime('%Y-%m-%d')
                date_range = f'{start_date}/to/{end_date}'
                # outf = join(outdir, f'{y}{m:02d}.nc')
                # print(date_range)
                date_list.append(date_range)
        for date_range in tqdm(date_list):
            # print(date_range)
            start_date = date_range.split('/')[0]
            end_date = date_range.split('/')[2]
            start_date = start_date.replace('-','')
            end_date = end_date.replace('-','')
            outf = join(outdir, f'{start_date}_{end_date}.nc')
            server.retrieve({
                "class": "ei",
                "dataset": "interim",
                "date": date_range,
                "expver": "1",
                "grid": "0.5/0.5",
                "levtype": "sfc",
                "param": "167.128",
                "step": "12",
                "stream": "oper",
                "time": "12:00:00",
                "type": "fc",
                "target": outf,
                "format": "netcdf",
            })
            # exit()
        pass

    def download_monthly(self):
        import cdsapi
        c = cdsapi.Client()
        outdir = join(self.datadir,'nc', 'monthly')
        T.mk_dir(outdir,force=True)
        date_list = []
        init_date = datetime.datetime(1982,1,1)
        flag = 1
        for y in range(1982, 2016):
            outf = join(outdir, f'{y}.nc')
            c.retrieve(
                'reanalysis-era5-single-levels-monthly-means',
                {
                    'format': 'netcdf',
                    'variable': '2m_temperature',
                    'year': f'{y}',
                    'month': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                    ],
                    'time': '00:00',
                    'product_type': 'monthly_averaged_reanalysis',
                },
                f'{outf}')
            # exit()
        pass

    def nc_to_tif(self):
        fdir = join(self.datadir,'nc','monthly')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir,force=True)

        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            T.nc_to_tif(fpath,'t2m',outdir)
        pass

    def resample(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_05')
        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outf = join(outdir,f)
            ToRaster().resample_reproj(fpath,outf,0.5)

        pass

    def perpix(self):
        fdir = join(self.datadir,'tif_05')
        outdir = join(self.datadir,'perpix')
        T.mk_dir(outdir,force=True)
        Pre_Process().data_transform(fdir,outdir)

    def anomaly(self):
        fdir = join(self.datadir,'perpix')
        outdir = join(self.datadir,'anomaly')
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)

class ERA_Precip:

    def __init__(self):
        self.datadir = join(data_root,'ERA_Precip')
        pass

    def run(self):
        # self.download_monthly()
        # self.nc_to_tif()
        self.scale_offset()
        self.resample()
        self.perpix()
        self.anomaly()
        pass


    def download_monthly(self):
        import cdsapi
        c = cdsapi.Client()
        outdir = join(self.datadir,'nc', 'monthly')
        T.mk_dir(outdir,force=True)
        date_list = []
        init_date = datetime.datetime(1982,1,1)
        flag = 1
        for y in range(1982, 2016):
            outf = join(outdir, f'{y}.nc')
            c.retrieve(
                'reanalysis-era5-single-levels-monthly-means',
                {
                    'format': 'netcdf',
                    'variable': 'total_precipitation',
                    'year': f'{y}',
                    'month': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                    ],
                    'time': '00:00',
                    'product_type': 'monthly_averaged_reanalysis',
                },
                f'{outf}')
            # exit()
        pass
    def nc_to_tif(self):
        fdir = join(self.datadir,'nc','monthly')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir,force=True)

        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            T.nc_to_tif(fpath,'tp',outdir)
        pass

    def resample(self):
        fdir = join(self.datadir,'tif_offset')
        outdir = join(self.datadir,'tif_offset_05')
        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outf = join(outdir,f)
            ToRaster().resample_reproj(fpath,outf,0.5)

        pass

    def scale_offset(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_offset')
        T.mk_dir(outdir,force=True)
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            date = f.split('.')[0]
            year,mon,day = date[:4],date[4:6],date[6:]
            year,mon,day = int(year),int(mon),int(day)
            days = T.number_of_days_in_month(year,mon)
            fpath = join(fdir,f)
            outf = join(outdir,f)
            arr = DIC_and_TIF().spatial_tif_to_arr(fpath)
            arr = arr * 1000 * days
            D = DIC_and_TIF(tif_template=fpath)
            D.arr_to_tif(arr,outf)
        pass

    def perpix(self):
        fdir = join(self.datadir,'tif_offset_05')
        outdir = join(self.datadir,'perpix','1982-2015')
        T.mk_dir(outdir,force=True)
        Pre_Process().data_transform(fdir,outdir)

    def anomaly(self):
        fdir = join(self.datadir,'perpix','1982-2015')
        outdir = join(self.datadir,'anomaly','1982-2015')
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)

class GPCC:

    def __init__(self):
        self.datadir = join(data_root, 'GPCC')
        pass

    def run(self):
        # self.download_monthly()
        # self.nc_to_tif()
        self.perpix()
        self.anomaly()
        pass

    def nc_to_tif(self):
        fdir = join(self.datadir,'nc')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir,force=True)

        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            T.nc_to_tif(fpath,'precip',outdir)
        pass

    def perpix(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'perpix')
        T.mk_dir(outdir,force=True)
        Pre_Process().data_transform(fdir,outdir)

    def anomaly(self):
        fdir = join(self.datadir,'perpix')
        outdir = join(self.datadir,'anomaly')
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)

class BEST:
    # Berkeley Earth Surface Temperatures (BEST)

    def __init__(self):
        self.datadir = join(data_root, 'Berkeley Earth Surface Temperatures')
        pass

    def run(self):
        # self.download_monthly()
        # self.nc_to_tif()
        # self.resample()
        self.perpix()
        # self.anomaly()
        pass

    def nc_to_tif(self):
        fdir = join(self.datadir,'nc')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir,force=True)

        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            self.__nc_to_tif(fpath,'temperature',outdir)
        pass

    def perpix(self):
        fdir = join(self.datadir,'tif_05')
        outdir = join(self.datadir,'perpix')
        T.mk_dir(outdir,force=True)
        Pre_Process().data_transform(fdir,outdir)

    def resample(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_05')
        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outf = join(outdir,f)
            ToRaster().resample_reproj(fpath,outf,0.5)
        pass

    def anomaly(self):
        fdir = join(self.datadir,'perpix')
        outdir = join(self.datadir,'anomaly')
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)


    def __nc_to_tif(self, fname, var_name, outdir):
        try:
            ncin = Dataset(fname, 'r')
            print(ncin.variables.keys())

        except:
            raise UserWarning('File not supported: ' + fname)
        try:
            lat = ncin.variables['lat'][:]
            lon = ncin.variables['lon'][:]
        except:
            try:
                lat = ncin.variables['latitude'][:]
                lon = ncin.variables['longitude'][:]
            except:
                try:
                    lat = ncin.variables['lat_FULL'][:]
                    lon = ncin.variables['lon_FULL'][:]
                except:
                    raise UserWarning('lat or lon not found')
        shape = np.shape(lat)
        try:
            time = ncin.variables['time_counter'][:]
            basetime_str = ncin.variables['time_counter'].units
        except:
            time = ncin.variables['time'][:]
            basetime_str = ncin.variables['time'].units

        basetime_unit = 'month'
        print(basetime_unit)
        print(basetime_str)
        # basetime = basetime_str.strip(f'{timedelta_unit} since ')
        # basetime = '0000-00-00'
        basetime = datetime.datetime(1,1,1)
        data = ncin.variables[var_name]
        if len(shape) == 2:
            xx, yy = lon, lat
        else:
            xx, yy = np.meshgrid(lon, lat)
        for time_i in tqdm(range(len(time))):
            time_str = time[time_i]
            # print(time_str)
            # print(type(time_str))
            ratio = time_str - int(time_str)
            # print(ratio)
            mon = int(ratio * 12) + 1
            year = int(time_str)
            day = 1
            outf_name = f'{year}{mon:02d}{day:02d}.tif'
            outpath = join(outdir, outf_name)
            if isfile(outpath):
                continue
            arr = data[time_i]
            arr = np.array(arr)
            lon_list = xx.flatten()
            lat_list = yy.flatten()
            val_list = arr.flatten()
            lon_list[lon_list > 180] = lon_list[lon_list > 180] - 360
            df = pd.DataFrame()
            df['lon'] = lon_list
            df['lat'] = lat_list
            df['val'] = val_list
            lon_list_new = df['lon'].tolist()
            lat_list_new = df['lat'].tolist()
            val_list_new = df['val'].tolist()
            DIC_and_TIF().lon_lat_val_to_tif(lon_list_new, lat_list_new, val_list_new, outpath)

class GOME2_SIF:
    '''
    ref: Spatially downscaling sun-induced chlorophyll fluorescence leads to an improved temporal correlation with gross primary productivity
    '''
    def __init__(self):
        self.datadir = join(data_root, 'GOME2_SIF')
        pass

    def run(self):
        # self.nc_to_tif()
        # self.resample()
        # self.monthly_compose()
        # self.drop_invalid_value()
        # self.per_pix()
        # self.pick_year_range()
        # self.anomaly()
        self.detrend()
        pass

    def nc_to_tif(self):
        fdir = join(self.datadir,'nc')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outdir_i = join(outdir,f.split('.')[0])
            T.mk_dir(outdir_i,force=True)
            T.nc_to_tif(fpath,'SIF',outdir_i)

    def resample(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_05')
        T.mk_dir(outdir,force=True)
        for folder in T.listdir(fdir):
            fdir_i = join(fdir,folder)
            for f in tqdm(T.listdir(fdir_i),desc=folder):
                fpath = join(fdir_i,f)
                outpath = join(outdir,f)
                ToRaster().resample_reproj(fpath,outpath,0.5)
        pass

    def monthly_compose(self):
        fdir = join(self.datadir,'tif_05')
        outdir = join(self.datadir,'tif_mvc_05')
        T.mk_dir(outdir)
        Pre_Process().monthly_compose(fdir,outdir,method='max')
        pass

    def drop_invalid_value(self):
        '''
        nan value: -32768
        :return:
        '''
        fdir = join(self.datadir,'tif_mvc_05')
        outdir = join(self.datadir,'tif_mvc_05_drop_nan')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array[array<0] = np.nan
            DIC_and_TIF().arr_to_tif(array,outpath)

    def per_pix(self):
        fdir = join(self.datadir,'tif_mvc_05_drop_nan')
        outdir = join(self.datadir,'per_pix/2007-2018')
        T.mk_dir(outdir,force=True)
        Pre_Process().data_transform(fdir,outdir)

    def anomaly(self):
        # year_range = '2007-2018'
        year_range = '2007-2015'
        fdir = join(self.datadir,'per_pix',year_range)
        outdir = join(self.datadir,'anomaly',year_range)
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)

    def detrend(self):
        # year_range = '2007-2018'
        year_range = '2007-2015'
        fdir = join(self.datadir,'anomaly',year_range)
        outdir = join(self.datadir,'detrend',year_range)
        T.mk_dir(outdir,force=True)
        Pre_Process().detrend(fdir,outdir)

    def pick_year_range(self):
        origin_year_range = '2007-2018'
        year_range = '2007-2015'

        fdir = join(self.datadir,'per_pix',origin_year_range)
        year_range_list = []
        outdir = join(self.datadir,'per_pix',year_range)
        T.mk_dir(outdir)
        origin_start_year = int(origin_year_range.split('-')[0])
        origin_end_year = int(origin_year_range.split('-')[1])
        start_year = int(year_range.split('-')[0])
        end_year = int(year_range.split('-')[1])
        date_list = []
        for y in range(origin_start_year,origin_end_year + 1):
            for m in range(1,13):
                date = f'{y}-{m:02d}'
                date_list.append(date)
        pick_date_list = []
        for y in range(start_year, end_year + 1):
            for m in range(1, 13):
                date = f'{y}-{m:02d}'
                pick_date_list.append(date)
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            outf = join(outdir,f)
            dic = T.load_npy(fpath)
            picked_vals_dic = {}
            for pix in tqdm(dic):
                vals = dic[pix]
                dic_i = dict(zip(date_list,vals))
                picked_vals = []
                for date in pick_date_list:
                    val = dic_i[date]
                    picked_vals.append(val)
                picked_vals = np.array(picked_vals)
                picked_vals_dic[pix] = picked_vals
            T.save_npy(picked_vals_dic,outf)


class MODIS_LAI_Yuan:

    def __init__(self):
        self.data_dir = '/Users/liyang/Downloads/lai_monthly_yuanhua/'
        pass

    def run(self):
        # self.nc_to_tif()
        # self.per_pix()
        self.trend()
        pass

    def nc_to_tif(self):
        fdir = join(self.data_dir,'nc')
        outdir = join(self.data_dir,'tif')
        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            year = f.split('.')[1].split('_')[1]
            fpath = join(fdir,f)
            ncin = Dataset(fpath, 'r')
            # print(ncin['time'])
            arrs = ncin['lai'][:]
            t = 1
            for arr in arrs:
                arr = np.array(arr,dtype=float)
                arr[arr<=0] = np.nan
                outf = join(outdir,f'{year}{t:02d}.tif')
                DIC_and_TIF().arr_to_tif(arr,outf)
                t += 1
        pass


    def per_pix(self):
        fdir = join(self.data_dir,'tif')
        outdir = join(self.data_dir,'per_pix')
        T.mk_dir(outdir,force=True)
        Pre_Process().data_transform(fdir,outdir)

    def trend(self):
        fdir = join(self.data_dir,'per_pix')
        outdir = join(self.data_dir,'trend')
        T.mk_dir(outdir,force=True)
        spatial_dict = T.load_npy_dir(fdir)
        mon_list = list(range(5,11))
        for m in tqdm(mon_list):
            trend_dict = {}
            gs = [m]
            for pix in tqdm(spatial_dict):
                vals = spatial_dict[pix]
                vals = T.mask_999999_arr(vals,warning=False)
                if T.is_all_nan(vals):
                    continue
                vals_gs = T.monthly_vals_to_annual_val(vals,gs)
                try:
                    a,b,r,p = T.nan_line_fit(np.arange(len(vals_gs)),vals_gs)
                    trend_dict[pix] = a
                except:
                    continue
            outf = join(outdir,f'{m}_trend.tif')
            DIC_and_TIF().pix_dic_to_tif(trend_dict,outf)


class MODIS_LAI_Chen:

    def __init__(self):
        self.data_dir = '/Volumes/NVME2T/greening_project_redo/data/MODIS_LAI/'
        pass

    def run(self):
        # self.monthly_compose()
        # self.perpix()
        self.trend()
        pass

    def monthly_compose(self):
        fdir = join(self.data_dir,'tif_05')
        outdir = join(self.data_dir,'monthly_compose')
        T.mk_dir(outdir,force=True)
        Pre_Process().monthly_compose(fdir,outdir,date_fmt='doy')

    def perpix(self):
        fdir = join(self.data_dir,'monthly_compose')
        outdir = join(self.data_dir,'per_pix_monthly')
        T.mk_dir(outdir,force=True)
        Pre_Process().data_transform(fdir,outdir)
        pass

    def trend(self):
        fdir = join(self.data_dir,'per_pix_monthly')
        outdir = join(self.data_dir,'trend')
        T.mk_dir(outdir,force=True)
        spatial_dict = T.load_npy_dir(fdir)
        mon_list = list(range(5,11))
        for m in tqdm(mon_list):
            trend_dict = {}
            gs = [m]
            for pix in tqdm(spatial_dict):
                r,c = pix
                if r > 180:
                    continue
                vals = spatial_dict[pix]
                vals = T.mask_999999_arr(vals,warning=False)
                if T.is_all_nan(vals):
                    continue
                # print(len(vals))
                # plt.plot(vals)
                # plt.show()
                vals_gs = T.monthly_vals_to_annual_val(vals,gs)
                try:
                    a,b,r,p = T.nan_line_fit(np.arange(len(vals_gs)),vals_gs)
                    trend_dict[pix] = a
                except:
                    continue
            outf = join(outdir,f'{m}_trend_chen.tif')
            DIC_and_TIF().pix_dic_to_tif(trend_dict,outf)
        pass

def main():
    # GIMMS_NDVI().run()
    # SPEI().run()
    # SPI().run()
    # TMP().run()
    # Precipitation().run()
    # VPD().run()
    # CCI_SM().run()
    # ERA_SM().run()
    # Terraclimate().run()
    # GLC2000().run()
    # CCI_SM().run()
    # VOD_Kband().run()
    # VOD_AMSRU().run()
    # CSIF().run()
    # Terraclimate().run()
    # ERA().run()
    # SPI().run()
    # GLEAM_ET().run()
    # GLEAM_SMRoot().run()
    # ERA_2m_T().run()
    ERA_Precip().run()
    # GPCC().run()
    # BEST().run()
    # GOME2_SIF().run()
    # MODIS_LAI_Yuan().run()
    # MODIS_LAI_Chen().run()

    pass



if __name__ == '__main__':
    main()

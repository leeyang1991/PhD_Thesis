# coding=utf-8
from meta_info import *
result_root_this_script = join(results_root, 'Chapter5/analysis')
import xymap

class GLobal_var:
    def __init__(self):

        pass

    def load_data(self, var_i, year_range=global_year_range):
        data_path_dict = Meta_information().path(year_range)
        data_path = data_path_dict[var_i]['path']
        path_type = data_path_dict[var_i]['path_type']
        if path_type == 'file':
            spatial_dict = T.load_npy(data_path)
        elif path_type == 'dir':
            spatial_dict = T.load_npy_dir(data_path)
        elif path_type == 'multi-files':
            spatial_dict = {}
            for f in T.listdir(data_path):
                print(f'loading {f}')
                key = f.split('.')[0]
                spatial_dict_i = T.load_npy(join(data_path, f))
                spatial_dict[key] = spatial_dict_i
        else:
            raise ValueError('path_type not recognized')
        return spatial_dict

    def load_df(self):
        import statistic
        dff = join(statistic.Dataframe().dff)
        df = T.load_df(dff)
        cols = self.get_rs_rt_cols()
        for col in cols:
            df[df[col] > 1.4] = np.nan
            df[df[col] < 0.6] = np.nan
        df = df[df['lat'] > 30]
        df = df.drop_duplicates()
        return df

    def dff(self):
        import statistic
        dff = join(statistic.Dataframe().dff)
        return dff

    def get_rs_rt_cols(self):
        post_n_list = [1, 2, 3, 4]
        cols = ['rt']
        for n in post_n_list:
            cols.append('rs_{}'.format(n))
        return cols


class Water_energy_limited_area:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Water_energy_limited_area',
                                                                                       result_root_this_script, mode=2)
        pass

    def run(self):
        # self.kendall_corr_all_vars()
        self.Ecosystem_Limited_Index_p()
        # self.Ecosystem_Limited_Index()
        pass

    def kendall_corr_all_vars(self):
        outdir = join(self.this_class_arr, 'kendall_corr')
        var_1 = 'GLEAM-ET'
        var_21 = 'ERA-SM'
        var_22 = 'CCI-SM'
        var_3 = 'Temperature'
        var_31 = 'Radiation'
        # self.kendall_corr(var_1,var_21,outdir)
        # self.kendall_corr(var_1,var_22,outdir)
        # self.kendall_corr(var_1,var_3,outdir)
        self.kendall_corr(var_1, var_31, outdir)

    def kendall_corr(self, var_1, var_2, outdir):
        T.mk_dir(outdir)
        outf = join(outdir, f'{var_1}_{var_2}.df')
        spatial_dict_1 = GLobal_var().load_data(var_1)
        spatial_dict_2 = GLobal_var().load_data(var_2)
        spatial_dict_corr = {}
        spatial_dict_corr_p = {}
        for pix in tqdm(spatial_dict_1):
            if not pix in spatial_dict_2:
                continue
            val1 = spatial_dict_1[pix]
            val2 = spatial_dict_2[pix]
            r, p = T.nan_correlation(val1, val2, method='kendall')
            spatial_dict_corr[pix] = r
            spatial_dict_corr_p[pix] = p

        spatial_dict_all = {
            f'{var_1}_{var_2}_r': spatial_dict_corr,
            f'{var_1}_{var_2}_p': spatial_dict_corr_p,
        }
        df = T.spatial_dics_to_df(spatial_dict_all)
        T.save_df(df, outf)
        T.df_to_excel(df, outf)

    def Ecosystem_Limited_Index(self):
        var_1 = 'GLEAM-ET'
        var_21 = 'ERA-SM'
        var_22 = 'CCI-SM'
        # var_3 = 'Temperature'
        var_31 = 'Radiation'
        fdir = join(self.this_class_arr, 'kendall_corr')
        outdir = join(self.this_class_tif, 'ELI')
        T.mk_dir(outdir)
        # outf = join(outdir, f'{var_1}_{var_21}_{var_3}.tif')
        outf = join(outdir, f'{var_1}_{var_21}_{var_31}.tif')
        ELI_equation = 'ELI = corr(ET,SM) - corr(ET,T)'
        dff1 = join(fdir, f'{var_1}_{var_21}.df')
        # dff2 = join(fdir,f'{var_1}_{var_3}.df')
        dff2 = join(fdir, f'{var_1}_{var_31}.df')
        df1 = T.load_df(dff1)
        df2 = T.load_df(dff2)
        spatial_dict1 = T.df_to_spatial_dic(df1, f'{var_1}_{var_21}_r')
        # spatial_dict2 = T.df_to_spatial_dic(df2,f'{var_1}_{var_3}_r')
        spatial_dict2 = T.df_to_spatial_dic(df2, f'{var_1}_{var_31}_r')

        ELI_spatial_dict = {}
        for pix in tqdm(spatial_dict1):
            if not pix in spatial_dict2:
                continue
            val1 = spatial_dict1[pix]
            val2 = spatial_dict2[pix]
            ELI = val1 - val2
            ELI_spatial_dict[pix] = ELI

        DIC_and_TIF().pix_dic_to_tif(ELI_spatial_dict, outf)

    def Ecosystem_Limited_Index_p(self):
        var_1 = 'GLEAM-ET'
        var_21 = 'ERA-SM'
        var_22 = 'CCI-SM'
        var_3 = 'Temperature'
        # var_31 = 'Radiation'
        fdir = join(self.this_class_arr, 'kendall_corr')
        outdir = join(self.this_class_tif, 'ELI')
        T.mk_dir(outdir)
        # outf = join(outdir, f'{var_1}_{var_21}_{var_31}_p.tif')
        outf = join(outdir, f'{var_1}_{var_21}_{var_3}_p.tif')
        ELI_equation = 'ELI = corr(ET,SM) - corr(ET,T)'
        dff1 = join(fdir, f'{var_1}_{var_21}.df')
        dff2 = join(fdir, f'{var_1}_{var_3}.df')
        # dff2 = join(fdir,f'{var_1}_{var_31}.df')
        df1 = T.load_df(dff1)
        df2 = T.load_df(dff2)
        spatial_dict1 = T.df_to_spatial_dic(df1, f'{var_1}_{var_21}_p')
        spatial_dict2 = T.df_to_spatial_dic(df2, f'{var_1}_{var_3}_p')
        # spatial_dict2 = T.df_to_spatial_dic(df2,f'{var_1}_{var_31}_p')

        ELI_spatial_dict = {}
        for pix in tqdm(spatial_dict1):
            if not pix in spatial_dict2:
                continue
            val1 = spatial_dict1[pix]
            val2 = spatial_dict2[pix]
            if val1 > 0.05 or val2 > 0.05:
                ELI = 1
            else:
                ELI = 0
            ELI_spatial_dict[pix] = ELI

        DIC_and_TIF().pix_dic_to_tif(ELI_spatial_dict, outf)


class Water_energy_limited_area_daily:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Water_energy_limited_area_daily',
            result_root_this_script, mode=2)
        pass

    def run(self):
        # self.ELI()
        # self.ELI_perpix()
        self.anomaly()

        pass

    def load_data(self, year, mon):
        year = int(year)
        year = str(year)
        mon = int(mon)
        mon = f'{mon:02d}'

        ET_path = join(data_root, 'GLEAM_daily', 'perpix', year, 'Et', mon)
        SMsurf_path = join(data_root, 'GLEAM_daily', 'perpix', year, 'SMsurf', mon)
        SMroot_path = join(data_root, 'GLEAM_daily', 'perpix', year, 'SMroot', mon)
        T_path = join(data_root, 'ERA_daily_Tair', 'perpix', year, mon)

        ET_dict = T.load_npy_dir(ET_path)
        SMsurf_dict = T.load_npy_dir(SMsurf_path)
        SMroot_dict = T.load_npy_dir(SMroot_path)
        T_dict = T.load_npy_dir(T_path)
        return ET_dict, SMsurf_dict, SMroot_dict, T_dict

    def ELI(self):
        outdir = join(self.this_class_tif, 'ELI')
        T.mk_dir(outdir)
        year_list = list(range(global_start_year, global_end_year + 1))
        mon_list = list(range(1, 13))
        for year in tqdm(year_list):
            for mon in mon_list:
                outf = join(outdir, f'{year}{mon:02d}.tif')
                ET_dict, SMsurf_dict, SMroot_dict, T_dict = self.load_data(year, mon)
                spatial_dict = {}
                for pix in ET_dict:
                    r, c = pix
                    if r > 180:
                        continue
                    if not pix in SMsurf_dict or not pix in SMroot_dict or not pix in T_dict:
                        continue
                    ET = ET_dict[pix]
                    SMsurf = SMsurf_dict[pix]
                    # SMroot = SMroot_dict[pix]
                    Temp = T_dict[pix]
                    if T.is_all_nan(ET):
                        continue
                    ET_sm_corr, _ = T.nan_correlation(ET, SMsurf)
                    ET_Temp_corr, _ = T.nan_correlation(ET, Temp)
                    ELI = ET_sm_corr - ET_Temp_corr
                    spatial_dict[pix] = ELI
                arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
                DIC_and_TIF().arr_to_tif(arr, outf)

    def ELI_perpix(self):
        fdir = join(self.this_class_tif, 'ELI')
        outdir = join(self.this_class_arr, 'ELI', 'perpix', '1982-2015')
        T.mk_dir(outdir, force=1)
        Pre_Process().data_transform(fdir, outdir)

        pass

    def anomaly(self):
        fdir = join(self.this_class_arr, 'ELI', 'perpix', '1982-2015')
        outdir = join(self.this_class_arr, 'ELI', 'anomaly', '1982-2015')
        T.mk_dir(outdir, force=1)
        Pre_Process().cal_anomaly(fdir, outdir)

        pass


class Growing_season:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Growing_season',
                                                                                       result_root_this_script, mode=2)
        pass

    def run(self):
        # self.longterm_growing_season()
        self.peak_month_tif()
        pass

    def longterm_growing_season(self):
        # transmit from hot drought project, calculated via HANTS NDVI
        fdir = join(self.this_class_arr, 'longterm_growing_season')
        dff = join(fdir, 'longterm_growing_season.df')
        df = T.load_df(dff)
        gs_dict = T.df_to_spatial_dic(df, 'gs')
        # return gs_dict
        return None

    def peak_growing_season(self):
        dff = join(self.this_class_arr, 'longterm_mean_phenology', 'longterm_mean_phenology.df')
        df = T.load_df(dff)
        gs_dict = T.df_to_spatial_dic(df, 'peak_mon')
        return gs_dict

    def peak_month_tif(self):
        gs_dict = self.peak_growing_season()
        outf = join(self.this_class_tif, 'peak_month.tif')
        DIC_and_TIF().pix_dic_to_tif(gs_dict, outf)
        pass


class Max_Scale_and_Lag_correlation_SPEI:
    # supplementary
    # max scale: only multiple scale SPEI vs NDVI
    # max lag: multiple scale SPEI and multiple lag vs NDVI
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Max_Scale_and_Lag_correlation_SPEI',
            result_root_this_script, mode=2)
        pass

    def run(self):
        # self.NDVI_SPEI_correlation_scale_and_lag()
        # self.NDVI_SPEI_max_correlation_scale_and_lag()
        # self.NDVI_SPEI_correlation_only_scale()
        # self.NDVI_SPEI_max_scale_only_scale()

        # self.NDVI_SPEI_correlation_every_scale_and_month()
        # self.compose_scale_month_to_df()
        self.max_scale_and_month()

        # self.scale_lag_bivariate_plot()
        pass

    def NDVI_SPEI_correlation_scale_and_lag(self):
        '''
        annual GS NDVI and SPEI correlation
        :return:
        '''
        outdir = join(self.this_class_arr, 'NDVI_SPEI_correlation_scale_and_lag')
        T.mk_dir(outdir)
        outf = join(outdir, 'NDVI_SPEI_correlation.df')
        lag_list = list(range(5))

        # gs_dict = Growing_season().longterm_growing_season()
        NDVI_spatial_dict = GLobal_var().load_data('NDVI')
        SPEI_spatial_dicts = GLobal_var().load_data('SPEI')
        dict_all = {}
        for lag in lag_list:
            for scale in SPEI_spatial_dicts:
                SPEI_spatial_dict = SPEI_spatial_dicts[scale]
                scale = scale.split('.')[0]
                correlation_spatial_dict = {}
                for pix in tqdm(NDVI_spatial_dict, desc=f'lag:{lag},scale:{scale}'):
                    # if not pix in gs_dict:
                    #     continue
                    if not pix in SPEI_spatial_dict:
                        continue
                    # gs = gs_dict[pix]
                    ndvi = NDVI_spatial_dict[pix]
                    spei = SPEI_spatial_dict[pix]
                    spei = T.detrend_vals(spei)
                    gs = list(gs)
                    # ndvi_annual = T.monthly_vals_to_annual_val(ndvi,gs)
                    # spei_annual = T.monthly_vals_to_annual_val(spei,gs)
                    ndvi_gs = T.pick_gs_monthly_data(ndvi, gs)
                    spei_gs = T.pick_gs_monthly_data(spei, gs)
                    r, p = T.lag_correlation(spei_gs, ndvi_gs, lag * len(gs), method='spearman')
                    correlation_spatial_dict[pix] = r
                key = '{}-lag{}'.format(scale, lag)
                dict_all[key] = correlation_spatial_dict
        df = T.spatial_dics_to_df(dict_all)
        T.save_df(df, outf)
        T.df_to_excel(df, outf)

    def NDVI_SPEI_max_correlation_scale_and_lag(self):
        outdir = join(self.this_class_tif, 'NDVI_SPEI_max_correlation_scale_and_lag')
        T.mk_dir(outdir)
        dff = join(self.this_class_arr, 'NDVI_SPEI_correlation_scale_and_lag', 'NDVI_SPEI_correlation.df')
        df = T.load_df(dff)
        # df = df.dropna()
        cols = df.columns.tolist()
        cols.remove('pix')
        # exit()
        max_r = []
        max_lag_list = []
        max_scale_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            dict_i = {}
            for col in cols:
                dict_i[col] = row[col]
            max_key = T.get_max_key_from_dict(dict_i)
            scale, lag = max_key.split('-')
            scale = scale.replace('spei', '')
            scale = int(scale)
            lag = lag.replace('lag', '')
            lag = int(lag)
            max_scale_list.append(scale)
            max_lag_list.append(lag)
            r = dict_i[max_key]
            max_r.append(r)
        df['max_r'] = max_r
        df['max_scale'] = max_scale_list
        df['max_lag'] = max_lag_list

        spatial_dict_r = T.df_to_spatial_dic(df, 'max_r')
        spatial_dict_scale = T.df_to_spatial_dic(df, 'max_scale')
        spatial_dict_lag = T.df_to_spatial_dic(df, 'max_lag')

        outf_r = join(outdir, 'max_r.tif')
        outf_scale = join(outdir, 'max_scale.tif')
        outf_lag = join(outdir, 'max_lag.tif')
        DIC_and_TIF().pix_dic_to_tif(spatial_dict_r, outf_r)
        DIC_and_TIF().pix_dic_to_tif(spatial_dict_scale, outf_scale)
        DIC_and_TIF().pix_dic_to_tif(spatial_dict_lag, outf_lag)

    def NDVI_SPEI_correlation_only_scale(self):
        outdir = join(self.this_class_arr, 'NDVI_SPEI_correlation_only_scale')
        T.mk_dir(outdir)
        outf = join(outdir, 'NDVI_SPEI_correlation.df')

        # gs_dict = Growing_season().longterm_growing_season()
        NDVI_spatial_dict = GLobal_var().load_data('NDVI')
        SPEI_spatial_dicts = GLobal_var().load_data('SPEI')
        dict_all = {}
        for scale in SPEI_spatial_dicts:
            SPEI_spatial_dict = SPEI_spatial_dicts[scale]
            scale = scale.split('.')[0]
            correlation_spatial_dict = {}
            for pix in tqdm(NDVI_spatial_dict, desc=f'scale:{scale}'):
                # if not pix in gs_dict:
                #     continue
                if not pix in SPEI_spatial_dict:
                    continue
                # gs = gs_dict[pix]
                ndvi = NDVI_spatial_dict[pix]
                spei = SPEI_spatial_dict[pix]
                gs = list(gs)
                # ndvi_annual = T.monthly_vals_to_annual_val(ndvi,gs)
                # spei_annual = T.monthly_vals_to_annual_val(spei,gs)
                ndvi_gs = T.pick_gs_monthly_data(ndvi, gs)
                spei_gs = T.pick_gs_monthly_data(spei, gs)
                # r,p = T.lag_correlation(spei_annual,ndvi_annual,0,method='spearman')
                r, p = T.lag_correlation(spei_gs, ndvi_gs, 0, method='spearman')
                correlation_spatial_dict[pix] = r
            key = '{}'.format(scale)
            dict_all[key] = correlation_spatial_dict
        df = T.spatial_dics_to_df(dict_all)
        T.save_df(df, outf)
        T.df_to_excel(df, outf)

    def NDVI_SPEI_max_scale_only_scale(self):
        outdir = join(self.this_class_tif, 'NDVI_SPEI_max_scale_only_scale')
        T.mk_dir(outdir)
        dff = join(self.this_class_arr, 'NDVI_SPEI_correlation_only_scale', 'NDVI_SPEI_correlation.df')
        df = T.load_df(dff)
        # df = df.dropna()
        cols = df.columns.tolist()
        cols.remove('pix')
        # exit()
        max_r = []
        max_scale_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            dict_i = {}
            for col in cols:
                dict_i[col] = row[col]
            scale = T.get_max_key_from_dict(dict_i)
            r = dict_i[scale]

            scale = scale.replace('spei', '')
            scale = int(scale)
            max_scale_list.append(scale)
            max_r.append(r)
        df['max_r'] = max_r
        df['max_scale'] = max_scale_list

        spatial_dict_r = T.df_to_spatial_dic(df, 'max_r')
        spatial_dict_scale = T.df_to_spatial_dic(df, 'max_scale')

        outf_r = join(outdir, 'max_r.tif')
        outf_scale = join(outdir, 'max_scale.tif')
        DIC_and_TIF().pix_dic_to_tif(spatial_dict_r, outf_r)
        DIC_and_TIF().pix_dic_to_tif(spatial_dict_scale, outf_scale)

    def NDVI_SPEI_correlation_every_scale_and_month(self):
        import preprocess
        outdir = join(self.this_class_arr, 'NDVI_SPEI_correlation_every_scale_and_month')
        T.mk_dir(outdir)
        NDVI_monthly_dir = join(preprocess.GIMMS_NDVI().datadir, 'every_month', global_year_range)
        SPEI_monthly_dir = join(preprocess.SPEI().datadir, 'every_month', global_year_range)

        params_list = []
        for scale in T.listdir(SPEI_monthly_dir):
            fdir_i = join(SPEI_monthly_dir, scale)
            for f in T.listdir(fdir_i):
                params = [f, scale, outdir, NDVI_monthly_dir, fdir_i]
                # self.kernel_correlation(params)
                params_list.append(params)
        MULTIPROCESS(self.kernel_correlation, params_list).run(process=7)

    def kernel_correlation(self, param):
        f, scale, outdir, NDVI_monthly_dir, fdir_i = param
        mon = f.split('.')[0]
        outfname = f'{scale}_{mon}.npy'
        outf = join(outdir, outfname)
        spei_f = join(fdir_i, f)
        ndvi_f = join(NDVI_monthly_dir, mon + '.npy')
        spei_dict = T.load_npy(spei_f)
        ndvi_dict = T.load_npy(ndvi_f)
        corr_dict = {}
        for pix in spei_dict:
            if not pix in ndvi_dict:
                continue
            spei = spei_dict[pix]
            ndvi = ndvi_dict[pix]
            # r, p = T.nan_correlation(spei, ndvi, method='spearman')
            r, p = T.nan_correlation(spei, ndvi, method='pearson')
            corr_dict[pix] = r
        T.save_npy(corr_dict, outf)
        pass

    def compose_scale_month_to_df(self):
        fdir = join(self.this_class_arr, 'NDVI_SPEI_correlation_every_scale_and_month', 'pearson')
        outdir = join(self.this_class_arr, 'compose_scale_month_to_df', 'pearson')
        T.mk_dir(outdir, force=True)
        all_spatial_dict = {}
        for f in tqdm(T.listdir(fdir), desc='loading'):
            scale = f.split('_')[0]
            mon = f.split('_')[1].split('.')[0]
            scale = scale.replace('spei', '')
            key = f'{scale}_{mon}'
            spatial_dict = T.load_npy(join(fdir, f))
            all_spatial_dict[key] = spatial_dict
        df = T.spatial_dics_to_df(all_spatial_dict)
        outf = join(outdir, 'dataframe.df')
        T.save_df(df, outf)
        T.df_to_excel(df, outf)

    def max_scale_and_month(self):
        # method = 'pearson'
        method = 'spearman'
        valid_mon_list = ['04', '05', '06', '07', '08', '09', '10']
        valid_mon_list = set(valid_mon_list)
        outdir = join(self.this_class_tif, 'max_scale_and_month', method)
        T.mk_dir(outdir, force=True)
        dff = join(self.this_class_arr, 'compose_scale_month_to_df', method, 'dataframe.df')
        df = T.load_df(dff)
        cols = df.columns.tolist()
        cols.remove('pix')
        max_scale_dict = {}
        max_month_dict = {}
        max_r = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            dict_i = {}
            for col in cols:
                scale, mon = col.split('_')
                if not mon in valid_mon_list:
                    continue
                val = row[col]
                if np.isnan(val):
                    continue
                # val = abs(val)
                dict_i[col] = val
            if len(dict_i) == 0:
                continue
            max_key = T.get_max_key_from_dict(dict_i)
            r = dict_i[max_key]
            scale, mon = max_key.split('_')
            scale = int(scale)
            mon = int(mon)
            max_scale_dict[pix] = scale
            max_month_dict[pix] = mon
            max_r[pix] = r
        outf_max_scale = join(outdir, f'{method}_max_scale.tif')
        outf_max_month = join(outdir, f'{method}_max_month.tif')
        outf_max_r = join(outdir, f'{method}_max_r.tif')
        DIC_and_TIF().pix_dic_to_tif(max_scale_dict, outf_max_scale)
        DIC_and_TIF().pix_dic_to_tif(max_month_dict, outf_max_month)
        DIC_and_TIF().pix_dic_to_tif(max_r, outf_max_r)

    def scale_lag_bivariate_plot(self):
        outdir = join(self.this_class_tif, 'scale_lag_bivariate_plot')
        T.mk_dir(outdir)
        outf = join(outdir, 'spei_lag.tif')
        scale_tif = join(self.this_class_tif, 'NDVI_SPEI_max_scale', 'max_scale.tif')
        lag_tif = join(self.this_class_tif, 'NDVI_SPEI_max_scale', 'max_lag.tif')
        tif1 = scale_tif
        tif2 = lag_tif
        tif1_label = 'SPEI scale'
        tif2_label = 'response lag'
        min1, max1 = 1, 12
        min2, max2 = 0, 4
        xymap.Bivariate_plot().plot_bivariate_map(tif1, tif2, tif1_label, tif2_label, min1, max1, min2, max2, outf,
                                                  n=(5, 5), n_legend=(5, 5), zcmap=None, legend_title='')


class Max_Scale_and_Lag_correlation_SPI:
    # max scale: only multiple scale SPI vs NDVI
    # max lag: multiple scale SPI and multiple lag vs NDVI
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Max_Scale_and_Lag_correlation_SPI',
            result_root_this_script, mode=2)
        pass

    def run(self):
        # individual max scale
        # self.NDVI_SPI_correlation_every_scale_and_month()
        # self.compose_scale_month_to_df()
        # self.max_scale_and_month()

        # individual max lag based on max scale
        ###### with issue
        # self.lag_corr_depends_on_max_scale_and_month()
        # self.max_lag()

        # corresponding max scale, month, and max
        # self.NDVI_SPI_correlation_every_scale_month_and_lag()
        # self.compose_scale_month_lag_to_df()
        # self.max_scale_month_lag()

        # mean max scale, month, and max
        self.mean_max_scale_month_lag()

        # self.scale_lag_bivariate_plot()
        pass

    def NDVI_SPI_correlation_every_scale_and_month(self):
        import preprocess
        outdir = join(self.this_class_arr, 'NDVI_SPI_correlation_every_scale_and_month')
        T.mk_dir(outdir)
        NDVI_monthly_dir = join(preprocess.GIMMS_NDVI().datadir, 'every_month', global_year_range)
        SPEI_monthly_dir = join(preprocess.SPI().datadir, 'every_month', global_year_range)

        params_list = []
        for scale in T.listdir(SPEI_monthly_dir):
            fdir_i = join(SPEI_monthly_dir, scale)
            for f in T.listdir(fdir_i):
                params = [f, scale, outdir, NDVI_monthly_dir, fdir_i]
                # self.kernel_correlation(params)
                params_list.append(params)
        MULTIPROCESS(self.kernel_correlation, params_list).run(process=7)

    def kernel_correlation(self, param):
        f, scale, outdir, NDVI_monthly_dir, fdir_i = param
        mon = f.split('.')[0]
        outfname = f'{scale}_{mon}.npy'
        outf = join(outdir, outfname)
        SPI_f = join(fdir_i, f)
        ndvi_f = join(NDVI_monthly_dir, mon + '.npy')
        SPI_dict = T.load_npy(SPI_f)
        ndvi_dict = T.load_npy(ndvi_f)
        corr_dict = {}
        for pix in SPI_dict:
            if not pix in ndvi_dict:
                continue
            SPI = SPI_dict[pix]
            ndvi = ndvi_dict[pix]
            r, p = T.nan_correlation(SPI, ndvi, method='spearman')
            # r, p = T.nan_correlation(SPI, ndvi, method='pearson')
            corr_dict[pix] = r
        T.save_npy(corr_dict, outf)
        pass

    def compose_scale_month_to_df(self):
        # method = 'spearman'
        method = 'pearson'
        fdir = join(self.this_class_arr, 'NDVI_SPI_correlation_every_scale_and_month', method)
        outdir = join(self.this_class_arr, 'compose_scale_month_to_df', method)
        T.mk_dir(outdir, force=True)
        all_spatial_dict = {}
        for f in tqdm(T.listdir(fdir), desc='loading'):
            scale = f.split('_')[0]
            mon = f.split('_')[1].split('.')[0]
            scale = scale.replace('spi', '')
            key = f'{scale}_{mon}'
            spatial_dict = T.load_npy(join(fdir, f))
            all_spatial_dict[key] = spatial_dict
        df = T.spatial_dics_to_df(all_spatial_dict)
        outf = join(outdir, 'dataframe.df')
        T.save_df(df, outf)
        T.df_to_excel(df, outf)

    def max_scale_and_month(self):
        method = 'pearson'
        # method = 'spearman'
        valid_mon_list = ['04', '05', '06', '07', '08', '09', '10']
        valid_mon_list = set(valid_mon_list)
        outdir = join(self.this_class_tif, 'max_scale_and_month', method)
        T.mk_dir(outdir, force=True)
        dff = join(self.this_class_arr, 'compose_scale_month_to_df', method, 'dataframe.df')
        df = T.load_df(dff)
        cols = df.columns.tolist()
        cols.remove('pix')
        max_scale_dict = {}
        max_month_dict = {}
        max_r = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            dict_i = {}
            for col in cols:
                scale, mon = col.split('_')
                if not mon in valid_mon_list:
                    continue
                val = row[col]
                if np.isnan(val):
                    continue
                # val = abs(val)
                dict_i[col] = val
            if len(dict_i) == 0:
                continue
            max_key = T.get_max_key_from_dict(dict_i)
            r = dict_i[max_key]
            scale, mon = max_key.split('_')
            scale = int(scale)
            mon = int(mon)
            max_scale_dict[pix] = scale
            max_month_dict[pix] = mon
            max_r[pix] = r
        outf_max_scale = join(outdir, f'{method}_max_scale.tif')
        outf_max_month = join(outdir, f'{method}_max_month.tif')
        outf_max_r = join(outdir, f'{method}_max_r.tif')
        DIC_and_TIF().pix_dic_to_tif(max_scale_dict, outf_max_scale)
        DIC_and_TIF().pix_dic_to_tif(max_month_dict, outf_max_month)
        DIC_and_TIF().pix_dic_to_tif(max_r, outf_max_r)

    def lag_corr_depends_on_max_scale_and_month(self):
        method = 'pearson'
        spi_month_dict = self.__spi_scale_and_month_dict()
        ndvi_month_dict = self.__ndvi_scale_and_month_dict()
        outdir = join(self.this_class_arr, 'lag_corr_depends_on_max_scale_and_month')
        # NDVI_monthly_dir = join(preprocess.GIMMS_NDVI().datadir,'every_month',global_year_range)
        # SPI_monthly_dir = join(preprocess.SPI().datadir,'every_month',global_year_range)
        T.mk_dir(outdir, force=True)
        outf = join(outdir, f'{method}_lag_corr.df')
        max_scale_f = join(self.this_class_tif, 'max_scale_and_month', method, f'{method}_max_scale.tif')
        max_month_f = join(self.this_class_tif, 'max_scale_and_month', method, f'{method}_max_month.tif')
        max_scale_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(max_scale_f)
        max_month_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(max_month_f)
        lag_list = list(range(5))

        all_dict = {}
        for lag in lag_list:
            corr_spatial_dict = {}
            for pix in tqdm(max_month_spatial_dict, desc=f'lag {lag}'):
                max_scale = max_scale_spatial_dict[pix]
                max_month = max_month_spatial_dict[pix]
                if np.isnan(max_scale) or np.isnan(max_month):
                    continue
                max_scale = int(max_scale)
                max_month = int(max_month)
                spi_key = f'{max_scale:02d}_{max_month:02d}'
                ndvi_key = f'{max_month:02d}'
                spi_value = spi_month_dict[spi_key][pix]
                ndvi_value = ndvi_month_dict[ndvi_key][pix]
                r, p = T.lag_correlation(spi_value, ndvi_value, lag, method=method)
                corr_spatial_dict[pix] = r
            all_dict[lag] = corr_spatial_dict
        df = T.spatial_dics_to_df(all_dict)
        T.save_df(df, outf)
        T.df_to_excel(df, outf)

    def max_lag(self):
        method = 'pearson'
        outdir = join(self.this_class_tif, 'max_lag', method)
        T.mk_dir(outdir, force=True)
        dff = join(self.this_class_arr, 'lag_corr_depends_on_max_scale_and_month', f'{method}_lag_corr.df')
        df = T.load_df(dff)
        cols = df.columns.tolist()
        cols.remove('pix')
        max_lag_dict = {}
        max_r = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            dict_i = {}
            for col in cols:
                val = row[col]
                if np.isnan(val):
                    continue
                dict_i[col] = val
            if len(dict_i) == 0:
                continue
            max_key = T.get_max_key_from_dict(dict_i)
            r = dict_i[max_key]
            lag = int(max_key)
            max_lag_dict[pix] = lag
            max_r[pix] = r
        outf_max_lag = join(outdir, f'{method}_max_lag.tif')
        outf_max_r = join(outdir, f'{method}_max_r.tif')
        DIC_and_TIF().pix_dic_to_tif(max_lag_dict, outf_max_lag)
        DIC_and_TIF().pix_dic_to_tif(max_r, outf_max_r)

    def lag_corr_depends_on_max_scale_and_monthsss(self):
        import preprocess
        method = 'pearson'
        spi_month_dict = self.__spi_scale_and_month_dict()
        ndvi_month_dict = self.__ndvi_scale_and_month_dict()
        outdir = join(self.this_class_arr, 'lag_corr_depends_on_max_scale_and_month')
        NDVI_monthly_dir = join(preprocess.GIMMS_NDVI().datadir, 'every_month', global_year_range)
        SPI_monthly_dir = join(preprocess.SPI().datadir, 'every_month', global_year_range)
        T.mk_dir(outdir, force=True)
        outf = join(outdir, f'{method}_lag_corr.df')
        max_scale_f = join(self.this_class_tif, 'max_scale_and_month', method, f'{method}_max_scale.tif')
        max_month_f = join(self.this_class_tif, 'max_scale_and_month', method, f'{method}_max_month.tif')
        max_scale_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(max_scale_f)
        max_month_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(max_month_f)
        lag_list = list(range(5))

        all_dict = {}
        for lag in lag_list:
            corr_spatial_dict = {}
            for pix in tqdm(max_month_spatial_dict, desc=f'lag {lag}'):
                max_scale = max_scale_spatial_dict[pix]
                max_month = max_month_spatial_dict[pix]
                if np.isnan(max_scale) or np.isnan(max_month):
                    continue
                max_scale = int(max_scale)
                max_month = int(max_month)
                spi_key = f'{max_scale:02d}_{max_month:02d}'
                ndvi_key = f'{max_month:02d}'
                spi_value = spi_month_dict[spi_key][pix]
                ndvi_value = ndvi_month_dict[ndvi_key][pix]
                r, p = T.lag_correlation(spi_value, ndvi_value, lag, method=method)
                corr_spatial_dict[pix] = r
            all_dict[lag] = corr_spatial_dict
        df = T.spatial_dics_to_df(all_dict)
        T.save_df(df, outf)
        T.df_to_excel(df, outf)

    def NDVI_SPI_correlation_every_scale_month_and_lag(self):
        import preprocess
        method = 'spearman'
        # method = 'pearson'
        outdir = join(self.this_class_arr, 'NDVI_SPI_correlation_every_scale_month_and_lag', method)
        T.mk_dir(outdir, force=True)
        NDVI_monthly_dir = join(preprocess.GIMMS_NDVI().datadir, 'every_month', global_year_range)
        SPEI_monthly_dir = join(preprocess.SPI().datadir, 'every_month', global_year_range)
        lag_list = list(range(5))
        params_list = []
        for scale in T.listdir(SPEI_monthly_dir):
            fdir_i = join(SPEI_monthly_dir, scale)
            for f in T.listdir(fdir_i):
                for lag in lag_list:
                    param = [f, scale, lag, outdir, fdir_i, NDVI_monthly_dir, method]
                    params_list.append(param)
                    # self.kernel_NDVI_SPI_correlation_every_scale_month_and_lag(param)
        MULTIPROCESS(self.kernel_NDVI_SPI_correlation_every_scale_month_and_lag, params_list).run(process=7)

    def kernel_NDVI_SPI_correlation_every_scale_month_and_lag(self, param):
        f, scale, lag, outdir, fdir_i, NDVI_monthly_dir, method = param
        mon = f.split('.')[0]
        outfname = f'{scale}_{mon}_lag{lag}.npy'
        outf = join(outdir, outfname)
        SPI_f = join(fdir_i, f)
        ndvi_f = join(NDVI_monthly_dir, mon + '.npy')
        SPI_dict = T.load_npy(SPI_f)
        ndvi_dict = T.load_npy(ndvi_f)
        corr_dict = {}
        for pix in SPI_dict:
            if not pix in ndvi_dict:
                continue
            SPI = SPI_dict[pix]
            ndvi = ndvi_dict[pix]
            r, p = T.lag_correlation(SPI, ndvi, lag, method=method)
            corr_dict[pix] = r
        T.save_npy(corr_dict, outf)

        pass

    def compose_scale_month_lag_to_df(self):
        method = 'spearman'
        # method = 'pearson'
        fdir = join(self.this_class_arr, 'NDVI_SPI_correlation_every_scale_month_and_lag', method)
        outdir = join(self.this_class_arr, 'compose_scale_month_lag_to_df', method)
        T.mk_dir(outdir, force=True)
        all_spatial_dict = {}
        for f in tqdm(T.listdir(fdir), desc='loading'):
            scale = f.split('_')[0]
            mon = f.split('_')[1]
            lag = f.split('_')[2].split('.')[0]
            scale = scale.replace('spi', '')
            lag = lag.replace('lag', '')
            key = f'{scale}_{mon}_{lag}'
            spatial_dict = T.load_npy(join(fdir, f))
            all_spatial_dict[key] = spatial_dict
        df = T.spatial_dics_to_df(all_spatial_dict)
        outf = join(outdir, 'dataframe.df')
        T.save_df(df, outf)
        T.df_to_excel(df, outf)

    def max_scale_month_lag(self):
        # method = 'pearson'
        method = 'spearman'
        valid_mon_list = ['04', '05', '06', '07', '08', '09', '10', '11']
        valid_mon_list = set(valid_mon_list)
        # valid_scale_list = list(range(1,13))
        valid_scale_list = list(range(1, 25))
        valid_scale_list = [f'{i:02d}' for i in valid_scale_list]
        outdir = join(self.this_class_tif, 'max_scale_month_lag', method)
        T.mk_dir(outdir, force=True)
        dff = join(self.this_class_arr, 'compose_scale_month_lag_to_df', method, 'dataframe.df')
        print('loading', dff)
        df = T.load_df(dff)
        print('loaded')
        # exit()
        cols = df.columns.tolist()
        cols.remove('pix')
        max_scale_dict = {}
        max_month_dict = {}
        max_lag_dict = {}
        max_r = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            dict_i = {}
            for col in cols:
                scale, mon, lag = col.split('_')
                if not mon in valid_mon_list:
                    continue
                if not scale in valid_scale_list:
                    continue
                val = row[col]
                if np.isnan(val):
                    continue
                # val = abs(val)
                dict_i[col] = val
            if len(dict_i) == 0:
                continue
            max_key = T.get_max_key_from_dict(dict_i)
            r = dict_i[max_key]
            scale, mon, lag = max_key.split('_')
            scale = int(scale)
            mon = int(mon)
            lag = int(lag)
            max_scale_dict[pix] = scale
            max_month_dict[pix] = mon
            max_lag_dict[pix] = lag
            max_r[pix] = r
        outf_max_scale = join(outdir, f'{method}_max_scale.tif')
        outf_max_month = join(outdir, f'{method}_max_month.tif')
        outf_max_lag = join(outdir, f'{method}_max_lag.tif')
        outf_max_r = join(outdir, f'{method}_max_r.tif')
        DIC_and_TIF().pix_dic_to_tif(max_scale_dict, outf_max_scale)
        DIC_and_TIF().pix_dic_to_tif(max_month_dict, outf_max_month)
        DIC_and_TIF().pix_dic_to_tif(max_lag_dict, outf_max_lag)
        DIC_and_TIF().pix_dic_to_tif(max_r, outf_max_r)

    def mean_max_scale_month_lag(self):
        # method = 'pearson'
        method = 'spearman'
        # valid_mon_list = ['04','05','06','07','08','09','10','11']
        # valid_mon_list = set(valid_mon_list)
        # valid_scale_list = list(range(1,13))
        valid_scale_list = list(range(1, 25))
        valid_scale_list = [f'{i:02d}' for i in valid_scale_list]
        outdir = join(self.this_class_tif, 'mean_max_scale_month_lag', method)
        T.mk_dir(outdir, force=True)
        dff = join(self.this_class_arr, 'compose_scale_month_lag_to_df', method, 'dataframe.df')
        print('loading', dff)
        df = T.load_df(dff)
        print('loaded')
        cols = df.columns.tolist()
        cols.remove('pix')
        max_scale_dict = {}
        max_month_dict = {}
        max_lag_dict = {}
        max_r_dict = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            dict_i = {}
            for col in cols:
                scale, mon, lag = col.split('_')
                # if not mon in valid_mon_list:
                #     continue
                if not scale in valid_scale_list:
                    continue
                val = row[col]
                if np.isnan(val):
                    continue
                # val = abs(val)
                dict_i[col] = val
            if len(dict_i) == 0:
                continue
            # print(dict_i)
            scale_dict = {}
            month_dict = {}
            lag_dict = {}
            for key in dict_i:
                scale, mon, lag = key.split('_')
                val = dict_i[key]
                if not scale in scale_dict:
                    scale_dict[scale] = []
                scale_dict[scale].append(val)
                if not mon in month_dict:
                    month_dict[mon] = []
                month_dict[mon].append(val)
                if not lag in lag_dict:
                    lag_dict[lag] = []
                lag_dict[lag].append(val)
            lag_mean_dict = {}
            for lag in lag_dict:
                lag_mean_dict[lag] = np.mean(lag_dict[lag])
            max_lag = T.get_max_key_from_dict(lag_mean_dict)

            month_mean_dict = {}
            for mon in month_dict:
                month_mean_dict[mon] = np.mean(month_dict[mon])
            max_mon = T.get_max_key_from_dict(month_mean_dict)

            scale_mean_dict = {}
            for scale in scale_dict:
                scale_mean_dict[scale] = np.mean(scale_dict[scale])
            max_scale = T.get_max_key_from_dict(scale_mean_dict)

            # print(max_scale,max_mon,max_lag)
            max_scale = int(max_scale)
            max_mon = int(max_mon)
            max_lag = int(max_lag)
            # r = lag_mean_dict[max_lag]

            max_scale_dict[pix] = max_scale
            max_month_dict[pix] = max_mon
            max_lag_dict[pix] = max_lag
            r_list = row[cols].tolist()
            max_r = np.nanmax(r_list)
            max_r_dict[pix] = max_r
        outf_max_scale = join(outdir, f'{method}_max_scale.tif')
        outf_max_month = join(outdir, f'{method}_max_month.tif')
        outf_max_lag = join(outdir, f'{method}_max_lag.tif')
        outf_max_r = join(outdir, f'{method}_max_r.tif')
        DIC_and_TIF().pix_dic_to_tif(max_scale_dict, outf_max_scale)
        DIC_and_TIF().pix_dic_to_tif(max_month_dict, outf_max_month)
        DIC_and_TIF().pix_dic_to_tif(max_lag_dict, outf_max_lag)
        DIC_and_TIF().pix_dic_to_tif(max_r_dict, outf_max_r)

    def __spi_scale_and_month_dict(self):
        import preprocess
        fdir = join(preprocess.SPI().datadir, 'every_month', global_year_range)
        all_spatial_dict = {}
        for folder in tqdm(T.listdir(fdir), desc='loading SPI'):
            scale = folder.replace('spi', '')
            for f in T.listdir(join(fdir, folder)):
                mon = f.split('.')[0]
                key = f'{scale}_{mon}'
                spatial_dict = T.load_npy(join(fdir, folder, f))
                all_spatial_dict[key] = spatial_dict
        return all_spatial_dict

    def __ndvi_scale_and_month_dict(self):
        import preprocess
        fdir = join(preprocess.GIMMS_NDVI().datadir, 'every_month', global_year_range)
        all_spatial_dict = {}
        for f in tqdm(T.listdir(join(fdir)), desc='loading NDVI'):
            mon = f.split('.')[0]
            key = f'{mon}'
            spatial_dict = T.load_npy(join(fdir, f))
            all_spatial_dict[key] = spatial_dict
        return all_spatial_dict

    def scale_lag_bivariate_plot(self):
        outdir = join(self.this_class_tif, 'scale_lag_bivariate_plot')
        T.mk_dir(outdir)
        outf = join(outdir, 'SPI_lag.tif')
        scale_tif = join(self.this_class_tif, 'NDVI_SPI_max_scale', 'max_scale.tif')
        lag_tif = join(self.this_class_tif, 'NDVI_SPI_max_scale', 'max_lag.tif')
        tif1 = scale_tif
        tif2 = lag_tif
        tif1_label = 'SPI scale'
        tif2_label = 'response lag'
        min1, max1 = 1, 12
        min2, max2 = 0, 4
        xymap.Bivariate_plot().plot_bivariate_map(tif1, tif2, tif1_label, tif2_label, min1, max1, min2, max2, outf,
                                                  n=(5, 5), n_legend=(5, 5), zcmap=None, legend_title='')


class Pick_Drought_Events:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Pick_Drought_Events', result_root_this_script, mode=2)

    def run(self):
        # self.pick_normal_drought_events()
        self.pick_normal_hot_events()
        # self.pick_single_events(year_range_str)
        # self.check_drought_events()
        pass

    def pick_normal_hot_events(self):
        outdir = join(self.this_class_arr, 'normal_hot_events')
        T.mk_dir(outdir)
        threshold_quantile = 75
        # gs_dict = Growing_season().longterm_growing_season()
        t_anomaly_dic = GLobal_var().load_data('Temperature')
        drought_events_dir = join(self.this_class_arr, 'picked_events')
        for f in T.listdir(drought_events_dir):
            scale = f.split('.')[0]
            fpath = join(drought_events_dir, f)
            drought_events_dict = T.load_npy(fpath)
            hot_dic = {}
            normal_dic = {}
            for pix in tqdm(drought_events_dict):
                spi_drought_year = drought_events_dict[pix]
                temp_anomaly = t_anomaly_dic[pix]
                # if not pix in global_gs_dict:
                #     continue
                # gs_mon = global_gs_dict[pix]
                # gs_mon = list(gs_mon)
                gs_mon = Growing_season().constant_growing_season()
                T_annual_val = T.monthly_vals_to_annual_val(temp_anomaly, gs_mon, method='mean')
                T_quantile = np.percentile(T_annual_val, threshold_quantile)
                hot_index_True_False = T_annual_val > T_quantile
                hot_years = []
                for i, val in enumerate(hot_index_True_False):
                    if val == True:
                        hot_years.append(i + global_start_year)
                hot_years = set(hot_years)
                # print(hot_years)
                # exit()
                hot_drought_year = []
                spi_drought_year_spare = []
                for dr in spi_drought_year:
                    if dr in hot_years:
                        hot_drought_year.append(dr)
                    else:
                        spi_drought_year_spare.append(dr)
                hot_dic[pix] = hot_drought_year
                normal_dic[pix] = spi_drought_year_spare
            hot_outf = join(outdir, f'hot-drought_{scale}.npy')
            normal_outf = join(outdir, f'normal-drought_{scale}.npy')
            T.save_npy(hot_dic, hot_outf)
            T.save_npy(normal_dic, normal_outf)

    def pick_normal_drought_events(self):
        outdir = join(self.this_class_arr, 'picked_events')
        T.mk_dir(outdir)
        threshold = -2
        SPI_dict_all = GLobal_var().load_data('SPI')

        for scale in SPI_dict_all:
            SPI_dict = SPI_dict_all[scale]
            events_dic = {}
            params_list = []
            for pix in tqdm(SPI_dict, desc=f'{scale}'):
                vals = SPI_dict[pix]
                vals = np.array(vals)
                params = (vals, threshold)
                params_list.append(params)
                events_list = self.kernel_find_drought_period(params)
                if len(events_list) == 0:
                    continue
                drought_year_list = []
                for drought_range in events_list:
                    min_index = T.pick_min_indx_from_1darray(vals, drought_range)
                    drought_year = min_index // 12 + global_start_year
                    drought_year_list.append(drought_year)
                drought_year_list = np.array(drought_year_list)
                events_dic[pix] = drought_year_list
            outf = join(outdir, '{}'.format(scale))
            T.save_npy(events_dic, outf)

    def kernel_find_drought_period(self, params):
        # 根据不同干旱程度查找干旱时期
        pdsi = params[0]
        threshold = params[1]
        drought_month = []
        for i, val in enumerate(pdsi):
            if val < threshold:  # SPEI
                drought_month.append(i)
            else:
                drought_month.append(-99)
        # plt.plot(drought_month)
        # plt.show()
        events = []
        event_i = []
        for ii in drought_month:
            if ii > -99:
                event_i.append(ii)
            else:
                if len(event_i) > 0:
                    events.append(event_i)
                    event_i = []
                else:
                    event_i = []

        flag = 0
        events_list = []
        # 不取两个端点
        for i in events:
            # 去除两端pdsi值小于-0.5
            if 0 in i or len(pdsi) - 1 in i:
                continue
            new_i = []
            for jj in i:
                new_i.append(jj)
            flag += 1
            vals = []
            for j in new_i:
                try:
                    vals.append(pdsi[j])
                except:
                    print(j)
                    print('error')
                    print(new_i)
                    exit()
            # print(vals)

            # if 0 in new_i:
            # SPEI
            min_val = min(vals)
            if min_val < -99999:
                continue

            events_list.append(new_i)
        return events_list

    def __get_drought_events(self):
        outdir = join(self.this_class_arr, 'drought_events_df')
        T.mk_dir(outdir)
        outf = join(outdir, 'drought_events.df')
        if isfile(outf):
            df = T.load_df(outf)
            return df
        drought_events_dir = join(self.this_class_arr, 'normal_hot_events')
        spatial_dict_all = {}
        for f in T.listdir(drought_events_dir):
            fpath = join(drought_events_dir, f)
            var_i = f.split('.')[0]
            spatial_dict = T.load_npy(fpath)
            spatial_dict_all[var_i] = spatial_dict
        df = T.spatial_dics_to_df(spatial_dict_all)
        T.save_df(df, outf)
        T.df_to_excel(df, outf)
        return df
        pass

    def check_drought_events(self):
        drought_events_df = self.__get_drought_events()
        T.print_head_n(drought_events_df, 10)
        pass

    def pick_single_events(self, year_range_str):
        outdir = join(self.this_class_arr, 'pick_single_events/{}'.format(year_range_str))
        T.mk_dir(outdir, force=True)
        outf = join(outdir, 'single_events.df')
        hot_drought_f = join(self.this_class_arr, f'picked_events/spi_and_hot_12/{year_range_str}/hot_drought_12.npy')
        spi_drought_f = join(self.this_class_arr, f'picked_events/spi_and_hot_12/{year_range_str}/spi_drought_12.npy')
        hot_drought_dic = T.load_npy(hot_drought_f)
        spi_drought_dic = T.load_npy(spi_drought_f)
        pix_list = DIC_and_TIF().void_spatial_dic()
        spatial_dic_dic = {}
        for pix in pix_list:
            spatial_dic_dic[pix] = {}
        for pix in pix_list:
            if not pix in hot_drought_dic:
                continue
            spatial_dic_dic[pix]['hot_drought'] = hot_drought_dic[pix]

        for pix in pix_list:
            if not pix in spi_drought_dic:
                continue
            spatial_dic_dic[pix]['dry_drought'] = spi_drought_dic[pix]
        single_events_spatial_dic = {}
        for pix in tqdm(spatial_dic_dic):
            dic = spatial_dic_dic[pix]
            if len(dic) == 0:
                continue
            drought_years_list = []
            for dtype in dic:
                drought_years = dic[dtype]
                for year in drought_years:
                    drought_years_list.append(year)
            # print(dic)
            drought_years_list = T.drop_repeat_val_from_list(drought_years_list)
            drought_years_list.sort()
            # print('drought_years_list',drought_years_list)
            single_events_list = self.__pick_single_events(drought_years_list)
            # print('single_events_list',single_events_list)
            single_events_dic = {}
            for dtype in dic:
                drought_years = dic[dtype]
                single_event = []
                for year in single_events_list:
                    if year in drought_years:
                        single_event.append(year)
                single_event = np.array(single_event, dtype=int)
                if len(single_event) == 0:
                    single_events_dic[dtype] = np.nan
                else:
                    single_events_dic[dtype] = single_event
            single_events_spatial_dic[pix] = single_events_dic
        df = T.dic_to_df(single_events_spatial_dic, 'pix')
        # self.shasum_variable(df)
        # exit()

        col_list = df.columns.to_list()
        col_list.remove('pix')
        df = df.dropna(how='all', subset=col_list)
        T.save_df(df, outf)
        T.df_to_excel(df, outf)

    def shasum_variable(self, variable):
        readable_hash = hashlib.sha256(str(variable).encode('ascii')).hexdigest()
        print(readable_hash)
        return readable_hash

    def __pick_single_events(self, drought_year_list):
        n = 4
        single_events_list = []
        for i in range(len(drought_year_list)):
            year = drought_year_list[i]
            if i - 1 < 0:  # first drought event
                if len(drought_year_list) == 1:
                    single_events_list.append(year)
                    break
                if year + n <= drought_year_list[i + 1]:
                    single_events_list.append(year)
                continue
            if i + 1 >= len(drought_year_list):  # the last drought event
                if drought_year_list[i] - drought_year_list[i - 1] >= n:
                    single_events_list.append(drought_year_list[i])
                break
            if drought_year_list[i] - drought_year_list[i - 1] >= n and drought_year_list[i] + n <= drought_year_list[
                i + 1]:  # middle drought events
                single_events_list.append(drought_year_list[i])
        return single_events_list


class Resistance_Resilience:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Resistance_Resilience', result_root_this_script, mode=2)
        T.mkdir(join(self.this_class_arr, 'dataframe'))
        self.dff = join(self.this_class_arr, 'dataframe/dataframe.df')
        pass

    def run(self):
        # self.check_lag_and_scale()
        # self.gen_dataframe()
        df = self.__gen_df_init()
        # df = self.add_max_lag_and_scale(df)
        df = self.add_max_r(df)
        # df = self.cal_rt(df)
        # df = self.cal_rs(df)
        # # self.rt_tif(df)
        #
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)

        pass

    def __gen_df_init(self):
        if not os.path.isfile(self.dff):
            df = pd.DataFrame()
            T.save_df(df, self.dff)
            return df
        else:
            df, dff = self.__load_df()
            return df

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        print('len(df):', len(df))
        return df, dff

    def gen_dataframe(self):
        drought_envents_df = self.__get_drought_events()
        drought_envents_df_dict = T.df_to_dic(drought_envents_df, 'pix')
        pix_list = []
        drought_type_with_scale_list = []
        drought_type_list = []
        drought_year_list = []
        for pix in tqdm(drought_envents_df_dict):
            dict_i = drought_envents_df_dict[pix]
            for drought_type_with_scale in dict_i:
                if drought_type_with_scale == 'pix':
                    continue
                drought_year_list_i = dict_i[drought_type_with_scale]
                if type(drought_year_list_i) == float:
                    continue
                drought_type = drought_type_with_scale.split('_')[0]
                for year in drought_year_list_i:
                    pix_list.append(pix)
                    drought_type_with_scale_list.append(drought_type_with_scale)
                    drought_type_list.append(drought_type)
                    drought_year_list.append(year)
        df = pd.DataFrame()
        df['pix'] = pix_list
        df['drought_type'] = drought_type_list
        df['drought_type_with_scale'] = drought_type_with_scale_list
        df['drought_year'] = drought_year_list
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)

    def add_max_lag_and_scale(self, df):
        max_scale_and_lag_df = self.__get_max_scale_and_lag()
        max_lag_spatial_dict = T.df_to_spatial_dic(max_scale_and_lag_df, 'max_lag')
        max_scale_spatial_dict = T.df_to_spatial_dic(max_scale_and_lag_df, 'max_scale')
        # max_r_spatial_dict = T.df_to_spatial_dic(max_scale_and_lag_df, 'max_r')
        print('adding max_scale...')
        df = T.add_spatial_dic_to_df(df, max_scale_spatial_dict, 'max_scale')
        # filter df with max scale
        selected_index = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            max_scale = row['max_scale']
            if np.isnan(max_scale):
                continue
            drought_type = row['drought_type_with_scale']
            max_scale = int(max_scale)
            if f'{max_scale:02d}' in drought_type:
                selected_index.append(i)
        df = df.iloc[selected_index]

        print('adding max_lag...')
        df = T.add_spatial_dic_to_df(df, max_lag_spatial_dict, 'max_lag')
        # df = T.add_spatial_dic_to_df(df, max_r_spatial_dict, 'max_r')

        return df

    def add_max_r(self, df):
        max_scale_and_lag_df = self.__get_max_scale_and_lag()
        max_r_spatial_dict = T.df_to_spatial_dic(max_scale_and_lag_df, 'max_r')
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(max_r_spatial_dict)
        # plt.imshow(arr, cmap='jet',vmin=0,vmax=1)
        # plt.colorbar()
        # plt.show()
        df = T.add_spatial_dic_to_df(df, max_r_spatial_dict, 'max_r')
        return df
        pass

    def cal_rt(self, df):
        gs = global_gs

        NDVI_spatial_dict = GLobal_var().load_data('NDVI-origin')
        # gs_dict = Growing_season().longterm_growing_season()
        year_list = list(range(global_start_year, global_end_year + 1))
        rt_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            # if not pix in gs_dict:
            #     rt_list.append(np.nan)
            #     continue
            if not pix in NDVI_spatial_dict:
                rt_list.append(np.nan)
                continue
            lag = row['max_lag']
            drought_year = row['drought_year']
            # gs = gs_dict[pix]
            # gs = list(gs)
            NDVI = NDVI_spatial_dict[pix]
            NDVI = np.array(NDVI)
            NDVI[NDVI < 0] = np.nan
            NDVI_annual = T.monthly_vals_to_annual_val(NDVI, gs)
            NDVI_annual = T.detrend_vals(NDVI_annual)
            NDVI_annual_dict = dict(zip(year_list, NDVI_annual))
            lagged_drought_year = drought_year + lag
            lagged_drought_year = int(lagged_drought_year)
            if lagged_drought_year > global_end_year:
                rt_list.append(np.nan)
                continue
            NDVI_lagged_drought_year = NDVI_annual_dict[lagged_drought_year]
            long_term_mean = np.nanmean(NDVI_annual)
            rt = NDVI_lagged_drought_year / long_term_mean
            rt_list.append(rt)
        df['rt'] = rt_list
        return df

    def cal_rs(self, df):
        post_n_list = [1, 2, 3, 4]
        # post_n_list = [4]
        gs = global_gs
        NDVI_spatial_dict = GLobal_var().load_data('NDVI-origin')
        # gs_dict = Growing_season().longterm_growing_season()
        year_list = list(range(global_start_year, global_end_year + 1))
        for post_year in post_n_list:
            rs_list = []
            for i, row in tqdm(df.iterrows(), total=len(df), desc=f'cal_rs post_year:{post_year}'):
                pix = row['pix']
                # if not pix in gs_dict:
                #     rs_list.append(np.nan)
                #     continue
                if not pix in NDVI_spatial_dict:
                    rs_list.append(np.nan)
                    continue
                lag = row['max_lag']
                drought_year = row['drought_year']
                # gs = gs_dict[pix]
                # gs = list(gs)
                NDVI = NDVI_spatial_dict[pix]
                NDVI = np.array(NDVI)
                NDVI[NDVI < 0] = np.nan
                NDVI_annual = T.monthly_vals_to_annual_val(NDVI, gs)
                NDVI_annual = T.detrend_vals(NDVI_annual)
                NDVI_annual_dict = dict(zip(year_list, NDVI_annual))
                lagged_drought_year = drought_year + lag
                lagged_drought_year = int(lagged_drought_year)
                if lagged_drought_year > global_end_year:
                    rs_list.append(np.nan)
                    continue
                post_year_list = list(range(lagged_drought_year + 1, lagged_drought_year + 1 + post_year))
                post_year_NDVI = []
                for post_year_i in post_year_list:
                    if post_year_i > global_end_year:
                        post_year_NDVI = []
                        break
                    post_year_NDVI.append(NDVI_annual_dict[post_year_i])
                if len(post_year_NDVI) == 0:
                    rs_list.append(np.nan)
                    continue
                post_year_NDVI_mean = np.nanmean(post_year_NDVI)
                long_term_mean = np.nanmean(NDVI_annual)
                rs = post_year_NDVI_mean / long_term_mean
                rs_list.append(rs)
            df[f'rs_{post_year}'] = rs_list
        return df

    def rt_tif(self, df):
        NDVI_dict = GLobal_var().load_data('NDVI-origin')
        outdir = join(self.this_class_tif, 'rt')
        T.mk_dir(outdir)
        drought_type_list = T.get_df_unique_val_list(df, 'drought_type')
        for drt in drought_type_list:
            df_drt = df[df['drought_type'] == drt]
            spatial_dict = {}
            df_group_dict = T.df_groupby(df_drt, 'pix')
            for pix in tqdm(df_group_dict):
                df_i = df_group_dict[pix]
                rt = df_i['rt'].values
                rt_mean = np.nanmean(rt)
                if rt_mean < 0:
                    NDVI = NDVI_dict[pix]
                    T.print_head_n(df_i, 10)
                    plt.plot(NDVI)
                    plt.show()
                spatial_dict[pix] = rt_mean

            outf = join(outdir, f'{drt}.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict, outf)

    def check_lag_and_scale(self):
        df = self.__get_max_scale_and_lag()
        T.print_head_n(df, 10)
        max_lag_spatial_dict = T.df_to_spatial_dic(df, 'max_lag')
        max_scale_spatial_dict = T.df_to_spatial_dic(df, 'max_scale')
        max_scale_arr = DIC_and_TIF().pix_dic_to_spatial_arr(max_scale_spatial_dict)
        max_lag_arr = DIC_and_TIF().pix_dic_to_spatial_arr(max_lag_spatial_dict)
        plt.imshow(max_scale_arr)
        plt.colorbar()
        plt.title('max_scale')
        plt.figure()
        plt.imshow(max_lag_arr)
        plt.title('max_lag')
        plt.colorbar()
        plt.show()
        exit()
        pass

    def __get_drought_events(self):
        outdir = join(self.this_class_arr, 'drought_events')
        T.mk_dir(outdir)
        outf = join(outdir, 'drought_events.df')
        if isfile(outf):
            df = T.load_df(outf)
            return df
        drought_events_dir = join(Pick_Drought_Events().this_class_arr, 'normal_hot_events')
        spatial_dict_all = {}
        for f in T.listdir(drought_events_dir):
            fpath = join(drought_events_dir, f)
            var_i = f.split('.')[0]
            spatial_dict = T.load_npy(fpath)
            spatial_dict_all[var_i] = spatial_dict
        df = T.spatial_dics_to_df(spatial_dict_all)
        T.save_df(df, outf)
        T.df_to_excel(df, outf)
        return df
        pass

    def __get_max_scale_and_lag(self, method='spearman'):
        outdir = join(self.this_class_arr, 'max_scale_and_lag')
        T.mk_dir(outdir)
        outf = join(outdir, 'max_scale_and_lag.df')
        if isfile(outf):
            df = T.load_df(outf)
            return df
        max_lag_fdir = join(Max_Scale_and_Lag_correlation_SPI().this_class_tif, f'mean_max_scale_month_lag/{method}')
        max_scale_fdir = join(Max_Scale_and_Lag_correlation_SPI().this_class_tif, f'mean_max_scale_month_lag/{method}')
        max_r_fdir = join(Max_Scale_and_Lag_correlation_SPI().this_class_tif, f'mean_max_scale_month_lag/{method}')
        max_lag_f = join(max_lag_fdir, f'{method}_max_lag.tif')
        max_scale_f = join(max_scale_fdir, f'{method}_max_scale.tif')
        max_r_f = join(max_r_fdir, f'{method}_max_r.tif')

        max_lag_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(max_lag_f)
        max_scale_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(max_scale_f)
        max_r_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(max_r_f)

        dict_all = {'max_lag': max_lag_spatial_dict, 'max_scale': max_scale_spatial_dict, 'max_r': max_r_spatial_dict}

        df = T.spatial_dics_to_df(dict_all)
        T.save_df(df, outf)
        T.df_to_excel(df, outf)
        return df


def line_to_shp(inputlist, outSHPfn):
    ############重要#################
    gdal.SetConfigOption("SHAPE_ENCODING", "GBK")
    ############重要#################
    # start,end,outSHPfn,val1,val2,val3,val4,val5
    # _,_,_,_=start[1],start[0],end[0],end[1]

    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outSHPfn):
        shpDriver.DeleteDataSource(outSHPfn)
    outDataSource = shpDriver.CreateDataSource(outSHPfn)
    outLayer = outDataSource.CreateLayer(outSHPfn, geom_type=ogr.wkbLineString)

    # create line geometry
    line = ogr.Geometry(ogr.wkbLineString)

    for i in range(len(inputlist)):
        start = inputlist[i][0]
        end = inputlist[i][1]

        line.AddPoint(start[0], start[1])
        line.AddPoint(end[0], end[1])

        featureDefn = outLayer.GetLayerDefn()
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(line)
        outLayer.CreateFeature(outFeature)
        outFeature.Destroy()
        line = ogr.Geometry(ogr.wkbLineString)
        outFeature = None

    # define the spatial reference, WGS84
    spatialRef = osr.SpatialReference()
    spatialRef.ImportFromEPSG(4326)
    spatialRef.MorphToESRI()
    file = open(outSHPfn[:-4] + '.prj', 'w')
    file.write(spatialRef.ExportToWkt())
    file.close()


def gen_world_grid_shp():
    x_interval = 30
    y_interval = 30

    outf = join(this_root, 'shp/world_line/world_grid.shp')
    lon_list = np.arange(-180, 180, x_interval)
    lat_list = np.arange(-90, 90, y_interval)
    lon_list = np.array(lon_list, dtype=float)
    lat_list = np.array(lat_list, dtype=float)
    # print(lon_list)
    # exit()
    lon_lines = []
    lat_lines = []
    for lon in lon_list:
        lon_lines.append([[lon, -90], [lon, 90]])
    for lat in lat_list:
        lat_lines.append([[-180, lat], [180, lat]])
    lines = lon_lines + lat_lines
    line_to_shp(lines, outf)

    pass


def main():
    # Water_energy_limited_area().run()
    Water_energy_limited_area_daily().run()
    # Growing_season().run()
    # Max_Scale_and_Lag_correlation_SPEI().run()
    # Max_Scale_and_Lag_correlation_SPI().run()
    # Pick_Drought_Events().run()
    # Resistance_Resilience().run()

    # gen_world_grid_shp()
    pass


if __name__ == '__main__':
    main()
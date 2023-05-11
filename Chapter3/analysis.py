# coding=utf-8
import numpy as np

from meta_info import *
result_root_this_script = join(results_root, 'Chapter3/analysis')

class GLobal_var:
    def __init__(self):

        pass

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
        # self.Ecosystem_Limited_Index_p()
        # self.Ecosystem_Limited_Index()
        self.plot_png()
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

    def plot_png(self):
        fpath = join(self.this_class_tif,'ELI','GLEAM-ET_ERA-SM_Temperature.tif')
        outdir = join(self.this_class_png,'GLEAM-ET_ERA-SM_Temperature')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        outf = join(outdir,'GLEAM-ET_ERA-SM_Temperature.jpg')
        plt.figure(figsize=(5, 5))
        ax = plt.subplot(1, 1, 1)
        m,ret = Plot().plot_ortho(fpath, vmin=-0.7, vmax=0.7, ax=ax, cmap=global_cmap_r)
        m.colorbar(ret,location='bottom')
        ax.set_title('ELI')
        plt.savefig(outf, dpi=600)
        plt.close()
        pass


class Water_energy_limited_area_daily:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Water_energy_limited_area_daily',
                                                                                       result_root_this_script, mode=2)
        pass

    def run(self):
        # self.ELI()
        # self.ELI_perpix()
        # self.anomaly()
        # self.trend()
        self.plot_ELI_trend()

        pass

    def load_data(self,year,mon):
        year = int(year)
        year = str(year)
        mon = int(mon)
        mon = f'{mon:02d}'

        ET_path = join(data_root,'GLEAM_daily','perpix',year,'Et',mon)
        SMsurf_path = join(data_root,'GLEAM_daily','perpix',year,'SMsurf',mon)
        SMroot_path = join(data_root,'GLEAM_daily','perpix',year,'SMroot',mon)
        T_path = join(data_root,'ERA_daily_Tair','perpix',year,mon)

        ET_dict = T.load_npy_dir(ET_path)
        SMsurf_dict = T.load_npy_dir(SMsurf_path)
        SMroot_dict = T.load_npy_dir(SMroot_path)
        T_dict = T.load_npy_dir(T_path)
        return ET_dict,SMsurf_dict,SMroot_dict,T_dict

    def ELI(self):
        outdir = join(self.this_class_tif,'ELI')
        T.mk_dir(outdir)
        year_list = list(range(global_start_year,global_end_year+1))
        mon_list = list(range(1,13))
        for year in tqdm(year_list):
            for mon in mon_list:
                outf = join(outdir,f'{year}{mon:02d}.tif')
                ET_dict, SMsurf_dict, SMroot_dict, T_dict = self.load_data(year,mon)
                spatial_dict = {}
                for pix in ET_dict:
                    r,c = pix
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
                    ET_sm_corr,_ = T.nan_correlation(ET,SMsurf)
                    ET_Temp_corr,_ = T.nan_correlation(ET,Temp)
                    ELI = ET_sm_corr - ET_Temp_corr
                    spatial_dict[pix] = ELI
                arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
                DIC_and_TIF().arr_to_tif(arr,outf)


    def ELI_perpix(self):
        fdir = join(self.this_class_tif,'ELI')
        outdir = join(self.this_class_arr,'ELI','perpix','1982-2015')
        T.mk_dir(outdir,force=1)
        Pre_Process().data_transform(fdir,outdir)

        pass

    def anomaly(self):
        fdir = join(self.this_class_arr,'ELI','perpix','1982-2015')
        outdir = join(self.this_class_arr,'ELI','anomaly','1982-2015')
        T.mk_dir(outdir,force=1)
        Pre_Process().cal_anomaly(fdir,outdir)

        pass

    def trend(self):
        fdir = join(self.this_class_arr,'ELI','perpix','1982-2015')
        outdir = join(self.this_class_tif,'trend','1982-2015')
        T.mk_dir(outdir,force=1)
        gs = global_gs
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_a = {}
        spatial_dict_p = {}
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            vals = T.mask_999999_arr(vals,warning=False)
            if T.is_all_nan(vals):
                continue
            annual_gs_vals = T.monthly_vals_to_annual_val(vals,grow_season=gs,method='mean')
            a,b,r,p = T.nan_line_fit(list(range(len(annual_gs_vals))),annual_gs_vals)
            spatial_dict_a[pix] = a
            spatial_dict_p[pix] = p
        outf_a = join(outdir,'ELI_trend.tif')
        outf_p = join(outdir,'ELI_trend_p.tif')
        DIC_and_TIF().pix_dic_to_tif(spatial_dict_a,outf_a)
        DIC_and_TIF().pix_dic_to_tif(spatial_dict_p,outf_p)


    def plot_ELI_trend(self):
        fdir = join(self.this_class_tif,'trend','1982-2015')
        outdir = join(self.this_class_png,'trend','1982-2015')
        T.mk_dir(outdir,force=1)
        trend_f = join(fdir, 'ELI_trend.tif')
        p_f = join(fdir, 'ELI_trend_p.tif')
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111)
        m, ret = Plot().plot_ortho(trend_f, vmin=-0.01, vmax=0.01, ax=ax, cmap=global_cmap_r)
        m = Plot().plot_ortho_significance_scatter(m, p_f, temp_root, ax=ax, s=30,linewidths=1)
        # ax.set_title('ELI trend')
        plt.colorbar(ret, ax=ax, shrink=0.5, location='bottom',pad=0.05)
        outf = join(outdir, 'ELI_trend.png')
        plt.savefig(outf, dpi=600)
        plt.close()
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


class SPEI_trend:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'SPEI_trend',
            result_root_this_script, mode=2)
        pass

    def run(self):
        # self.trend()
        # self.plot_trend()
        # self.dataframe_time_sereis()
        # self.plot_time_sereis()
        self.plot_time_sereis_ltd()
        pass

    def trend(self):
        outdir = join(self.this_class_tif, 'trend')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        gs = global_gs
        spei_scale = global_spei_list
        SPEI_data = Meta_information().load_data('SPEI')

        for scale in spei_scale:
            SPEI_spatial_dict = SPEI_data[scale]
            trend_spatial_dict = {}
            p_spatial_dict = {}
            for pix in tqdm(SPEI_spatial_dict,desc=f'{scale}'):
                SPEI = SPEI_spatial_dict[pix]
                SPEI_gs = T.monthly_vals_to_annual_val(SPEI,gs,method='mean')
                a,b,r,p = T.nan_line_fit(range(len(SPEI_gs)),SPEI_gs)
                trend_spatial_dict[pix] = a
                p_spatial_dict[pix] = p
            arr_trend = DIC_and_TIF().pix_dic_to_spatial_arr(trend_spatial_dict)
            arr_p = DIC_and_TIF().pix_dic_to_spatial_arr(p_spatial_dict)
            outf_trend = join(outdir, f'{scale}_trend.tif')
            outf_p = join(outdir, f'{scale}_trend_p.tif')
            DIC_and_TIF().arr_to_tif(arr_trend, outf_trend)
            DIC_and_TIF().arr_to_tif(arr_p, outf_p)
        pass

    def plot_trend(self):

        fdir = join(self.this_class_tif, 'trend')
        outdir = join(self.this_class_png, 'trend')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        spei_list = global_spei_list

        plt.figure(figsize=(12, 6))
        flag = 1
        for scale in spei_list:
            trend_path = join(fdir, f'{scale}_trend.tif')
            p_path = join(fdir, f'{scale}_trend_p.tif')
            ax = plt.subplot(2, 4, flag)
            # ax = plt.subplot(1, 1, 1)
            flag += 1
            m,ret = Plot().plot_ortho(trend_path,vmin=-0.06,vmax=0.06,ax=ax,cmap=global_cmap)
            # m,ret = Plot().plot_ortho(trend_path,ax=ax,cmap=global_cmap)
            m = Plot().plot_ortho_significance_scatter(m, p_path, temp_root,ax=ax,s=10)
            # save colorbar
            # plt.colorbar(ret,location='bottom')
            # plt.tight_layout()
            # outf = join(outdir, 'trend_colorbar.png')
            # plt.savefig(outf, dpi=600)
            # plt.close()
            # exit()
            #############
            # plt.show()
            # plt.title(f'{scale}')
        plt.tight_layout()
        outf = join(outdir, 'trend.png')
        plt.savefig(outf, dpi=600)
        plt.close()
        # plt.show()
        pass


    def dataframe_time_sereis(self):
        from Chapter3 import statistic
        outdir = join(self.this_class_arr, 'dataframe_time_sereis')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        gs = global_gs
        spei_scale = global_spei_list
        SPEI_data = Meta_information().load_data('SPEI')

        all_dict = {}
        for scale in tqdm(spei_scale):
            SPEI_spatial_dict = SPEI_data[scale]
            all_dict[scale] = SPEI_spatial_dict
        df = T.spatial_dics_to_df(all_dict)
        df = statistic.Dataframe_func(df).df
        outf = join(outdir, 'ts.df')
        T.save_df(df, outf)
        T.df_to_excel(df, outf)


    def plot_time_sereis(self):
        dff = join(self.this_class_arr, 'dataframe_time_sereis', 'ts.df')
        df = T.load_df(dff)

        outdir = join(self.this_class_png, 'time_sereis')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)

        ELI_bins = global_ELI_bins
        spei_scale = ['spei12']

        for scale in spei_scale:
            df_group,bins_list_str = T.df_bin(df, 'ELI', ELI_bins)
            date_list = []
            for year in range(global_start_year, global_end_year+1):
                for month in range(1, 13):
                    date_list.append(f'{year}')
            y_list = []
            matrix = []
            for name,df_group_i in df_group:
                eli = name.left
                print(eli)
                vals = df_group_i[scale].tolist()
                vals = np.array(vals)
                vals_mean = np.nanmean(vals,axis=0)
                vals_mean = vals_mean.tolist()
                matrix.append(vals_mean)
                y_list.append(eli)
            plt.figure(figsize=(12, 4))
            plt.imshow(matrix, cmap=global_cmap, vmin=-0.8, vmax=0.8, aspect='auto')
            plt.colorbar()
            plt.yticks(range(len(y_list))[::4], y_list[::4])
            plt.xticks(range(len(date_list))[3*12:][::12][::5], date_list[3*12:][::12][::5], rotation=90)
            # plt.xlabel('ELI')
            plt.ylabel('ELI')
            plt.title(f'{scale}')
            plt.tight_layout()
            outf = join(outdir, f'{scale}.pdf')
            # plt.savefig(outf)
            # plt.close()
            plt.show()

    def plot_time_sereis_ltd(self):
        dff = join(self.this_class_arr, 'dataframe_time_sereis', 'ts.df')
        df = T.load_df(dff)

        outdir = join(self.this_class_png, 'plot_time_sereis_ltd')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)

        gs = global_gs
        year_list = []
        for year in range(global_start_year, global_end_year + 1):
            year_list.append(year)
        y_list = []
        matrix = []
        ltd_list = global_ELI_class
        for ltd in ltd_list:
            df_ltd = df[df['ELI_class'] == ltd]
            vals = df_ltd['spei12'].tolist()
            vals_gs = []
            for val in tqdm(vals, desc=f'{ltd}'):
                val_gs = T.monthly_vals_to_annual_val(val, grow_season=gs, method='mean')
                vals_gs.append(val_gs)
            vals_gs_mean = np.nanmean(vals_gs, axis=0)
            vals_gs_err = np.nanstd(vals_gs, axis=0)
            vals_gs_err = vals_gs_err / 4.
            plt.figure(figsize=(5, 3))
            plt.plot(year_list, vals_gs_mean)
            plt.xticks(year_list, rotation=90)
            plt.fill_between(year_list, np.array(vals_gs_mean) - np.array(vals_gs_err),
                             np.array(vals_gs_mean) + np.array(vals_gs_err), alpha=0.5)
            plt.title(ltd)
            plt.xlabel('Year')
            plt.ylabel('spei12')
            plt.ylim(-0.9, 0.9)
            plt.tight_layout()
            # plt.show()
            outf = join(outdir, f'{ltd}.pdf')
            plt.savefig(outf)
            plt.close()

        pass


class VIs_trend:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'VIs_trend',
            result_root_this_script, mode=2)
        pass

    def run(self):
        # self.trend()
        # self.plot_trend()
        # self.dataframe_time_sereis()
        # self.plot_time_sereis_ELI()
        # self.plot_time_sereis_ltd()
        self.every_month_time_series()
        self.plot_every_month_time_series()
        pass

    def trend(self):
        outdir = join(self.this_class_tif, 'trend')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        gs = global_gs
        spei_scale = global_spei_list
        VIs_list = ['NDVI-origin']

        # for VI in VIs_list[2:]:
        for VI in VIs_list:
            year_range = global_VIs_year_range_dict[VI]
            VI_spatial_dict = Meta_information().load_data(VI,year_range=year_range)
            trend_spatial_dict = {}
            p_spatial_dict = {}
            for pix in tqdm(VI_spatial_dict,desc=f'{VI}'):
                VI_vals = VI_spatial_dict[pix]
                # print(VI_vals)
                VI_vals = np.array(VI_vals,dtype=np.float)
                VI_vals = VI_vals / 10000
                VI_vals[VI_vals<-9999] = np.nan
                VI_vals_gs = T.monthly_vals_to_annual_val(VI_vals,gs,method='mean')
                try:
                    a,b,r,p = T.nan_line_fit(range(len(VI_vals_gs)),VI_vals_gs)
                except:
                    continue
                trend_spatial_dict[pix] = a
                p_spatial_dict[pix] = p
            arr_trend = DIC_and_TIF().pix_dic_to_spatial_arr(trend_spatial_dict)
            arr_p = DIC_and_TIF().pix_dic_to_spatial_arr(p_spatial_dict)
            outf_trend = join(outdir, f'{VI}_trend.tif')
            outf_p = join(outdir, f'{VI}_trend_p.tif')
            DIC_and_TIF().arr_to_tif(arr_trend, outf_trend)
            DIC_and_TIF().arr_to_tif(arr_p, outf_p)
        pass

    def plot_trend(self):

        fdir = join(self.this_class_tif, 'trend')
        outdir = join(self.this_class_png, 'trend')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        VI = 'NDVI-origin'
        plt.figure(figsize=(8, 8))
        flag = 1
        trend_path = join(fdir, f'{VI}_trend.tif')
        p_path = join(fdir, f'{VI}_trend_p.tif')
        # ax = plt.subplot(1, 3, flag)
        ax = plt.subplot(1, 1, 1)
        flag += 1
        m,ret = Plot().plot_ortho(trend_path,vmin=-0.004,vmax=0.004,ax=ax,cmap=global_cmap)
        # m,ret = Plot().plot_ortho(trend_path,ax=ax,cmap=global_cmap_r)
        m = Plot().plot_ortho_significance_scatter(m, p_path, temp_root,ax=ax,s=20,linewidths=1)
        # save colorbar

        outf = join(outdir, 'trend.png')
        plt.savefig(outf, dpi=600)
        plt.colorbar(ret, location='bottom')
        # plt.show()
        plt.tight_layout()
        outf = join(outdir, 'trend_colorbar.png')
        plt.savefig(outf, dpi=600)

        plt.close()
        pass

    def dataframe_time_sereis(self):
        from Chapter3 import statistic
        outdir = join(self.this_class_arr, 'dataframe_time_sereis')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)

        NDVI_origin_spatial_dict = Meta_information().load_data('NDVI')

        df = T.spatial_dics_to_df({'NDVI':NDVI_origin_spatial_dict})
        df = statistic.Dataframe_func(df).df
        outf = join(outdir, 'ts.df')
        T.save_df(df, outf)
        T.df_to_excel(df, outf)

    def plot_time_sereis_ELI(self):
        dff = join(self.this_class_arr, 'dataframe_time_sereis', 'ts.df')
        df = T.load_df(dff)

        outdir = join(self.this_class_png, 'time_sereis_ELI')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)

        ELI_bins = global_ELI_bins
        gs = global_gs
        df_group,bins_list_str = T.df_bin(df, 'ELI', ELI_bins)
        date_list = []
        for year in range(global_start_year, global_end_year+1):
            for month in range(1,13):
                date_list.append(f'{year}')
        y_list = []
        matrix = []
        for name,df_group_i in df_group:
            eli = name.left
            print(eli)
            vals = df_group_i['NDVI'].tolist()
            vals = np.array(vals)
            # annual_vals = []
            # for val in vals:
            #     val_annual = T.monthly_vals_to_annual_val(val,grow_season=(4,5,6),method='mean')
            #     annual_vals.append(val_annual)
            # annual_vals = np.array(annual_vals)
            vals_mean = np.nanmean(vals,axis=0)
            vals_mean = vals_mean.tolist()
            matrix.append(vals_mean)
            y_list.append(eli)
        plt.figure(figsize=(12, 4))
        plt.imshow(matrix, cmap=global_cmap, vmin=-0.6, vmax=0.6, aspect='auto')
        plt.colorbar()
        plt.yticks(range(len(y_list))[::4], y_list[::4])
        plt.xticks(range(len(date_list))[3*12:][::12][::5], date_list[3*12:][::12][::5], rotation=0)
        # plt.xlabel('ELI')
        plt.ylabel('ELI')
        plt.title('NDVI anomaly')
        plt.tight_layout()
        outf = join(outdir, f'NDVI_anomaly.pdf')
        # plt.savefig(outf)
        # plt.close()
        plt.show()

    def plot_time_sereis_ltd(self):
        dff = join(self.this_class_arr, 'dataframe_time_sereis', 'ts.df')
        df = T.load_df(dff)
        outdir = join(self.this_class_png, 'time_sereis_ltd')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        gs = global_gs
        year_list = []
        for year in range(global_start_year, global_end_year+1):
            year_list.append(year)
        y_list = []
        matrix = []
        ltd_list = global_ELI_class
        for ltd in ltd_list:
            df_ltd = df[df['ELI_class'] == ltd]
            vals = df_ltd['NDVI'].tolist()
            vals_gs = []
            for val in tqdm(vals,desc=f'{ltd}'):
                val_gs = T.monthly_vals_to_annual_val(val, grow_season=gs, method='mean')
                vals_gs.append(val_gs)
            vals_gs_mean = np.nanmean(vals_gs, axis=0)
            vals_gs_err = np.nanstd(vals_gs, axis=0)
            vals_gs_err = vals_gs_err / 2.
            plt.figure(figsize=(5, 3))
            plt.plot(year_list, vals_gs_mean)
            plt.xticks(year_list, rotation=90)
            plt.fill_between(year_list, np.array(vals_gs_mean) - np.array(vals_gs_err),
                             np.array(vals_gs_mean) + np.array(vals_gs_err), alpha=0.5)
            plt.title(ltd)
            plt.xlabel('Year')
            plt.ylabel('NDVI anomaly')
            plt.ylim(-0.7, 0.7)
            plt.tight_layout()
            plt.show()
            # outf = join(outdir, f'{ltd}.pdf')
            # plt.savefig(outf)
            # plt.close()
        pass

    def every_month_time_series(self):
        outdir = join(self.this_class_arr,'every_month_time_series')
        from Chapter3 import statistic
        T.mk_dir(outdir)
        # VIs_list = ['NDVI-origin']
        VIs_list = ['lai3g']
        month_list = list(range(1,13))
        # for VI in VIs_list[2:]:
        for VI in VIs_list:
            # year_range = global_VIs_year_range_dict[VI]
            # VI_spatial_dict = Meta_information().load_data(VI, year_range=year_range)
            data_dir = '/Users/liyang/Desktop/LAI3g'
            VI_spatial_dict = T.load_npy_dir(data_dir)
            monthly_vals_dict_all = {}
            flag = 1
            for pix in tqdm(VI_spatial_dict, desc=f'{VI}'):

                r,c = pix
                if r > 120:
                    continue
                VI_vals = VI_spatial_dict[pix]
                # print(VI_vals)
                VI_vals = np.array(VI_vals, dtype=np.float)
                VI_vals[VI_vals < -0] = np.nan
                # VI_vals = VI_vals / 10000.
                if T.is_all_nan(VI_vals):
                    continue
                monthly_vals_dict = {}
                for mon in month_list:
                    gs = [mon]
                    VI_vals_gs = T.monthly_vals_to_annual_val(VI_vals, gs, method='mean')
                    VI_vals_gs_anomaly = VI_vals_gs - np.nanmean(VI_vals_gs)
                    key = f'{mon:02d}'
                    # monthly_vals_dict[key] = VI_vals_gs
                    monthly_vals_dict[key] = VI_vals_gs_anomaly
                monthly_vals_dict_all[pix] = monthly_vals_dict
                # if flag > 10:
                #     break
                # flag += 1
            df = T.dic_to_df(monthly_vals_dict_all,'pix')
            df = statistic.Dataframe_func(df).df
            outf = join(outdir,f'{VI}.df')
            T.open_path_and_file(outdir)
            T.save_df(df,outf)
            T.df_to_excel(df,outf.replace('.df','.xlsx'))

        pass

    def plot_every_month_time_series(self):
        from Chapter3 import statistic
        # dff = join(self.this_class_arr, 'every_month_time_series', 'NDVI-origin.df')
        dff = join(self.this_class_arr, 'every_month_time_series', 'lai3g.df')
        df = T.load_df(dff)
        # df = statistic.Dataframe_func(df).df
        df = self.__AI_reclass(df)
        # year_list = global_year_range_list
        year_list = list(range(1982,2019))
        AI_class_list = T.get_df_unique_val_list(df,'AI_class_detail')
        # print(AI_class_list)
        flag = 1
        for AI_class in AI_class_list:
            plt.subplot(2,2,flag)
            df_AI = df[df['AI_class_detail'] == AI_class]
            DIC_and_TIF().plot_df_spatial_pix(df_AI,global_land_tif)
            plt.title(AI_class)
            flag += 1
        plt.show()

        for AI_class in AI_class_list:
            # plt.figure()
            df_AI = df[df['AI_class_detail'] == AI_class]
            plt.figure(figsize=(10,8))
            for mon in list(range(1,13)):
                plt.subplot(4,3,mon)
                key = f'{mon:02d}'
                vals = df_AI[key].tolist()
                vals = np.array(vals)
                mean = np.nanmean(vals, axis=0)
                std = np.nanstd(vals, axis=0) / 8.
                plt.plot(year_list,mean,label=key,color='r')
                plt.fill_between(year_list, mean-std, mean+std, alpha=0.2,color='gray')
                plt.legend()
                # plt.xlabel('Year')
                # plt.ylabel('NDVI')
            plt.suptitle(AI_class)
            plt.tight_layout()
        plt.show()

    def __AI_reclass(self,df):
        AI_class = []
        for i,row in df.iterrows():
            val = row['aridity_index']
            if np.isnan(val):
                label = np.nan
                AI_class.append(label)
                continue
            if val >= 0.65:
                label = 'Humid'
                # label = 3
            elif val < 0.2:
                label = 'Arid'
                # label = 0
            elif val >= 0.2 and val < 0.5:
                label = 'Semi Arid'
                # label = 1
            elif val >= 0.5 and val < 0.65:
                label = 'Semi Humid'
            else:
                print(val)
                raise ValueError('AI error')
            AI_class.append(label)
        df['AI_class_detail'] = AI_class
        return df


class VIs_and_SPEI_correlation:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'VIs_and_SPEI_correlation',
            result_root_this_script, mode=2)
        pass


    def run(self):
        # self.cal_correlations()
        # self.tif_correlation()
        # self.reproj()
        # self.land_reproj()
        # self.png_correlation()
        self.png_correlation_3_products()
        # self.tif_max_spei_scale()
        # self.png_max_spei_scale()

        pass

    def cal_correlations(self):
        outdir = join(self.this_class_arr, 'correlation_dataframe')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)

        gs = global_gs
        VIs_list = global_VIs_list
        VIs_year_range_dict = global_VIs_year_range_dict

        for VI in VIs_list:
            outf = join(outdir, f'{VI}.df')
            if os.path.isfile(outf):
                continue
            VI_year_range = VIs_year_range_dict[VI]
            SPEI_data = Meta_information().load_data('SPEI', VI_year_range)
            VI_dict = Meta_information().load_data(VI, VI_year_range)
            result_dict = {}
            for scale in SPEI_data:
                SPEI_dict = SPEI_data[scale]
                spatial_dict_r = {}
                spatial_dict_p = {}
                for pix in tqdm(SPEI_dict,desc=f'{VI}_{scale}'):
                    SPEI = SPEI_dict[pix]
                    if not pix in VI_dict:
                        continue
                    VI_value = VI_dict[pix]
                    if T.is_all_nan(SPEI) or T.is_all_nan(VI_value):
                        continue
                    SPEI_gs = T.monthly_vals_to_annual_val(SPEI, gs, 'array')
                    VI_gs = T.monthly_vals_to_annual_val(VI_value, gs, 'array')
                    SPEI_gs_flatten = SPEI_gs.flatten()
                    VI_gs_flatten = VI_gs.flatten()
                    if not len(SPEI_gs_flatten) == len(VI_gs_flatten):
                        continue
                    r,p = T.nan_correlation(SPEI_gs_flatten,VI_gs_flatten)
                    spatial_dict_r[pix] = r
                    spatial_dict_p[pix] = p
                    key_r = f'{VI}_{scale}_r'
                    key_p = f'{VI}_{scale}_p'
                    result_dict[key_r] = spatial_dict_r
                    result_dict[key_p] = spatial_dict_p
            df = T.spatial_dics_to_df(result_dict)
            T.save_df(df, outf)
            T.df_to_excel(df, outf)

    def tif_correlation(self):
        VIs_list = global_VIs_list
        fdir = join(self.this_class_arr, 'correlation_dataframe')
        for VI in VIs_list:
            outdir_r = join(self.this_class_tif, 'correlation_tif/{}/r'.format(VI))
            outdir_p = join(self.this_class_tif, 'correlation_tif/{}/p'.format(VI))
            T.mk_dir(outdir_r,force=True)
            T.mk_dir(outdir_p,force=True)
            dff = join(fdir,'{}.df'.format(VI))
            df = T.load_df(dff)
            cols = df.columns.tolist()
            cols.remove('pix')
            for col in cols:
                dic = T.df_to_spatial_dic(df, col)
                if col.endswith('_r'):
                    outf = join(outdir_r, f'{col}.tif')
                elif col.endswith('_p'):
                    outf = join(outdir_p, f'{col}.tif')
                else:
                    raise ValueError('col name error')
                DIC_and_TIF().pix_dic_to_tif(dic, outf)

    def ortho_wkt(self):
        wkt = '''
        PROJCRS["North_Pole_Orthographic",
    BASEGEOGCRS["WGS 84",
        DATUM["World Geodetic System 1984",
            ELLIPSOID["WGS 84",6378137,298.257223563,
                LENGTHUNIT["metre",1]]],
        PRIMEM["Greenwich",0,
            ANGLEUNIT["Degree",0.0174532925199433]]],
    CONVERSION["North_Pole_Orthographic",
        METHOD["Orthographic (Spherical)"],
        PARAMETER["Latitude of natural origin",90,
            ANGLEUNIT["Degree",0.0174532925199433],
            ID["EPSG",8801]],
        PARAMETER["Longitude of natural origin",0,
            ANGLEUNIT["Degree",0.0174532925199433],
            ID["EPSG",8802]],
        PARAMETER["False easting",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8806]],
        PARAMETER["False northing",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8807]]],
    CS[Cartesian,2],
        AXIS["(E)",east,
            ORDER[1],
            LENGTHUNIT["metre",1]],
        AXIS["(N)",north,
            ORDER[2],
            LENGTHUNIT["metre",1]],
    USAGE[
        SCOPE["Not known."],
        AREA["Northern hemisphere."],
        BBOX[0,-180,90,180]],
    ID["ESRI",102035]]'''
        return wkt

    def reproj(self):
        VIs_list = global_VIs_list
        pix_to_lon_lat_dict = DIC_and_TIF().spatial_tif_to_lon_lat_dic(temp_root)
        wkt = self.ortho_wkt()
        srs = DIC_and_TIF().gen_srs_from_wkt(wkt)
        for VI in VIs_list:
            fdir_r = join(self.this_class_tif, 'correlation_tif/{}/r'.format(VI))
            outdir_r = join(self.this_class_tif, 'correlation_tif/{}/r_reproj'.format(VI))
            T.mk_dir(outdir_r,force=True)
            for f in os.listdir(fdir_r):
                fpath = join(fdir_r, f)
                outf = join(outdir_r, f)
                ToRaster().resample_reproj(fpath, outf, 50000,dstSRS=srs)
                # exit()
        pass


    def png_correlation(self):
        outdir = join(self.this_class_png, 'correlation_png')
        T.mk_dir(outdir)
        VIs_list = global_VIs_list
        for VI in VIs_list:
            outf = join(outdir, '{}.png'.format(VI))
            fdir_r = join(self.this_class_tif, 'correlation_tif/{}/r_reproj'.format(VI))
            fig = plt.figure(figsize=(8, 12))
            flag = 1
            for f in tqdm(os.listdir(fdir_r)):
                if not f.endswith('.tif'):
                    continue
                fpath = join(fdir_r, f)
                ax = fig.add_subplot(6, 4, flag)
                flag += 1
                Plot().plot_ortho(fpath,ax)
                # ax.set_title(f.split('.')[0].split('_')[1])
            plt.tight_layout()
            plt.savefig(outf,dpi=600)
            plt.close()

    def png_correlation_3_products(self):
        outdir = join(self.this_class_png, 'png_correlation_3_products')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        outf = join(outdir, 'png_correlation_3_products_notitle_vertical.png')
        VIs_list = global_VIs_list
        spei_list = ['spei03', 'spei06', 'spei09', 'spei12',
                     'spei15','spei18', 'spei21', 'spei24',
                     ]
        fig = plt.figure(figsize=(8, 12))
        flag = 1
        for scale in spei_list:
            for VI in VIs_list:
                fpath = join(self.this_class_tif, f'correlation_tif/{VI}/r/{VI}_{scale}_r.tif')
                print(fpath,isfile(fpath))
                ax = fig.add_subplot(8, 3, flag)
                flag += 1
                Plot().plot_ortho(fpath, ax, vmin=-0.4,vmax=0.4)
                # ax.set_title(f'{VI}_{scale}')
        plt.tight_layout()
        # plt.show()
        plt.savefig(outf, dpi=600)
        plt.close()
        pass

    def tif_max_spei_scale(self):
        outdir = join(self.this_class_tif, 'max_spei_scale')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        fdir = join(self.this_class_arr,'correlation_dataframe')
        VIs_list = global_VIs_list
        for VI in VIs_list:
            dff = join(fdir,'{}.df'.format(VI))
            df = T.load_df(dff)
            cols = df.columns.tolist()
            cols.remove('pix')
            cols_r_list = []
            for col in cols:
                if col.endswith('_r'):
                    cols_r_list.append(col)
            max_r_sptatial_dict = {}
            max_scale_spaital_dict = {}
            for i,row in tqdm(df.iterrows(),total=len(df),desc=VI):
                pix = row['pix']
                r_list = [row[col] for col in cols_r_list]
                r_dict = dict(zip(cols_r_list,r_list))
                max_r = np.nanmax(r_list)
                max_scale = T.get_max_key_from_dict(r_dict)
                scale = max_scale.split('_')[1].replace('spei','')
                scale = int(scale)
                max_r_sptatial_dict[pix] = max_r
                max_scale_spaital_dict[pix] = scale
            outf_r = join(outdir,'{}_max_r.tif'.format(VI))
            outf_scale = join(outdir,'{}_max_scale.tif'.format(VI))
            DIC_and_TIF().pix_dic_to_tif(max_r_sptatial_dict,outf_r)
            DIC_and_TIF().pix_dic_to_tif(max_scale_spaital_dict,outf_scale)

    def png_max_spei_scale(self):
        fdir = join(self.this_class_tif, 'max_spei_scale')
        outdir = join(self.this_class_png, 'png_max_spei_scale')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        VI_list = global_VIs_list
        # plot correlation
        # flag = 1
        # plt.figure(figsize=(8, 5))
        # for VI in global_VIs_list:
        #     print(VI)
        #     fpath = join(fdir,f'{VI}_max_r.tif')
        #     ax = plt.subplot(1,3,flag)
        #     flag += 1
        #     Plot().plot_ortho(fpath,vmin=-0.7,vmax=0.7,ax=ax)
        #     # ax.set_title(VI)
        # outf = join(outdir,'max_r1.png')
        # plt.tight_layout()
        # plt.savefig(outf,dpi=600)
        # plt.close()

        # plot scale
        flag = 1
        plt.figure(figsize=(8, 5))
        for VI in global_VIs_list:
            print(VI)
            fpath = join(fdir, f'{VI}_max_scale.tif')
            ax = plt.subplot(1, 3, flag)
            flag += 1
            Plot().plot_ortho(fpath, vmin=1, vmax=24, ax=ax,cmap=global_cmap)
            # ax.set_title(VI)
        outf = join(outdir, 'max_scale1.png')
        plt.tight_layout()
        plt.savefig(outf, dpi=600)
        plt.close()

        pass


class VIs_and_SPEI_lag_correlation:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'VIs_and_SPEI_lag_correlation',
            result_root_this_script, mode=2)
        pass


    def run(self):
        # self.cal_correlations()
        # self.tif_correlation()
        # self.reproj()
        # self.png_correlation()
        self.tif_max_lag()
        # self.png_max_lag()


    def cal_correlations(self):
        outdir = join(self.this_class_arr, 'correlation_dataframe')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)

        gs = global_gs
        VIs_list = global_VIs_list
        VIs_year_range_dict = global_VIs_year_range_dict
        spei_scale_list = global_spei_list
        lag_list = global_lag_list

        for VI in VIs_list:
            outf = join(outdir, f'{VI}.df')
            if os.path.isfile(outf):
                continue
            VI_year_range = VIs_year_range_dict[VI]
            SPEI_data = Meta_information().load_data('SPEI', VI_year_range)
            VI_dict = Meta_information().load_data(VI, VI_year_range)
            result_dict = {}
            for scale in spei_scale_list:
                SPEI_dict = SPEI_data[scale]
                for lag in lag_list:
                    spatial_dict_r = {}
                    for pix in tqdm(SPEI_dict,desc=f'{VI}_{scale}_{lag}'):
                        SPEI = SPEI_dict[pix]
                        if not pix in VI_dict:
                            continue
                        VI_value = VI_dict[pix]
                        if T.is_all_nan(SPEI) or T.is_all_nan(VI_value):
                            continue
                        SPEI_gs = T.monthly_vals_to_annual_val(SPEI, gs, 'array')
                        VI_gs = T.monthly_vals_to_annual_val(VI_value, gs, 'array')
                        SPEI_gs_flatten = SPEI_gs.flatten()
                        VI_gs_flatten = VI_gs.flatten()
                        if not len(SPEI_gs_flatten) == len(VI_gs_flatten):
                            continue
                        r,p = T.lag_correlation(SPEI_gs_flatten,VI_gs_flatten,lag)
                        spatial_dict_r[pix] = r
                        key_r = f'{VI}_{scale}_{lag}'
                        result_dict[key_r] = spatial_dict_r
            df = T.spatial_dics_to_df(result_dict)
            T.save_df(df, outf)
            T.df_to_excel(df, outf)

    def tif_correlation(self):
        VIs_list = global_VIs_list
        fdir = join(self.this_class_arr, 'correlation_dataframe')
        for VI in VIs_list:
            outdir_r = join(self.this_class_tif, 'correlation_tif/{}/r'.format(VI))
            T.mk_dir(outdir_r,force=True)
            # T.open_path_and_file(outdir_r)
            dff = join(fdir,'{}.df'.format(VI))
            df = T.load_df(dff)
            cols = df.columns.tolist()
            cols.remove('pix')
            for col in tqdm(cols,desc=VI):
                dic = T.df_to_spatial_dic(df, col)
                outf = join(outdir_r, f'{col}.tif')
                DIC_and_TIF().pix_dic_to_tif(dic, outf)

    def reproj(self):
        VIs_list = global_VIs_list
        pix_to_lon_lat_dict = DIC_and_TIF().spatial_tif_to_lon_lat_dic(temp_root)
        wkt = ortho_wkt()
        srs = DIC_and_TIF().gen_srs_from_wkt(wkt)
        for VI in VIs_list:
            fdir_r = join(self.this_class_tif, 'correlation_tif/{}/r'.format(VI))
            outdir_r = join(self.this_class_tif, 'correlation_tif/{}/r_reproj'.format(VI))
            T.mk_dir(outdir_r,force=True)
            for f in tqdm(os.listdir(fdir_r),desc=VI):
                fpath = join(fdir_r, f)
                outf = join(outdir_r, f)
                ToRaster().resample_reproj(fpath, outf, 50000,dstSRS=srs)
        pass

    def png_correlation(self):
        outdir = join(self.this_class_png, 'correlation_png')
        T.mk_dir(outdir)
        VIs_list = global_VIs_list
        lag_list = global_lag_list
        spei_list = global_spei_list
        for VI in VIs_list:
            outf = join(outdir, '{}2.png'.format(VI))
            fdir_r = join(self.this_class_tif, 'correlation_tif/{}/r_reproj'.format(VI))
            fig = plt.figure(figsize=(8*2, 7*2))
            flag = 1
            for lag in lag_list:
                for scale in spei_list:
                    print(f'{VI}_{scale}_{lag}')
                    fpath = join(fdir_r, f'{VI}_{scale}_{lag}.tif')
                    ax = fig.add_subplot(7, 8, flag)
                    flag += 1
                    Plot().plot_ortho(fpath, ax,vmin=-0.5,vmax=0.5)
                    # ax.set_title(f'{scale}_{lag}')
                    # plt.show()
            plt.tight_layout()
            plt.savefig(outf,dpi=600)
            plt.close()

    def tif_max_lag(self):
        outdir = join(self.this_class_tif, 'max_lag')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        fdir = join(self.this_class_arr,'correlation_dataframe')
        VIs_list = global_VIs_list
        for VI in VIs_list:
            dff = join(fdir,'{}.df'.format(VI))
            df = T.load_df(dff)
            cols = df.columns.tolist()
            cols.remove('pix')
            max_r_sptatial_dict = {}
            max_lag_spaital_dict = {}
            for i,row in tqdm(df.iterrows(),total=len(df),desc=VI):
                pix = row['pix']
                r_list = [row[col] for col in cols]
                r_dict = dict(zip(cols,r_list))
                max_r = np.nanmax(r_list)
                max_lag = T.get_max_key_from_dict(r_dict)
                lag = max_lag.split('_')[-1]
                lag = int(lag)
                max_r_sptatial_dict[pix] = max_r
                max_lag_spaital_dict[pix] = lag
            outf_r = join(outdir,'{}_max_r.tif'.format(VI))
            outf_scale = join(outdir,'{}_max_lag.tif'.format(VI))
            DIC_and_TIF().pix_dic_to_tif(max_r_sptatial_dict,outf_r)
            DIC_and_TIF().pix_dic_to_tif(max_lag_spaital_dict,outf_scale)

    def png_max_lag(self):
        fdir = join(self.this_class_tif, 'max_lag')
        outdir = join(self.this_class_png, 'max_lag')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        VI_list = global_VIs_list
        # plot lag
        flag = 1
        plt.figure(figsize=(8, 5))
        for VI in global_VIs_list:
            print(VI)
            fpath = join(fdir, f'{VI}_max_lag.tif')
            ax = plt.subplot(1, 3, flag)
            flag += 1
            Plot().plot_ortho(fpath, vmin=0, vmax=6, ax=ax,cmap=global_cmap)
            ax.set_title(VI)
        outf = join(outdir, 'max_lag1.png')
        plt.tight_layout()
        plt.savefig(outf, dpi=600)
        plt.close()

        pass

class MAT_MAP:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'MAT_MAP',
            result_root_this_script, mode=2)
        pass

    def run(self):
        # self.tif_offset()
        self.plot_png()
        pass

    def tif_offset(self):
        fdir = join(data_root, 'MAT_MAP')
        for f in T.listdir(fdir):
            if not f.endswith('tif'):
                continue
            fpath = join(fdir, f)
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            DIC_and_TIF().pix_dic_to_tif(spatial_dict, fpath)
        pass

    def plot_png(self):
        fdir = join(data_root,'MAT_MAP')
        outdir = join(self.this_class_png,'MAT_MAP')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        outf = join(outdir,'MAT_MAP.jpg')
        mat_path = join(fdir,'MAT.tif')
        map_path = join(fdir,'MAP.tif')
        plt.figure(figsize=(8, 5))
        ax = plt.subplot(1, 2, 1)
        m,ret = Plot().plot_ortho(mat_path, vmin=-10, vmax=20, ax=ax, cmap=global_cmap_r)
        m.colorbar(ret,location='bottom')
        ax.set_title('MAT')

        ax = plt.subplot(1, 2, 2)
        m,ret = Plot().plot_ortho(map_path, vmin=0, vmax=1000, ax=ax, cmap=global_cmap)
        m.colorbar(ret,location='bottom')
        ax.set_title('MAP')
        plt.tight_layout()
        # plt.show()
        plt.savefig(outf, dpi=600)
        plt.close()
        pass


class Isohydricity:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Isohydricity',
            result_root_this_script, mode=2)
        pass

    def run(self):
        # self.tif_offset()
        self.plot_png()
        pass

    def tif_offset(self):
        fpath = join(data_root, 'ISO_Hydro/tif_all_year/ISO_Hydricity.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
        DIC_and_TIF().pix_dic_to_tif(spatial_dict, fpath)
        pass

    def plot_png(self):
        fpath = join(data_root, 'ISO_Hydro/tif_all_year/ISO_Hydricity.tif')
        outdir = join(self.this_class_png,'ISO_Hydro')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        outf = join(outdir,'ISO_Hydricity.jpg')
        plt.figure(figsize=(5, 5))
        ax = plt.subplot(1, 1, 1)
        m,ret = Plot().plot_ortho(fpath, vmin=0.3, vmax=1.3, ax=ax, cmap=global_cmap_r)
        m.colorbar(ret,location='bottom')
        ax.set_title('ISO_Hydricity')
        plt.savefig(outf, dpi=600)
        plt.close()
        pass

class Aridity_index:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Aridity_index',
            result_root_this_script, mode=2)
        pass

    def run(self):
        # self.tif_offset()
        self.plot_png()
        pass

    def tif_offset(self):
        fpath = join(data_root, 'Aridity_Index/aridity_index.tif.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
        DIC_and_TIF().pix_dic_to_tif(spatial_dict, fpath)
        pass

    def plot_png(self):
        fpath = join(data_root, 'Aridity_Index/aridity_index.tif')
        outdir = join(self.this_class_png,'Aridity_Index')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        outf = join(outdir,'Aridity_Index.jpg')
        plt.figure(figsize=(5, 5))
        ax = plt.subplot(1, 1, 1)
        m,ret = Plot().plot_ortho(fpath, vmin=0, vmax=1.3, ax=ax, cmap=global_cmap)
        m.colorbar(ret,location='bottom')
        ax.set_title('Aridity_Index')
        plt.savefig(outf, dpi=600)
        plt.close()
        pass

class Moving_window_correlation:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Moving_window_correlation',
            result_root_this_script, mode=2)
        pass

    def run(self):
        # self.VI_SPEI_moving_window_correlation()
        # self.plot_moving_window_correlation_time_sereis()
        # self.plot_moving_window_correlation_ELI()
        # self.moving_window_spatial_trend()
        # self.plot_moving_window_spatial_trend()
        # self.VI_SPEI_moving_window_correlation_lag()
        # self.pick_max_moving_window_lag_correlation()
        # self.max_lag_moving_window_spatial_trend()
        # self.plot_max_lag_moving_window_spatial_trend()
        self.plot_moving_window_lag_correlation_time_sereis()
        pass

    def moving_window_correlation(self, arr1, arr2, window_size:int=10, date_list:list=None):
        if not len(arr1) == len(arr2):
            raise ValueError('arr1 and arr2 must have the same length')
        if not date_list is None:
            if not len(arr1) == len(date_list):
                raise ValueError('arr and date_list must have the same length')
        if window_size <= 3:
            raise ValueError('window_size must be greater than 3')
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)
        corr_dict = {}
        for i in range(len(arr1)):
            if i + window_size >= len(arr1):
                break
            if not date_list is None:
                window_name = f'{date_list[i]}-{date_list[i+window_size]}'
            else:
                window_name = f'{i}-{i+window_size}'
            picked_arr1 = arr1[i:i+window_size]
            picked_arr2 = arr2[i:i+window_size]
            r,p = T.nan_correlation(picked_arr1, picked_arr2)
            corr_dict[window_name] = r
        return corr_dict

    def lag_correlation(self,earlier,later,lag,method='pearson'):
        '''
        earlier value can affect later value
        e.g. earlier SPEI, later NDVI
        :return: correlation
        '''
        if lag == 0:
            r, p = T.nan_correlation(earlier, later, method)
        else:
            later = later[lag:]
            earlier = earlier[:-lag]
            r,p = T.nan_correlation(earlier, later, method)
        return r,p

    def moving_window_lag_correlation(self, arr1, arr2, lag_list, window_size:int=10, date_list:list=None):
        if not len(arr1) == len(arr2):
            raise ValueError('arr1 and arr2 must have the same length')
        if not date_list is None:
            if not len(arr1) == len(date_list):
                raise ValueError('arr and date_list must have the same length')
        if window_size <= 3:
            raise ValueError('window_size must be greater than 3')
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)
        corr_dict = {}
        for i in range(len(arr1)):
            if i + window_size >= len(arr1):
                break
            if not date_list is None:
                window_name = f'{date_list[i]}-{date_list[i+window_size]}'
            else:
                window_name = f'{i}-{i+window_size}'
            picked_arr1 = arr1[i:i+window_size]
            picked_arr2 = arr2[i:i+window_size]
            # r,p = T.nan_correlation(picked_arr1, picked_arr2)
            for lag in lag_list:
                r,p = T.lag_correlation(picked_arr1, picked_arr2,lag)
                corr_dict[f'{window_name}_{lag}'] = r
        return corr_dict


    def VI_SPEI_moving_window_correlation(self):
        from Chapter3 import statistic
        outdir = join(self.this_class_arr,'Moving_window_correlation')
        T.mk_dir(outdir)
        # T.open_path_and_file(outdir)
        gs = global_gs
        VIs_list = global_VIs_list
        SPEI_list = global_spei_list

        for VI in VIs_list:
            max_spei_scale = join(VIs_and_SPEI_correlation().this_class_tif, 'max_spei_scale',f'{VI}_max_scale.tif')
            max_spei_scale_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(max_spei_scale)
            year_range = global_VIs_year_range_dict[VI]
            start_year, end_year = year_range.split('-')
            start_year = int(start_year)
            end_year = int(end_year)
            year_list = list(range(start_year, end_year + 1))
            VI_data = Meta_information().load_data(VI, year_range)
            SPEI_data = Meta_information().load_data('SPEI', year_range)
            VI_spatial_dict = VI_data
            correlation_spatial_dict = {}
            for pix in tqdm(VI_spatial_dict):
                VI_arr = VI_spatial_dict[pix]
                max_scale = max_spei_scale_spatial_dict[pix]
                if np.isnan(max_scale):
                    continue
                max_scale = int(max_scale)
                max_scale = f'spei{max_scale:02d}'
                if not pix in SPEI_data[max_scale]:
                    continue
                SPEI_arr = SPEI_data[max_scale][pix]
                VI_gs = T.monthly_vals_to_annual_val(VI_arr,gs,method='mean')
                SPEI_gs = T.monthly_vals_to_annual_val(SPEI_arr,gs,method='mean')
                corr_dict = self.moving_window_correlation(VI_gs, SPEI_gs, window_size=10,date_list=year_list)
                correlation_spatial_dict[pix] = corr_dict
            df = T.dic_to_df(correlation_spatial_dict,key_col_str='pix')
            df = statistic.Dataframe_func(df).df
            outf = join(outdir,f'{VI}.df')
            T.save_df(df,outf)
            T.df_to_excel(df,outf)

    def VI_SPEI_moving_window_correlation_lag(self):
        from Chapter3 import statistic
        outdir = join(self.this_class_arr,'VI_SPEI_moving_window_correlation_lag')
        T.mk_dir(outdir)
        # T.open_path_and_file(outdir)
        VI = 'NDVI'
        gs = global_gs
        VIs_list = global_VIs_list
        SPEI_list = global_spei_list
        lag_list = global_lag_list

        max_spei_scale = join(VIs_and_SPEI_correlation().this_class_tif, 'max_spei_scale',f'{VI}_max_scale.tif')
        max_spei_scale_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(max_spei_scale)
        year_range = global_VIs_year_range_dict[VI]
        start_year, end_year = year_range.split('-')
        start_year = int(start_year)
        end_year = int(end_year)
        year_list = list(range(start_year, end_year + 1))
        VI_data = Meta_information().load_data(VI, year_range)
        SPEI_data = Meta_information().load_data('SPEI', year_range)
        VI_spatial_dict = VI_data
        correlation_spatial_dict = {}
        for pix in tqdm(VI_spatial_dict):
            VI_arr = VI_spatial_dict[pix]
            max_scale = max_spei_scale_spatial_dict[pix]
            if np.isnan(max_scale):
                continue
            max_scale = int(max_scale)
            max_scale = f'spei{max_scale:02d}'
            if not pix in SPEI_data[max_scale]:
                continue
            SPEI_arr = SPEI_data[max_scale][pix]
            VI_gs = T.monthly_vals_to_annual_val(VI_arr,gs,method='mean')
            SPEI_gs = T.monthly_vals_to_annual_val(SPEI_arr,gs,method='mean')
            corr_dict = self.moving_window_lag_correlation(SPEI_gs, VI_gs,lag_list, window_size=10,date_list=year_list)
            # print(corr_dict)
            # exit()
            correlation_spatial_dict[pix] = corr_dict
        df = T.dic_to_df(correlation_spatial_dict,key_col_str='pix')
        df = statistic.Dataframe_func(df).df
        outf = join(outdir,f'{VI}.df')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

    def plot_moving_window_correlation_time_sereis(self):
        fdir = join(self.this_class_arr,'Moving_window_correlation')
        outdir = join(self.this_class_png,'plot_moving_window_correlation')
        T.mk_dir(outdir)
        # T.open_path_and_file(outdir)
        dff = join(fdir,'NDVI.df')
        df = T.load_df(dff)
        year_list = df.columns.tolist()
        year_list.remove('pix')
        other_cols = ['lon', 'lat', 'landcover_GLC', 'NDVI_MASK', 'aridity_index', 'ELI', 'ELI_class', 'AI_class', 'Koppen', 'MAT', 'MAP', 'ISO_Hydricity']
        for col in other_cols:
            year_list.remove(col)
        ltd_list = global_ELI_class
        for ltd in ltd_list:
            df_ltd = df[df['ELI_class'] == ltd]
            moving_window = []
            err_list = []
            for year in year_list:
                vals = df_ltd[year].tolist()
                # print(vals)
                vals_mean = np.nanmean(vals)
                # err = np.nanstd(vals)
                err,_,_ = T.uncertainty_err(vals)
                moving_window.append(vals_mean)
                err_list.append(err)
            plt.figure(figsize=(5, 3))
            plt.plot(year_list,moving_window)
            plt.xticks(year_list, rotation=90)
            plt.fill_between(year_list, np.array(moving_window) - np.array(err_list), np.array(moving_window) + np.array(err_list), alpha=0.5)
            plt.title(ltd)
            plt.xlabel('Year')
            plt.ylabel('Correlation')
            plt.tight_layout()
            outf = join(outdir,f'{ltd}.pdf')
            plt.savefig(outf)
            plt.close()

    def pick_max_moving_window_lag_correlation(self):
        from Chapter3 import statistic
        fdir = join(self.this_class_arr,'VI_SPEI_moving_window_correlation_lag')
        outdir = join(self.this_class_arr,'pick_max_moving_window_lag_correlation')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        dff = join(fdir,'NDVI.df')
        df = T.load_df(dff)
        cols_list = df.columns.tolist()
        cols_list.remove('pix')
        other_cols = ['lon', 'lat', 'landcover_GLC', 'NDVI_MASK', 'aridity_index', 'ELI', 'ELI_class', 'AI_class', 'Koppen', 'MAT', 'MAP', 'ISO_Hydricity']
        for col in other_cols:
            cols_list.remove(col)
        year_list = [col.split('_')[0] for col in cols_list]
        year_list = list(set(year_list))
        year_list.sort()
        lag_list = global_lag_list
        result_dict = {}
        for year in year_list:
            year_cols_list = []
            for lag in lag_list:
                col = f'{year}_{lag}'
                year_cols_list.append(col)
            max_lag_spatial_dict = {}
            for i,row in tqdm(df.iterrows(),total=len(df),desc=f'{year}'):
                pix = row['pix']
                df_i = row[year_cols_list]
                row_dict = df_i.to_dict()
                max_col = T.get_max_key_from_dict(row_dict)
                if max_col == None:
                    continue
                max_lag = max_col.split('_')[1]
                max_lag = int(max_lag)
                max_lag_spatial_dict[pix] = max_lag
            key = f'{year}'
            result_dict[key] = max_lag_spatial_dict
        df_max_lag = T.spatial_dics_to_df(result_dict)
        df_max_lag = statistic.Dataframe_func(df_max_lag).df
        outf = join(outdir,f'NDVI.df')
        T.save_df(df_max_lag,outf)
        T.df_to_excel(df_max_lag,outf)


    def plot_moving_window_correlation_ELI(self):
        fdir = join(self.this_class_arr,'Moving_window_correlation')
        outdir = join(self.this_class_png,'plot_moving_window_correlation_ELI')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        dff = join(fdir,'NDVI.df')
        df = T.load_df(dff)
        year_list = df.columns.tolist()
        year_list.remove('pix')
        other_cols = ['lon', 'lat', 'landcover_GLC', 'NDVI_MASK', 'aridity_index', 'ELI', 'ELI_class', 'AI_class', 'Koppen', 'MAT', 'MAP', 'ISO_Hydricity']
        for col in other_cols:
            year_list.remove(col)
        # print(year_list)
        # exit()
        ELI_bins = global_ELI_bins
        ELI_col = 'ELI'
        df_group,bins_list_str = T.df_bin(df,ELI_col,ELI_bins)
        matrix = []
        y_list = []
        for name,df_group_i in df_group:
            y_list.append(name.left)
            moving_window = []
            for year in year_list:
                vals = df_group_i[year].tolist()
                # print(vals)
                vals_mean = np.nanmean(vals)
                moving_window.append(vals_mean)
            matrix.append(moving_window)
        matrix = np.array(matrix)
        plt.imshow(matrix,aspect='auto',cmap='RdBu_r',vmin=-0.2,vmax=0.6)
        plt.colorbar()
        plt.xticks(range(len(year_list)),year_list,rotation=90)
        plt.yticks(range(len(y_list)),y_list)
        plt.xlabel('Year')
        plt.ylabel('ELI')
        plt.title('Moving window correlation')
        plt.tight_layout()
        outf = join(outdir,f'ELI.pdf')
        plt.savefig(outf)
        plt.close()
        # plt.show()

    def moving_window_spatial_trend(self):
        outdir = join(self.this_class_tif,'moving_window_spatial_trend')
        T.mk_dir(outdir)

        dff = join(self.this_class_arr,'Moving_window_correlation','NDVI.df')
        df = T.load_df(dff)
        T.print_head_n(df,5)
        year_list = df.columns.tolist()
        year_list.remove('pix')
        other_cols = ['lon', 'lat', 'landcover_GLC', 'NDVI_MASK', 'aridity_index', 'ELI', 'ELI_class', 'AI_class',
                      'Koppen', 'MAT', 'MAP', 'ISO_Hydricity']
        for col in other_cols:
            year_list.remove(col)
        moving_window_trend_dict = {}
        moving_window_trend_p_dict = {}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            year_vals = []
            for col in year_list:
                val = row[col]
                year_vals.append(val)
            pix = row['pix']
            a,b,r,p = T.nan_line_fit(list(range(len(year_vals))),year_vals)
            moving_window_trend_dict[pix] = a
            moving_window_trend_p_dict[pix] = p
        outf_trend = join(outdir,'moving_window_trend.tif')
        outf_p = join(outdir,'moving_window_trend_p.tif')
        DIC_and_TIF().pix_dic_to_tif(moving_window_trend_dict,outf_trend)
        DIC_and_TIF().pix_dic_to_tif(moving_window_trend_p_dict,outf_p)
        pass

    def plot_moving_window_spatial_trend(self):
        fdir = join(self.this_class_tif,'moving_window_spatial_trend')
        outdir = join(self.this_class_png,'plot_moving_window_spatial_trend')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        trend_tif = join(fdir,'moving_window_trend.tif')
        trend_p_tif = join(fdir,'moving_window_trend_p.tif')
        plt.figure(figsize=(8,8))
        # m,ret = Plot().plot_ortho(trend_tif,cmap='RdBu_r',vmin=-0.05,vmax=0.05)
        m,ret = Plot().plot_ortho(trend_tif,vmin=-0.05,vmax=0.05)
        Plot().plot_ortho_significance_scatter(m,trend_p_tif,temp_root,linewidths=1.5)
        # m.colorbar(ret,location='bottom',pad='5%')
        outf = join(outdir,'moving_window_trend.png')
        plt.savefig(outf,dpi=600)
        plt.close()
        # plt.show()

    def max_lag_moving_window_spatial_trend(self):
        outdir = join(self.this_class_tif,'max_lag_moving_window_spatial_trend')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)

        dff = join(self.this_class_arr,'pick_max_moving_window_lag_correlation','NDVI.df')
        df = T.load_df(dff)
        T.print_head_n(df,5)
        year_list = df.columns.tolist()
        year_list.remove('pix')
        other_cols = ['lon', 'lat', 'landcover_GLC', 'NDVI_MASK', 'aridity_index', 'ELI', 'ELI_class', 'AI_class',
                      'Koppen', 'MAT', 'MAP', 'ISO_Hydricity']
        for col in other_cols:
            year_list.remove(col)
        moving_window_trend_dict = {}
        moving_window_trend_p_dict = {}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            year_vals = []
            for col in year_list:
                val = row[col]
                year_vals.append(val)
            pix = row['pix']
            try:
                a,b,r,p = T.nan_line_fit(list(range(len(year_vals))),year_vals)
                moving_window_trend_dict[pix] = a
                moving_window_trend_p_dict[pix] = p
            except:
                continue
        outf_trend = join(outdir,'moving_window_trend.tif')
        outf_p = join(outdir,'moving_window_trend_p.tif')
        DIC_and_TIF().pix_dic_to_tif(moving_window_trend_dict,outf_trend)
        DIC_and_TIF().pix_dic_to_tif(moving_window_trend_p_dict,outf_p)
        pass

    def plot_max_lag_moving_window_spatial_trend(self):
        fdir = join(self.this_class_tif,'max_lag_moving_window_spatial_trend')
        outdir = join(self.this_class_png,'plot_max_lag_moving_window_spatial_trend')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        trend_tif = join(fdir,'moving_window_trend.tif')
        trend_p_tif = join(fdir,'moving_window_trend_p.tif')
        plt.figure(figsize=(8,8))
        # m,ret = Plot().plot_ortho(trend_tif,cmap='RdBu_r',vmin=-0.05,vmax=0.05)
        m,ret = Plot().plot_ortho(trend_tif,vmin=-0.2,vmax=0.2,cmap=global_cmap)
        Plot().plot_ortho_significance_scatter(m,trend_p_tif,temp_root,linewidths=1.5)
        # m.colorbar(ret,location='bottom',pad='5%')
        outf = join(outdir,'moving_window_trend.png')
        plt.savefig(outf,dpi=600)
        plt.close()
        # plt.show()

    def plot_moving_window_lag_correlation_time_sereis(self):
        fdir = join(self.this_class_arr,'pick_max_moving_window_lag_correlation')
        outdir = join(self.this_class_png,'plot_moving_window_lag_correlation_time_sereis')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        dff = join(fdir,'NDVI.df')
        df = T.load_df(dff)
        year_list = df.columns.tolist()
        year_list.remove('pix')
        other_cols = ['lon', 'lat', 'landcover_GLC', 'NDVI_MASK', 'aridity_index', 'ELI', 'ELI_class', 'AI_class', 'Koppen', 'MAT', 'MAP', 'ISO_Hydricity']
        for col in other_cols:
            year_list.remove(col)
        ltd_list = global_ELI_class
        for ltd in ltd_list:
            df_ltd = df[df['ELI_class'] == ltd]
            moving_window = []
            err_list = []
            for year in year_list:
                vals = df_ltd[year].tolist()
                # print(vals)
                vals_mean = np.nanmean(vals)
                # err = np.nanstd(vals)
                err,_,_ = T.uncertainty_err(vals)
                moving_window.append(vals_mean)
                err_list.append(err)
            plt.figure(figsize=(5, 3))
            plt.plot(year_list,moving_window)
            plt.xticks(year_list, rotation=90)
            plt.fill_between(year_list, np.array(moving_window) - np.array(err_list), np.array(moving_window) + np.array(err_list), alpha=0.5)
            plt.title(ltd)
            plt.xlabel('Year')
            plt.ylabel('Correlation')
            # plt.ylim(2,3.6)
            plt.tight_layout()
            outf = join(outdir,f'{ltd}.pdf')
            plt.savefig(outf)
            plt.close()


class ELI_vs_moving_window_correlation:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'ELI_vs_moving_window_correlation',
            result_root_this_script, mode=2)
        pass

    def run(self):
        # self.plot_scatter_ELI_vs_moving_window_correlation()
        # self.plot_line_ELI_vs_moving_window_correlation()
        self.plot_line_ELI_vs_moving_window_lag()
        pass

    def gen_df(self):
        ELI_trend_f = join(Water_energy_limited_area_daily().this_class_tif, 'trend/1982-2015/ELI_trend.tif')
        corr_trend_f = join(Moving_window_correlation().this_class_tif,
                            'moving_window_spatial_trend/moving_window_trend.tif')
        lag_trend_f = join(Moving_window_correlation().this_class_tif,
                           'max_lag_moving_window_spatial_trend/moving_window_trend.tif')

        ELI_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(ELI_trend_f)
        corr_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(corr_trend_f)
        lag_spatial_dict = DIC_and_TIF().spatial_tif_to_dic(lag_trend_f)

        all_dict = {'ELI': ELI_spatial_dict, 'corr': corr_spatial_dict, 'lag': lag_spatial_dict}
        df = T.spatial_dics_to_df(all_dict)
        df = df.dropna()
        return df


    def plot_scatter_ELI_vs_moving_window_correlation(self):
        df = self.gen_df()
        x = df['ELI'].tolist()
        y1 = df['corr'].tolist()
        y2 = df['lag'].tolist()
        # KDE_plot().plot_scatter_hex(x,y1,xlim=(-0.01,0.01),ylim=(-0.05,0.05))
        # x_y_lim = [-0.01,0.01,-0.05,0.05]
        x_y_lim = [-0.01,0.01,-0.2,0.2]
        cmap = KDE_plot().cmap_with_transparency('gray_r')
        # KDE_plot().plot_scatter(x,y1,x_y_lim=x_y_lim,cmap=cmap,s=10)
        KDE_plot().plot_scatter(x,y2,x_y_lim=x_y_lim,cmap=cmap,s=10)
        # plt.title('ELI vs moving window correlation')
        plt.xlabel('ELI')
        # plt.ylabel('Moving window correlation')
        plt.ylabel('Moving window lag')
        plt.show()

    def plot_line_ELI_vs_moving_window_correlation(self):
        df = self.gen_df()
        ELI_bins = np.linspace(-0.005,0.01,30)
        ELI_col = 'ELI'
        corr_col = 'corr'
        lag_col = 'lag'

        df_group,bins_list_str = T.df_bin(df,ELI_col,ELI_bins)

        y_list = []
        err_list = []
        for name,df_group_i in df_group:
            vals = df_group_i[corr_col].tolist()
            mean = np.nanmean(vals)
            err,_,_ = T.uncertainty_err(vals)
            # x_list.append(name)
            y_list.append(mean)
            err_list.append(err)
        plt.figure(figsize=(5,3))
        plt.plot(bins_list_str,y_list)
        plt.fill_between(bins_list_str, np.array(y_list)-np.array(err_list), np.array(y_list)+np.array(err_list), alpha=0.5)
        plt.xlabel('ELI trend')
        plt.ylabel('correlation trend')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def plot_line_ELI_vs_moving_window_lag(self):
        df = self.gen_df()
        ELI_bins = np.linspace(-0.005,0.01,30)
        ELI_col = 'ELI'
        lag_col = 'lag'

        df_group,bins_list_str = T.df_bin(df,ELI_col,ELI_bins)

        y_list = []
        err_list = []
        for name,df_group_i in df_group:
            vals = df_group_i[lag_col].tolist()
            mean = np.nanmean(vals)
            err,_,_ = T.uncertainty_err(vals)
            # x_list.append(name)
            y_list.append(mean)
            err_list.append(err)
        plt.figure(figsize=(5,3))
        plt.plot(bins_list_str,y_list)
        plt.fill_between(bins_list_str, np.array(y_list)-np.array(err_list), np.array(y_list)+np.array(err_list), alpha=0.5)
        plt.xlabel('ELI trend')
        plt.ylabel('lag trend')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()



        pass


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
    # Water_energy_limited_area_daily().run()
    # Growing_season().run()
    # SPEI_trend().run()
    VIs_trend().run()
    # VIs_and_SPEI_correlation().run()
    # VIs_and_SPEI_lag_correlation().run()
    # MAT_MAP().run()
    # Isohydricity().run()
    # Aridity_index().run()
    # Moving_window_correlation().run()
    # ELI_vs_moving_window_correlation().run()

    # gen_world_grid_shp()
    pass


if __name__ == '__main__':
    main()
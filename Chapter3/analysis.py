# coding=utf-8
from meta_info import *
import xymap
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
        outf = join(outdir, 'png_correlation_3_products_notitle.png')
        VIs_list = global_VIs_list
        spei_list = ['spei03', 'spei06', 'spei09', 'spei12',
                     'spei15','spei18', 'spei21', 'spei24',
                     ]
        fig = plt.figure(figsize=(16, 6))
        flag = 1
        for VI in VIs_list:
            for scale in spei_list:
                fpath = join(self.this_class_tif, f'correlation_tif/{VI}/r_reproj/{VI}_{scale}_r.tif')
                print(fpath,isfile(fpath))
                ax = fig.add_subplot(3, 8, flag)
                flag += 1
                Plot().plot_ortho(fpath, ax, vmin=-0.4,vmax=0.4)
                # ax.set_title(f'{VI}_{scale}')
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
        self.png_correlation()


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
    # Growing_season().run()
    # VIs_and_SPEI_correlation().run()
    VIs_and_SPEI_lag_correlation().run()
    # Max_Scale_and_Lag_correlation_SPEI().run()
    # Max_Scale_and_Lag_correlation_SPI().run()
    # Pick_Drought_Events().run()
    # Resistance_Resilience().run()

    # gen_world_grid_shp()
    pass


if __name__ == '__main__':
    main()
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
        # self.png_correlation_3_products()
        self.tif_max_spei_scale()
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
    # MAT_MAP().run()
    # Isohydricity().run()
    # Aridity_index().run()

    # gen_world_grid_shp()
    pass


if __name__ == '__main__':
    main()
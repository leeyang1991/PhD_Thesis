# coding=utf-8
from meta_info import *
result_root_this_script = join(results_root, 'Chapter3/statistic')

class Dataframe_func:

    def __init__(self,df):
        print('add lon lat')
        df = self.add_lon_lat(df)
        print('add landcover')
        df = self.add_GLC_landcover_data_to_df(df)
        print('add NDVI mask')
        df = self.add_NDVI_mask(df)
        print('add Aridity Index')
        df = self.add_AI_to_df(df)
        print('add ELI')
        df = self.add_ELI_to_df(df)
        print('add ELI_reclass')
        df = self.ELI_reclass(df)
        print('add AI_reclass')
        df = self.AI_reclass(df)
        print('add koppen')
        df = self.add_koppen(df)
        print('add MAT and MAP')
        df = self.add_MAT_MAP(df)
        print('add Isohydricity')
        df = self.add_isohydricity(df)

        df = self.clean_df(df)
        self.df = df

    def clean_df(self,df):

        df = df[df['lat']>=30]
        # df = df[df['landcover_GLC'] != 'Crop']
        df = df[df['NDVI_MASK'] == 1]
        return df

    def add_GLC_landcover_data_to_df(self, df):
        f = join(data_root,'GLC2000/reclass_lc_dic.npy')
        val_dic=T.load_npy(f)
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            val_list.append(vals)
        df['landcover_GLC'] = val_list
        return df

    def add_NDVI_mask(self,df):
        # f =rf'C:/Users/pcadmin/Desktop/Data/Base_data/NDVI_mask.tif'
        f = join(data_root, 'GIMMS_NDVI/NDVI_mask.tif')
        print(f)

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'NDVI_MASK'
        print(f_name)
        # exit()
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            if vals < -99:
                val_list.append(np.nan)
                continue
            val_list.append(vals)
        df[f_name] = val_list
        return df

    def add_AI_to_df(self, df):
        f = join(data_root, 'Aridity_Index/aridity_index.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(f)
        df = T.add_spatial_dic_to_df(df, spatial_dict, 'aridity_index')
        return df

    def add_lon_lat(self,df):
        df = T.add_lon_lat_to_df(df, DIC_and_TIF())
        return df

    def add_ELI_to_df(self,df):
        import analysis
        f = join(analysis.Water_energy_limited_area().this_class_tif, 'ELI/GLEAM-ET_ERA-SM_Temperature.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(f)
        df = T.add_spatial_dic_to_df(df, spatial_dict, 'ELI')
        return df

    def ELI_reclass(self,df):
        ELI_class = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            ELI = row['ELI']
            if ELI < 0:
                ELI_class.append('Energy-Limited')
            elif ELI > 0:
                ELI_class.append('Water-Limited')
            else:
                ELI_class.append(np.nan)
        df['ELI_class'] = ELI_class
        return df

    def AI_reclass(self,df):
        AI_class = []
        for i,row in df.iterrows():
            AI = row['aridity_index']
            if AI < 0.65:
                AI_class.append('Arid')
            elif AI >= 0.65:
                AI_class.append('Humid')
            elif np.isnan(AI):
                AI_class.append(np.nan)
            else:
                print(AI)
                raise ValueError('AI error')
        df['AI_class'] = AI_class
        return df

    def add_koppen(self,df):
        f = join(data_root, 'koppen/koppen_reclass_dic.npy')
        val_dic = T.load_npy(f)
        df = T.add_spatial_dic_to_df(df, val_dic, 'Koppen')
        return df

    def add_MAT_MAP(self,df):
        MAT_f = join(data_root, 'MAT_MAP/MAT.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(MAT_f)
        df = T.add_spatial_dic_to_df(df, spatial_dict, 'MAT')
        MAP_f = join(data_root, 'MAT_MAP/MAP.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(MAP_f)
        df = T.add_spatial_dic_to_df(df, spatial_dict, 'MAP')
        return df

    def add_isohydricity(self,df):
        iso_f = join(data_root,'ISO_Hydro/tif_all_year/ISO_Hydricity.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(iso_f)
        df = T.add_spatial_dic_to_df(df, spatial_dict, 'ISO_Hydricity')
        return df
        pass

class Dataframe:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Dataframe', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'dataframe.df')
        pass

    def run(self):
        # self.correlaion_to_df()
        # exit()
        df = self.__gen_df_init()
        df = Dataframe_func(df).df

        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)
        pass

    def correlaion_to_df(self):
        if os.path.isfile(self.dff):
            raise ValueError('dff already exist')
        import analysis
        VI_list = global_VIs_list
        max_scale_fdir = join(analysis.VIs_and_SPEI_correlation().this_class_tif, 'max_spei_scale')
        max_lag_fdir = join(analysis.VIs_and_SPEI_lag_correlation().this_class_tif, 'max_lag')

        spatial_dics = {}
        for f in T.listdir(max_scale_fdir):
            if not f.endswith('.tif'):
                continue
            col = f.split('.')[0]
            fpath = join(max_scale_fdir, f)
            spatial_dic = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dics[col] = spatial_dic

        for f in T.listdir(max_lag_fdir):
            if not f.endswith('.tif'):
                continue
            if f.endswith('_r.tif'):
                continue
            col = f.split('.')[0]
            fpath = join(max_lag_fdir, f)
            spatial_dic = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dics[col] = spatial_dic

        df = T.spatial_dics_to_df(spatial_dics)
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)




    def __gen_df_init(self):
        if not os.path.isfile(self.dff):
            df = pd.DataFrame()
            T.save_df(df,self.dff)
            return df
        else:
            df,dff = self.__load_df()
            return df

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        print('len(df):',len(df))
        return df,dff

class Constant_value:
    '''
    check dataframe constant value
    '''
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Constant_value',
            result_root_this_script, mode=2)
        pass

    def run(self):
        self.hist()
        # self.plot_spatial()
        pass

    def hist(self):
        outdir = join(self.this_class_png,'hist')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        constant_col = [
            'CSIF-anomaly_max_r',
            'aridity_index',
            'ELI',
            'MAT',
            'MAP',
            'ISO_Hydricity',
        ]
        df = Load_dataframe().load_chapter3()
        for col in constant_col:
            vals = df[col].values
            plt.figure()
            plt.hist(vals,bins=80)
            plt.title(col)
            outf = join(outdir,col+'.png')
            plt.savefig(outf)
            plt.close()

        pass

    def plot_spatial(self):
        df = Load_dataframe().load_chapter3()
        constant_col = [
            'CSIF-anomaly_max_r',
            'aridity_index',
            'ELI',
            'MAT',
            'MAP',
            'ISO_Hydricity',
        ]
        for col in constant_col:
            spatial_dict = T.df_to_spatial_dic(df, col)
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
            plt.figure()
            plt.imshow(arr)
            plt.title(col)
            plt.colorbar()
        plt.show()

class ELI_AI_gradient:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('ELI_AI_gradient', result_root_this_script, mode=2)
        pass

    def run(self):
        self.lag_ELI()
        # self.lag_AI()
        # self.scale_ELI()
        # self.scale_AI()
        # self.rt_rs_ELI()
        # self.rt_rs_AI()
        # self.max_r_ELI()
        pass

    def lag_ELI(self):
        outdir = join(self.this_class_png, 'lag_ELI')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        T.print_head_n(df, 5)
        df_group_dict = T.df_groupby(df, 'pix')

        lag_list = []
        ELI_list = []
        for pix in df_group_dict:
            df_pix = df_group_dict[pix]
            lag = df_pix['max_lag']
            ELI = df_pix['ELI']
            lag_mean = np.nanmean(lag)
            ELI_mean = np.nanmean(ELI)
            lag_list.append(lag_mean)
            ELI_list.append(ELI_mean)
        df_new = pd.DataFrame()
        df_new['lag'] = lag_list
        df_new['ELI'] = ELI_list
        bins = np.linspace(-0.6, 0.6, 20)
        df_group, bins_list_str = T.df_bin(df_new, 'ELI', bins)
        x_list = []
        y_list = []
        err_list = []
        for name,df_group_i in df_group:
            vals = df_group_i['lag'].tolist()
            mean = np.nanmean(vals)
            err,_,_ = T.uncertainty_err(vals)
            x_list.append(name.left)
            y_list.append(mean)
            err_list.append(err)
        plt.figure(figsize=(6,3))
        plt.errorbar(x_list, y_list, yerr=err_list)
        plt.xlabel('ELI (energy-limited --> water-limited)')
        plt.ylabel('Lag (years)')
        plt.tight_layout()

        # outf = join(outdir, 'lag_ELI.png')
        outf = join(outdir, 'lag_ELI.pdf')
        plt.savefig(outf, dpi=300)


    def lag_AI(self):
        outdir = join(self.this_class_png, 'lag_AI')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        T.print_head_n(df, 5)
        df_group_dict = T.df_groupby(df, 'pix')

        lag_list = []
        ELI_list = []
        for pix in df_group_dict:
            df_pix = df_group_dict[pix]
            lag = df_pix['max_lag']
            ELI = df_pix['aridity_index']
            lag_mean = np.nanmean(lag)
            ELI_mean = np.nanmean(ELI)
            lag_list.append(lag_mean)
            ELI_list.append(ELI_mean)
        df_new = pd.DataFrame()
        df_new['lag'] = lag_list
        df_new['aridity_index'] = ELI_list
        bins = np.linspace(0, 3, 20)
        df_group, bins_list_str = T.df_bin(df_new, 'aridity_index', bins)
        x_list = []
        y_list = []
        err_list = []
        for name,df_group_i in df_group:
            vals = df_group_i['lag'].tolist()
            mean = np.nanmean(vals)
            err,_,_ = T.uncertainty_err(vals)
            x_list.append(name.left)
            y_list.append(mean)
            err_list.append(err)
        plt.figure(figsize=(6,3))
        plt.errorbar(x_list, y_list, yerr=err_list)
        plt.xlabel('AI (Arid --> Humid)')
        plt.ylabel('Lag (years)')
        plt.tight_layout()

        outf = join(outdir, 'lag_ELI.png')
        plt.savefig(outf, dpi=300)

    def scale_ELI(self):
        outdir = join(self.this_class_png, 'scale_ELI')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        T.print_head_n(df, 5)
        df_group_dict = T.df_groupby(df, 'pix')

        lag_list = []
        ELI_list = []
        for pix in df_group_dict:
            df_pix = df_group_dict[pix]
            lag = df_pix['max_scale']
            ELI = df_pix['ELI']
            lag_mean = np.nanmean(lag)
            ELI_mean = np.nanmean(ELI)
            lag_list.append(lag_mean)
            ELI_list.append(ELI_mean)
        df_new = pd.DataFrame()
        df_new['max_scale'] = lag_list
        df_new['ELI'] = ELI_list
        bins = np.linspace(-0.6, 0.6, 20)
        df_group, bins_list_str = T.df_bin(df_new, 'ELI', bins)
        x_list = []
        y_list = []
        err_list = []
        for name,df_group_i in df_group:
            vals = df_group_i['max_scale'].tolist()
            mean = np.nanmean(vals)
            err,_,_ = T.uncertainty_err(vals)
            x_list.append(name.left)
            y_list.append(mean)
            err_list.append(err)
        plt.figure(figsize=(6,3))
        plt.errorbar(x_list, y_list, yerr=err_list)
        plt.xlabel('ELI (energy-limited --> water-limited)')
        plt.ylabel('SPEI scale')
        plt.tight_layout()

        # outf = join(outdir, 'scale_ELI.png')
        outf = join(outdir, 'scale_ELI.pdf')
        plt.savefig(outf, dpi=300)

    def scale_AI(self):
        outdir = join(self.this_class_png, 'scale_AI')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        T.print_head_n(df, 5)
        df_group_dict = T.df_groupby(df, 'pix')

        lag_list = []
        ELI_list = []
        for pix in df_group_dict:
            df_pix = df_group_dict[pix]
            lag = df_pix['max_scale']
            ELI = df_pix['aridity_index']
            lag_mean = np.nanmean(lag)
            ELI_mean = np.nanmean(ELI)
            lag_list.append(lag_mean)
            ELI_list.append(ELI_mean)
        df_new = pd.DataFrame()
        df_new['max_scale'] = lag_list
        df_new['aridity_index'] = ELI_list
        bins = np.linspace(0, 3, 20)
        df_group, bins_list_str = T.df_bin(df_new, 'aridity_index', bins)
        x_list = []
        y_list = []
        err_list = []
        for name,df_group_i in df_group:
            vals = df_group_i['max_scale'].tolist()
            mean = np.nanmean(vals)
            err,_,_ = T.uncertainty_err(vals)
            x_list.append(name.left)
            y_list.append(mean)
            err_list.append(err)
        plt.figure(figsize=(6,3))
        plt.errorbar(x_list, y_list, yerr=err_list)
        plt.xlabel('AI (Arid --> Humid)')
        plt.ylabel('SPEI scale')
        plt.tight_layout()

        outf = join(outdir, 'scale_ELI.png')
        plt.savefig(outf, dpi=300)

    def rt_rs_ELI(self):
        outdir = join(self.this_class_png, 'rt_rs_ELI')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        T.print_head_n(df, 5)
        drought_type_list = global_drought_type_list
        rs_rt_var_list = GLobal_var().get_rs_rt_cols()
        for drt in drought_type_list:
            df_drt = df[df['drought_type'] == drt]
            df_group_dict = T.df_groupby(df_drt, 'pix')
            for col in rs_rt_var_list:
                vals_list = []
                ELI_list = []
                for pix in tqdm(df_group_dict,desc=f'{drt} {col}'):
                    df_pix = df_group_dict[pix]
                    vals = df_pix[col]
                    ELI = df_pix['ELI']
                    vals_mean = np.nanmean(vals)
                    ELI_mean = np.nanmean(ELI)
                    vals_list.append(vals_mean)
                    ELI_list.append(ELI_mean)
                df_new = pd.DataFrame()
                df_new[col] = vals_list
                df_new['ELI'] = ELI_list
                bins = np.linspace(-0.6, 0.6, 20)
                df_group, bins_list_str = T.df_bin(df_new, 'ELI', bins)
                x_list = []
                y_list = []
                err_list = []
                for name, df_group_i in df_group:
                    vals = df_group_i[col].tolist()
                    mean = np.nanmean(vals)
                    err, _, _ = T.uncertainty_err(vals)
                    x_list.append(name.left)
                    y_list.append(mean)
                    err_list.append(err)
                plt.figure(figsize=(6, 3))
                plt.errorbar(x_list, y_list, yerr=err_list)
                plt.xlabel('ELI (energy-limited --> water-limited)')
                plt.title(f'{drt} {col}')
                if col == 'rt':
                    plt.ylim(0.89,1.05)
                else:
                    plt.ylim(0.95, 1.05)
                outf = join(outdir, f'{drt}_{col}.png')
                plt.tight_layout()
                plt.savefig(outf, dpi=300)
                plt.close()

    def rt_rs_AI(self):
        outdir = join(self.this_class_png, 'rt_rs_AI')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        T.print_head_n(df, 5)
        drought_type_list = global_drought_type_list
        rs_rt_var_list = GLobal_var().get_rs_rt_cols()
        for drt in drought_type_list:
            df_drt = df[df['drought_type'] == drt]
            df_group_dict = T.df_groupby(df_drt, 'pix')
            for col in rs_rt_var_list:
                vals_list = []
                ELI_list = []
                for pix in tqdm(df_group_dict,desc=f'{drt} {col}'):
                    df_pix = df_group_dict[pix]
                    vals = df_pix[col]
                    ELI = df_pix['aridity_index']
                    vals_mean = np.nanmean(vals)
                    ELI_mean = np.nanmean(ELI)
                    vals_list.append(vals_mean)
                    ELI_list.append(ELI_mean)
                df_new = pd.DataFrame()
                df_new[col] = vals_list
                df_new['aridity_index'] = ELI_list
                bins = np.linspace(0, 3, 20)
                df_group, bins_list_str = T.df_bin(df_new, 'aridity_index', bins)
                x_list = []
                y_list = []
                err_list = []
                for name, df_group_i in df_group:
                    vals = df_group_i[col].tolist()
                    mean = np.nanmean(vals)
                    err, _, _ = T.uncertainty_err(vals)
                    x_list.append(name.left)
                    y_list.append(mean)
                    err_list.append(err)
                plt.figure(figsize=(6, 3))
                plt.errorbar(x_list, y_list, yerr=err_list)
                plt.xlabel('AI (Arid --> Humid)')
                plt.title(f'{drt} {col}')
                if col == 'rt':
                    plt.ylim(0.89,1.05)
                else:
                    plt.ylim(0.95, 1.05)
                outf = join(outdir, f'{drt}_{col}.png')
                plt.tight_layout()
                plt.savefig(outf, dpi=300)
                plt.close()

    def max_r_ELI(self):
        outdir = join(self.this_class_png, 'max_r_ELI')
        T.mk_dir(outdir)
        df = GLobal_var().load_df()
        df_group_dict = T.df_groupby(df, 'pix')

        lag_list = []
        ELI_list = []
        spatial_dict = {}
        for pix in df_group_dict:
            df_pix = df_group_dict[pix]
            lag = df_pix['max_r']
            ELI = df_pix['ELI']
            lag_mean = np.nanmean(lag)
            ELI_mean = np.nanmean(ELI)
            lag_list.append(lag_mean)
            ELI_list.append(ELI_mean)
            spatial_dict[pix] = lag_mean
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        # plt.imshow(arr, cmap='jet',vmin=0,vmax=1)
        # plt.colorbar()
        # plt.show()

        df_new = pd.DataFrame()
        df_new['max_r'] = lag_list
        df_new['ELI'] = ELI_list
        bins = np.linspace(-0.6, 0.6, 20)
        df_group, bins_list_str = T.df_bin(df_new, 'ELI', bins)
        x_list = []
        y_list = []
        err_list = []
        for name,df_group_i in df_group:
            vals = df_group_i['max_r'].tolist()
            mean = np.nanmean(vals)
            err,_,_ = T.uncertainty_err(vals)
            # err = np.nanstd(vals)
            x_list.append(name.left)
            y_list.append(mean)
            err_list.append(err)
        plt.figure(figsize=(6,3))
        plt.errorbar(x_list, y_list, yerr=err_list)
        plt.xlabel('ELI (energy-limited --> water-limited)')
        plt.ylabel('Correlation')
        plt.tight_layout()

        outf = join(outdir, 'r_ELI.pdf')
        plt.savefig(outf, dpi=300)

class PFTs_and_koppen:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'PFTs_and_koppen',
            result_root_this_script, mode=2)
        pass

    def run(self):
        self.pfts_koppen_scatter()
        pass

    def pfts_koppen_scatter(self):
        outdir = join(self.this_class_png, 'pfts_koppen_scatter')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        df = Load_dataframe().load_chapter3()

        VIs_list = global_VIs_list
        lc_list = global_lc_list
        koppen_list = global_koppen_list
        lc_marker_dict = global_lc_marker_dict
        koppen_color_dict = global_koppen_color_dict

        col_x_list = [
            # 'aridity_index',
            'ELI',
            # 'ISO_Hydricity',
        ]

        col_y_list = []
        for VI in VIs_list:
            col_y_list.append(f'{VI}_max_lag')
            col_y_list.append(f'{VI}_max_r')
            col_y_list.append(f'{VI}_max_scale')

        col_y_list_pos = {
            'NDVI_max_r':(1,1),
            'VOD-anomaly_max_r':(1,2),
            'CSIF-anomaly_max_r':(1,3),

            'NDVI_max_lag':(2,1),
            'VOD-anomaly_max_lag':(2,2),
            'CSIF-anomaly_max_lag':(2,3),

            'NDVI_max_scale':(3,1),
            'VOD-anomaly_max_scale':(3,2),
            'CSIF-anomaly_max_scale':(3,3),
        }
        y_lim_dict = {
            1:(-0.1,0.8),
            2:(0,6),
            3:(0,24),
        }
        # print(col_y_list)
        # exit()
        for col_x in col_x_list:
            # plt.figure(figsize=(10,10))
            fig,axs = plt.subplots(3,3,figsize=(10,10))
            for col_y in col_y_list:
                lc_col = 'landcover_GLC'
                koppen_col = 'Koppen'
                ax_ind = col_y_list_pos[col_y]
                ax = axs[ax_ind[0]-1,ax_ind[1]-1]
                ylim = y_lim_dict[ax_ind[0]]
                xx = []
                yy = []
                for lc in lc_list:
                    df_lc = df[df[lc_col] == lc]
                    for kp in koppen_list:
                        df_kp = df_lc[df_lc[koppen_col] == kp]
                        if len(df_kp) <= 100:
                            continue
                        x = df_kp[col_x]
                        y = df_kp[col_y]
                        x = np.array(x)
                        y = np.array(y)
                        # x_err = T.uncertainty_err(x)[0]
                        # y_err = T.uncertainty_err(y)[0]
                        x_err = np.nanstd(x)
                        y_err = np.nanstd(y)
                        x_mean = np.nanmean(x)
                        y_mean = np.nanmean(y)
                        xx.append(x_mean)
                        yy.append(y_mean)
                        ax.errorbar(x_mean, y_mean, xerr=x_err, yerr=y_err,color='gray', alpha=0.5,zorder=-99)
                        ax.scatter(x_mean, y_mean, marker=lc_marker_dict[lc], color=koppen_color_dict[kp], label=f'{kp}-{lc}',edgecolors='k',zorder=99,s=70)
                        # ax.text(x_mean, y_mean, f'{lc} {kp}', fontsize=8)
                # plt.legend()
                sns.regplot(xx, yy, scatter=False, color='gray',ax=ax)
                a,b,r,p = T.nan_line_fit(xx,yy)
                # print(a,b,r,p)
                text1 = f'r2={r**2:.2f} p={p:.2f}'
                text2 = f'y={a:.2f}x+{b:.2f}'
                ax.text(0.05, 0.95, text1, transform=ax.transAxes, fontsize=10, verticalalignment='top')
                ax.text(0.05, 0.9, text2, transform=ax.transAxes, fontsize=10, verticalalignment='top')
                # plt.title(drt)
                ax.set_xlabel(col_x)
                ax.set_ylabel(col_y)
                ax.set_ylim(ylim)
                # plt.ylim(0.91, 1.02)
            plt.tight_layout()
            # outf = join(outdir, f'{col_x}.pdf')
            outf = join(outdir, f'{col_x}_with_formular.png')
            plt.savefig(outf, dpi=600)
            plt.close()
            # plt.show()

class MAT_MAP:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'MAT_MAP',
            result_root_this_script, mode=2)
        pass

    def run(self):
        self.matrix()
        pass

    def matrix(self):
        outdir = join(self.this_class_png, 'matrix')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        df = Load_dataframe().load_chapter3()
        VIs_list = global_VIs_list

        col_z_list = []
        for VI in VIs_list:
            col_z_list.append(f'{VI}_max_lag')
            col_z_list.append(f'{VI}_max_r')
            col_z_list.append(f'{VI}_max_scale')
        z_lim_dict = {
            1: (-0.1, 0.5),
            2: (0, 6),
            3: (0, 24),
        }
        z_cmap = {
            1: global_cmap_r,
            2: global_cmap,
            3: global_cmap,
        }
        col_z_list_pos = {
            'NDVI_max_r':(1,1),
            'VOD-anomaly_max_r':(1,2),
            'CSIF-anomaly_max_r':(1,3),

            'NDVI_max_lag':(2,1),
            'VOD-anomaly_max_lag':(2,2),
            'CSIF-anomaly_max_lag':(2,3),

            'NDVI_max_scale':(3,1),
            'VOD-anomaly_max_scale':(3,2),
            'CSIF-anomaly_max_scale':(3,3),
        }
        temp_bins = np.arange(-20, 20, 1)
        precip_bins = np.arange(0, 1500, 50)
        fig,axs = plt.subplots(3,3,figsize=(20,30))
        for col_z in col_z_list:
            lc_col = 'landcover_GLC'
            koppen_col = 'Koppen'
            ax_ind = col_z_list_pos[col_z]
            ax = axs[ax_ind[0]-1,ax_ind[1]-1]
            # ax = plt.subplot(1,1,1)
            df_group_t,bins_list_str_t = T.df_bin(df,'MAT',temp_bins)
            matrix = []
            for name_t,df_group_t_i in df_group_t:
                df_group_p,bins_list_str_p = T.df_bin(df_group_t_i,'MAP',precip_bins)
                temp = []
                for name_p,df_group_p_i in df_group_p:
                    vals = df_group_p_i[col_z].tolist()
                    mean = np.nanmean(vals)
                    temp.append(mean)
                matrix.append(temp)
            matrix = np.array(matrix)
            ret = ax.imshow(matrix, cmap=z_cmap[ax_ind[0]],vmin=z_lim_dict[ax_ind[0]][0],vmax=z_lim_dict[ax_ind[0]][1])
            ax.set_xticks(np.arange(len(bins_list_str_p)),bins_list_str_p,rotation=90)
            ax.set_yticks(np.arange(len(bins_list_str_t)),bins_list_str_t)
            ax.set_xlabel('MAP')
            ax.set_ylabel('MAT')
            ax.set_title(col_z)
            # plt.colorbar(ret)
            plt.tight_layout()
        plt.tight_layout()
        # outf = join(outdir, f'{col_x}.pdf')
        outf = join(outdir, f'matrix.pdf')
        plt.savefig(outf, dpi=600)
        plt.close()
        # plt.show()

class ELI:

    def __init__(self):

        pass

    def run(self):

        pass

class AI:

    def __init__(self):

        pass

    def run(self):

        pass

class Isohydricity:

    def __init__(self):

        pass

    def run(self):

        pass

class Latitude:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Latitude',
            result_root_this_script, mode=2)
        pass

    def run(self):
        self.plot_line()
        pass

    def plot_line(self):
        outdir = join(self.this_class_png, 'matrix')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        df = Load_dataframe().load_chapter3()
        VIs_list = global_VIs_list
        y_lim_dict = {
            '_max_lag': (0, 6),
            '_max_r': (-0.2, 0.6),
            '_max_scale': (0, 24),

        }

        suffix_list = ['_max_lag','_max_r','_max_scale']
        lag_bins = np.arange(30, 90, 1)
        for suffix in suffix_list:
            plt.figure(figsize=(10, 3))
            for VI in VIs_list:
                col_z = f'{VI}{suffix}'
                df_group,bins_list_str = T.df_bin(df,'lat',lag_bins)
                y_list = []
                err_list = []
                x_list = []
                for name,df_group_i in df_group:
                    x_list.append(name.left)
                    vals = df_group_i[col_z].tolist()
                    mean = np.nanmean(vals)
                    err,_,_ = T.uncertainty_err(vals)
                    # err = np.nanstd(vals)
                    # x_list.append(name)
                    y_list.append(mean)
                    err_list.append(err)
                plt.plot(x_list,y_list,label=col_z)
                plt.fill_between(x_list, np.array(y_list)-np.array(err_list), np.array(y_list)+np.array(err_list), alpha=0.5)
                plt.xlabel('Latitude')
                plt.ylabel(col_z)
                plt.title(col_z)
                plt.tight_layout()
            plt.legend()
            plt.grid()
            plt.ylim(y_lim_dict[suffix])
            outf = join(outdir, f'{suffix}.pdf')
            plt.savefig(outf, dpi=600)
            plt.close()
            # plt.show()

            pass


        pass

class Temporal:

    pass


def main():
    # Dataframe().run()
    # Constant_value().run()
    # PFTs_and_koppen().run()
    # MAT_MAP().run()
    Latitude().run()
    pass


if __name__ == '__main__':
    main()
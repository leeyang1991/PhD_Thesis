# coding=utf-8
import typer.main

from meta_info import *
result_root_this_script = join(results_root, 'Chapter2/analysis')



class Plot_SPEI:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Plot_SPEI',
            result_root_this_script, mode=2)
        pass

    def run(self):
        outdir = join(self.this_class_png, 'SPEI')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        SPEI_data_dict = Meta_information().load_data('SPEI')
        lon_list,lat_list = [-101],[38.5]
        pix_list = DIC_and_TIF().lon_lat_to_pix(lon_list,lat_list)
        date_list = []
        for y in global_year_range_list:
            for m in range(1,13):
                date = f'{y}'
                date_list.append(date)
        for pix in pix_list:
            matrix = []
            y = []
            for scale in SPEI_data_dict:
                print(scale)
                spatial_dict = SPEI_data_dict[scale]
                SPEI = spatial_dict[pix]
                matrix.append(SPEI)
                y.append(scale)
            matrix = np.array(matrix)
            plt.figure(figsize=(18*centimeter_factor,6*centimeter_factor))
            plt.imshow(matrix,vmin=-2,vmax=2,cmap=global_cmap_r,aspect='auto')
            plt.yticks(range(len(y)),y)
            plt.xticks(list(range(0,len(date_list),12*5)),date_list[::12*5])
            plt.colorbar()
            outf = join(outdir,f'{pix}.pdf')
            plt.savefig(outf)
            plt.close()

        pass

class Plot_ELI:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Plot_ELI',
            result_root_this_script, mode=2)
        pass

    def run(self):

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
        spatial_dict_1 = Meta_information().load_data(var_1)
        spatial_dict_2 = Meta_information().load_data(var_1)
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
        # var_22 = 'CCI-SM'
        var_3 = 'Temperature'
        # var_31 = 'Radiation'
        fdir = join(self.this_class_arr, 'kendall_corr')
        outdir = join(self.this_class_tif, 'ELI')
        T.mk_dir(outdir)
        outf = join(outdir, f'{var_1}_{var_21}_{var_3}.tif')
        # outf = join(outdir, f'{var_1}_{var_21}_{var_31}.tif')
        ELI_equation = 'ELI = corr(ET,SM) - corr(ET,T)'
        dff1 = join(fdir, f'{var_1}_{var_21}.df')
        dff2 = join(fdir,f'{var_1}_{var_3}.df')
        # dff2 = join(fdir, f'{var_1}_{var_31}.df')
        df1 = T.load_df(dff1)
        df2 = T.load_df(dff2)
        spatial_dict1 = T.df_to_spatial_dic(df1, f'{var_1}_{var_21}_r')
        spatial_dict2 = T.df_to_spatial_dic(df2,f'{var_1}_{var_3}_r')
        # spatial_dict2 = T.df_to_spatial_dic(df2, f'{var_1}_{var_31}_r')

        ELI_spatial_dict = {}
        for pix in tqdm(spatial_dict1):
            if not pix in spatial_dict2:
                continue
            val1 = spatial_dict1[pix]
            val2 = spatial_dict2[pix]
            ELI = val1 - val2
            ELI_spatial_dict[pix] = ELI

        DIC_and_TIF().pix_dic_to_tif(ELI_spatial_dict, outf)

class NDVI_average:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'NDVI_average',
            result_root_this_script, mode=2)
        pass

    def run(self):
        # self.NDVI_8km_MVC()
        self.average()
        pass

    def NDVI_8km_MVC(self):
        NDVI_dir = join(data_root, 'GIMMS_NDVI/tif_8km_bi_weekly')
        outdir = join(self.this_class_tif, 'NDVI_8km_MVC')
        Pre_Process().monthly_compose(NDVI_dir, outdir, method='max')
        pass

    def average(self):
        fdir = join(self.this_class_tif, 'NDVI_8km_MVC')
        outdir = join(self.this_class_tif, 'NDVI_8km_MVC_average')
        T.mk_dir(outdir)
        outf = join(outdir, '1982_GS.tif')
        flist = []
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            flist.append(join(fdir, f))
        Pre_Process().compose_tif_list(flist, outf, method='mean')

        pass

def main():
    # Plot_SPEI().run()
    # Plot_ELI().run()
    NDVI_average().run()

if __name__ == '__main__':
    main()
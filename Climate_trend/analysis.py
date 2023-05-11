# coding=utf-8
import matplotlib.pyplot as plt

from meta_info import *
result_root_this_script = join(results_root, 'Climate_trend/analysis')


class Trend_comparison:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Trend_comparison',
            result_root_this_script, mode=2)
        self.precip_product_list = [
            'Precipitation-origin',
            'ERA_precip-origin',
            'GPCC-origin',
        ]
        self.temp_product_list = [
            'Temperature-origin',
            'ERA_Ta2m-origin',
            'BEST-anomaly',
        ]
        pass

    def run(self):
        # self.land_pix()
        # self.precip_trend()
        self.plot_precip_trend()
        # self.temp_trend()
        # self.plot_temp_trend()

        pass

    def precip_trend(self):
        outdir = join(self.this_class_tif, 'precip_trend')
        T.mk_dir(outdir)
        precip_product_list = self.precip_product_list
        land_pix = self.land_pix()
        gs = global_gs
        for product in precip_product_list:
            data_dict = Meta_information().load_data(product)
            k_dict = {}
            p_dict = {}
            for pix in tqdm(data_dict,desc=product):
                r,c = pix
                if r > 121:
                    continue
                if not pix in land_pix:
                    continue
                vals = data_dict[pix]
                # print(vals)
                vals_gs = T.monthly_vals_to_annual_val(vals,grow_season=gs)
                try:
                    a,b,r,p = T.nan_line_fit(list(range(len(vals_gs))),vals_gs)
                except:
                    continue
                k_dict[pix] = a
                p_dict[pix] = p
            outf_k = join(outdir,f'{product}_k.tif')
            outf_p = join(outdir,f'{product}_p.tif')
            DIC_and_TIF().pix_dic_to_tif(k_dict,outf_k)
            DIC_and_TIF().pix_dic_to_tif(p_dict,outf_p)

    def plot_precip_trend(self):
        outdir = join(self.this_class_png,'precip_trend')
        T.mk_dir(outdir)
        temp_root = join(this_root,'temp')
        fdir = join(self.this_class_tif, 'precip_trend')
        product_list = self.precip_product_list

        vmin_max_dict = {
            'Precipitation-origin':[-1,1],
            'ERA_precip-origin':[-0.000035,0.000035],
            'GPCC-origin':[-1,1],
        }
        for product in product_list:
            vmin,vmax = vmin_max_dict[product]
            fpath_k = join(fdir,f'{product}_k.tif')
            fpath_p = join(fdir,f'{product}_p.tif')
            m,ret = Plot().plot_ortho(fpath_k,vmin=vmin,vmax=vmax,cmap=global_cmap)
            Plot().plot_ortho_significance_scatter(m,fpath_p,temp_root,s=1,marker='o')
            plt.title(product)
            # plt.show()
            outf = join(outdir,f'{product}.png')
            plt.savefig(outf,dpi=300)
            plt.close()
        T.open_path_and_file(outdir)

    def temp_trend(self):
        outdir = join(self.this_class_tif, 'temp_trend')
        T.mk_dir(outdir)
        product_list = self.temp_product_list
        land_pix = self.land_pix()
        gs = global_gs
        for product in product_list:
            data_dict = Meta_information().load_data(product)
            k_dict = {}
            p_dict = {}
            for pix in tqdm(data_dict,desc=product):
                r,c = pix
                if r > 121:
                    continue
                if not pix in land_pix:
                    continue
                vals = data_dict[pix]
                # print(vals)
                vals_gs = T.monthly_vals_to_annual_val(vals,grow_season=gs)
                try:
                    a,b,r,p = T.nan_line_fit(list(range(len(vals_gs))),vals_gs)
                except:
                    continue
                k_dict[pix] = a
                p_dict[pix] = p
            outf_k = join(outdir,f'{product}_k.tif')
            outf_p = join(outdir,f'{product}_p.tif')
            DIC_and_TIF().pix_dic_to_tif(k_dict,outf_k)
            DIC_and_TIF().pix_dic_to_tif(p_dict,outf_p)

    def plot_temp_trend(self):
        outdir = join(self.this_class_png,'temp_trend')
        T.mk_dir(outdir)
        temp_root = join(this_root,'temp')
        fdir = join(self.this_class_tif, 'temp_trend')
        product_list = self.temp_product_list

        for product in product_list:
            vmin,vmax = -0.1,0.1
            fpath_k = join(fdir,f'{product}_k.tif')
            fpath_p = join(fdir,f'{product}_p.tif')
            m,ret = Plot().plot_ortho(fpath_k,vmin=vmin,vmax=vmax)
            Plot().plot_ortho_significance_scatter(m,fpath_p,temp_root,s=1,marker='o')
            plt.title(product)
            # plt.show()
            outf = join(outdir,f'{product}.png')
            plt.savefig(outf,dpi=300)
            plt.close()
        T.open_path_and_file(outdir)

    def land_pix(self):

        land_tif = join(this_root,'conf','land.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(land_tif)
        land_pix_dict = {}
        for pix in spatial_dict:
            val = spatial_dict[pix]
            if np.isnan(val):
                continue
            land_pix_dict[pix] = val
        return land_pix_dict

class Correlation_comparison:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Correlation_comparison',
            result_root_this_script, mode=2)

        self.precip_x_product = 'Precipitation-origin'
        self.temp_x_product = 'Temperature-origin'
        self.precip_product_list = [
            'ERA_precip-origin',
            'GPCC-origin',
        ]
        self.temp_product_list = [
            'ERA_Ta2m-origin',
            'BEST-anomaly',
        ]
        pass

    def run(self):
        # self.precip_correlation()
        # self.plot_precip_correlation()
        # self.temp_correlation()
        self.plot_temp_correlation()

        pass

    def precip_correlation(self):
        outdir = join(self.this_class_tif, 'precip_correlation')
        T.mk_dir(outdir)
        product_list = self.precip_product_list
        land_pix = self.land_pix()
        gs = global_gs
        for product in product_list:
            data_dict = Meta_information().load_data(product)
            data_dict_x = Meta_information().load_data(self.precip_x_product)
            r_dict = {}
            p_dict = {}
            for pix in tqdm(data_dict,desc=product):
                r,c = pix
                if r > 121:
                    continue
                if not pix in land_pix:
                    continue
                if not pix in data_dict_x:
                    continue
                vals = data_dict[pix]
                vals_x = data_dict_x[pix]
                # print(vals)
                vals_gs = T.monthly_vals_to_annual_val(vals,grow_season=gs)
                valsx_gs = T.monthly_vals_to_annual_val(vals_x,grow_season=gs)
                try:
                    a,b,r,p = T.nan_line_fit(valsx_gs,vals_gs)
                except:
                    continue
                r_dict[pix] = r
                p_dict[pix] = p
            outf_k = join(outdir,f'{product}_r.tif')
            outf_p = join(outdir,f'{product}_p.tif')
            DIC_and_TIF().pix_dic_to_tif(r_dict,outf_k)
            DIC_and_TIF().pix_dic_to_tif(p_dict,outf_p)

    def plot_precip_correlation(self):
        outdir = join(self.this_class_png,'precip_correlation')
        T.mk_dir(outdir)
        temp_root = join(this_root,'temp')
        fdir = join(self.this_class_tif, 'precip_correlation')
        product_list = self.precip_product_list

        for product in product_list:
            vmin,vmax = -1,1
            fpath_k = join(fdir,f'{product}_r.tif')
            fpath_p = join(fdir,f'{product}_p.tif')
            m,ret = Plot().plot_ortho(fpath_k,vmin=vmin,vmax=vmax,cmap='RdBu_r')
            Plot().plot_ortho_significance_scatter(m,fpath_p,temp_root,s=1,marker='o')
            plt.title(product)
            # plt.show()
            outf = join(outdir,f'{product}.png')
            plt.savefig(outf,dpi=300)
            plt.close()
        T.open_path_and_file(outdir)

    def temp_correlation(self):
        outdir = join(self.this_class_tif, 'temp_correlation')
        T.mk_dir(outdir)
        product_list = self.temp_product_list
        land_pix = self.land_pix()
        gs = global_gs
        for product in product_list:
            data_dict = Meta_information().load_data(product)
            data_dict_x = Meta_information().load_data(self.temp_x_product)
            r_dict = {}
            p_dict = {}
            for pix in tqdm(data_dict,desc=product):
                r,c = pix
                if r > 121:
                    continue
                if not pix in land_pix:
                    continue
                if not pix in data_dict_x:
                    continue
                vals = data_dict[pix]
                vals_x = data_dict_x[pix]
                # print(vals)
                vals_gs = T.monthly_vals_to_annual_val(vals,grow_season=gs)
                valsx_gs = T.monthly_vals_to_annual_val(vals_x,grow_season=gs)
                try:
                    a,b,r,p = T.nan_line_fit(valsx_gs,vals_gs)
                except:
                    continue
                r_dict[pix] = r
                p_dict[pix] = p
            outf_k = join(outdir,f'{product}_r.tif')
            outf_p = join(outdir,f'{product}_p.tif')
            DIC_and_TIF().pix_dic_to_tif(r_dict,outf_k)
            DIC_and_TIF().pix_dic_to_tif(p_dict,outf_p)

    def plot_temp_correlation(self):
        outdir = join(self.this_class_png,'temp_correlation')
        T.mk_dir(outdir)
        temp_root = join(this_root,'temp')
        fdir = join(self.this_class_tif, 'temp_correlation')
        product_list = self.temp_product_list

        for product in product_list:
            vmin,vmax = -1,1
            fpath_k = join(fdir,f'{product}_r.tif')
            fpath_p = join(fdir,f'{product}_p.tif')
            m,ret = Plot().plot_ortho(fpath_k,vmin=vmin,vmax=vmax,cmap='RdBu_r')
            Plot().plot_ortho_significance_scatter(m,fpath_p,temp_root,s=1,marker='o')
            plt.title(product)
            # plt.show()
            outf = join(outdir,f'{product}.png')
            plt.savefig(outf,dpi=300)
            plt.close()
        T.open_path_and_file(outdir)

    def land_pix(self):

        land_tif = join(this_root,'conf','land.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(land_tif)
        land_pix_dict = {}
        for pix in spatial_dict:
            val = spatial_dict[pix]
            if np.isnan(val):
                continue
            land_pix_dict[pix] = val
        return land_pix_dict


class Time_series_comparison:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Time_series_comparison',
            result_root_this_script, mode=2)
        self.precip_product_list = [
            'Precipitation-origin',
            'ERA_precip-origin',
            'GPCC-origin',
        ]
        self.temp_product_list = [
            'Temperature-origin',
            'ERA_Ta2m-origin',
            'BEST-anomaly',
        ]
        pass

    def run(self):
        # self.precip_dataframe()
        # self.temp_dataframe()
        self.precip_ts()
        pass

    def precip_dataframe(self):
        from Chapter3 import statistic
        outdir = join(self.this_class_arr,'precip_dataframe')
        T.mk_dir(outdir)
        data_dict_all = {}
        for product in self.precip_product_list:
            data_dict = Meta_information().load_data(product)
            data_dict_all[product] = data_dict
        df = T.spatial_dics_to_df(data_dict_all)
        df = statistic.Dataframe_func(df).df
        outf = join(outdir,'precip_dataframe.df')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

        pass

    def temp_dataframe(self):
        from Chapter3 import statistic
        outdir = join(self.this_class_arr,'temp_dataframe')
        T.mk_dir(outdir)
        data_dict_all = {}
        for product in self.temp_product_list:
            data_dict = Meta_information().load_data(product)
            data_dict_all[product] = data_dict
        df = T.spatial_dics_to_df(data_dict_all)
        df = statistic.Dataframe_func(df).df
        outf = join(outdir,'temp_dataframe.df')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

        pass

    def precip_ts(self):
        dff = join(self.this_class_arr,'precip_dataframe','precip_dataframe.df')
        df = T.load_df(dff)
        T.print_head_n(df)
        product_list = self.precip_product_list
        pix_list = df['pix'].tolist()
        for product in product_list:
            for i,row in tqdm(df.iterrows(),total=len(df),desc=product):
                pix = row['pix']
                vals = row[product]
                vals = T.monthly_vals_to_annual_val(vals)
                plt.plot(vals)
                plt.title(f'{product} {pix}')
                plt.show()
                pass


def main():
    # Trend_comparison().run()
    # Correlation_comparison().run()
    Time_series_comparison().run()
    pass

if __name__ == '__main__':
    main()
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
            # 'Precipitation-origin',
            'ERA_precip-origin',
            # 'GPCC-origin',
        ]
        self.temp_product_list = [
            'Temperature-origin',
            'ERA_Ta2m-origin',
            'BEST-anomaly',
        ]
        pass

    def run(self):
        # self.land_pix()
        self.precip_trend()
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
            # 'ERA_precip-origin':[-0.000035,0.000035],
            'ERA_precip-origin':[-1,1],
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
        # self.precip_ts()
        self.temp_ts()
        pass

    def precip_dataframe(self):
        from Chapter3 import statistic
        product_list = self.precip_product_list
        gs = global_gs
        outdir = join(self.this_class_arr,'precip_dataframe')
        T.mk_dir(outdir)
        data_dict_all = {}
        for product in self.precip_product_list:
            data_dict = Meta_information().load_data(product)
            data_dict_all[product] = data_dict
        df = T.spatial_dics_to_df(data_dict_all)
        df = statistic.Dataframe_func(df).df
        for product in product_list:
            vals_list = []
            for i,row in tqdm(df.iterrows(),total=len(df),desc=product):
                pix = row['pix']
                vals = row[product]
                try:
                    vals = T.monthly_vals_to_annual_val(vals,grow_season=gs)
                except:
                    vals = np.nan
                vals_list.append(vals)
            df[product] = vals_list
        outf = join(outdir,'precip_dataframe.df')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

        pass

    def temp_dataframe(self):
        from Chapter3 import statistic
        outdir = join(self.this_class_arr,'temp_dataframe')
        T.mk_dir(outdir)
        product_list = self.temp_product_list
        gs = global_gs
        data_dict_all = {}
        for product in self.temp_product_list:
            data_dict = Meta_information().load_data(product)
            data_dict_all[product] = data_dict
        df = T.spatial_dics_to_df(data_dict_all)
        df = statistic.Dataframe_func(df).df
        for product in product_list:
            vals_list = []
            for i,row in tqdm(df.iterrows(),total=len(df),desc=product):
                pix = row['pix']
                vals = row[product]
                try:
                    vals = T.monthly_vals_to_annual_val(vals,grow_season=gs)
                except:
                    vals = np.nan
                vals_list.append(vals)
            df[product] = vals_list
        outf = join(outdir,'temp_dataframe.df')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

        pass


    def precip_ts(self):
        outdir = join(self.this_class_png,'precip_ts')
        T.mk_dir(outdir)
        gs = global_gs
        dff = join(self.this_class_arr,'precip_dataframe','precip_dataframe.df')
        df = T.load_df(dff)
        T.print_head_n(df)
        product_list = self.precip_product_list
        pix_list = df['pix'].tolist()
        date_list = []
        for y in range(global_start_year,global_end_year+1):
            date_list.append(y)

        y_mean_dict = {}
        for product in product_list:
            y_list = []
            for i,row in tqdm(df.iterrows(),total=len(df),desc=product):
                pix = row['pix']
                vals = row[product]
                if type(vals) == float:
                    continue
                y_list.append(vals)
            y_list = np.array(y_list)
            y_mean = np.nanmean(y_list,axis=0)
            y_std = T.uncertainty_err_2d(y_list,axis=0)
            y_mean_dict[product] = (y_mean,y_std)
        plt.figure(figsize=(8.8*centimeter_factor,5*centimeter_factor))
        for product in product_list:
            y_mean,y_std = y_mean_dict[product]
            y_mean = y_mean - np.nanmean(y_mean)
            plt.plot(date_list,y_mean,label=product)
            plt.fill_between(date_list,y_mean-y_std,y_mean+y_std,alpha=0.3)
        plt.legend()
        plt.xlim(global_start_year,global_end_year)
        # plt.show()
        outf = join(outdir,'precip_ts.pdf')
        plt.savefig(outf)
        plt.close()

    def temp_ts(self):
        outdir = join(self.this_class_png,'temp_ts')
        T.mk_dir(outdir)
        gs = global_gs
        dff = join(self.this_class_arr,'temp_dataframe','temp_dataframe.df')
        df = T.load_df(dff)
        T.print_head_n(df)
        product_list = self.temp_product_list
        pix_list = df['pix'].tolist()
        date_list = []
        for y in range(global_start_year,global_end_year+1):
            date_list.append(y)

        y_mean_dict = {}
        for product in product_list:
            y_list = []
            spatial_dict = {}
            for i,row in tqdm(df.iterrows(),total=len(df),desc=product):
                pix = row['pix']
                vals = row[product]
                if type(vals) == float:
                    continue
                vals = T.mask_999999_arr(vals,warning=False)
                y_list.append(vals)

            y_list = np.array(y_list)
            y_mean = np.nanmean(y_list,axis=0)
            y_std = T.uncertainty_err_2d(y_list,axis=0)
            y_mean_dict[product] = (y_mean,y_std)
        plt.figure(figsize=(8.8*centimeter_factor,5*centimeter_factor))
        for product in product_list:
            y_mean,y_std = y_mean_dict[product]
            y_mean = y_mean - np.nanmean(y_mean)
            plt.plot(date_list,y_mean,label=product)
            plt.fill_between(date_list,y_mean-y_std,y_mean+y_std,alpha=0.3)
        plt.legend()
        plt.xlim(global_start_year,global_end_year)
        # plt.show()
        outf = join(outdir,'temp_ts.pdf')
        plt.savefig(outf)
        plt.close()

class Scatter_comparison:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Scatter_comparison',
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
        # self.precip_dataframe()
        # self.temp_dataframe()
        # self.precip_scatter_plot()
        self.temp_scatter_plot()

    def precip_dataframe(self):
        from Chapter3 import statistic
        product_list = self.precip_product_list
        gs = global_gs
        outdir = join(self.this_class_arr,'precip_dataframe')
        T.mk_dir(outdir)
        data_dict_all = {}
        for product in self.precip_product_list:
            data_dict = Meta_information().load_data(product)
            data_dict_all[product] = data_dict
        df = T.spatial_dics_to_df(data_dict_all)
        df = statistic.Dataframe_func(df).df
        for product in product_list:
            vals_list = []
            for i,row in tqdm(df.iterrows(),total=len(df),desc=product):
                pix = row['pix']
                vals = row[product]
                try:
                    vals = T.monthly_vals_to_annual_val(vals,grow_season=gs)
                except:
                    vals = np.nan
                vals = vals - np.nanmean(vals) # anomaly
                vals_list.append(vals)
            df[product] = vals_list
        outf = join(outdir,'precip_dataframe.df')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

        pass
    def temp_dataframe(self):
        from Chapter3 import statistic
        outdir = join(self.this_class_arr,'temp_dataframe')
        T.mk_dir(outdir)
        product_list = self.temp_product_list
        gs = global_gs
        data_dict_all = {}
        for product in self.temp_product_list:
            data_dict = Meta_information().load_data(product)
            data_dict_all[product] = data_dict
        df = T.spatial_dics_to_df(data_dict_all)
        df = statistic.Dataframe_func(df).df
        for product in product_list:
            vals_list = []
            for i,row in tqdm(df.iterrows(),total=len(df),desc=product):
                pix = row['pix']
                vals = row[product]
                try:
                    vals = T.monthly_vals_to_annual_val(vals,grow_season=gs)
                except:
                    vals = np.nan
                vals = vals - np.nanmean(vals) # anomaly
                vals_list.append(vals)
            df[product] = vals_list
        outf = join(outdir,'temp_dataframe.df')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

        pass


    def precip_scatter_plot(self):
        # T.color_map_choice()
        # plt.show()
        outdir = join(self.this_class_png,'precip_scatter_plot')
        T.mk_dir(outdir)

        fdir = join(Trend_comparison().this_class_tif,'precip_trend')
        product_list = copy.copy(self.precip_product_list)
        product_list.append(self.precip_x_product)
        dict_all = {}
        for product in product_list:
            fpath = join(fdir,product+'_k.tif')
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            dict_all[product] = spatial_dict
        df = T.spatial_dics_to_df(dict_all)

        for product in self.precip_product_list:
            vals_x = df[self.precip_x_product].tolist()
            vals_y = df[product].tolist()
            # plt.figure(figsize=(8.8*centimeter_factor,8.8*centimeter_factor))
            fig,ax = plt.subplots(figsize=(8.8*centimeter_factor,8.8*centimeter_factor))
            # a,b,r,p  = KDE_plot().plot_scatter(vals_x,vals_y,is_plot_1_1_line=True,plot_fit_line=True,s=4,is_equal=True,ax=ax,alpha=1,cmap='gist_yarg')
            KDE_plot().plot_scatter(vals_x,vals_y,s=4,
                                               is_equal=True,ax=ax,alpha=.7,cmap='gray_r',marker='o',edgecolors='k',facecolors='none')
            a,b,r,p = T.nan_line_fit(vals_x,vals_y)
            plt.title(product+' r2='+str(round(r*r,2))+' p='+str(round(p,2)))
            # plt.axis('equal')
            plt.xlim(-.6,.6)
            plt.ylim(-.6,.6)

            outf = join(outdir,product+'.png')
            plt.savefig(outf,dpi=300)
            plt.close()
        T.open_path_and_file(outdir)

    def temp_scatter_plot(self):
        outdir = join(self.this_class_png,'temp_scatter_plot1')
        T.mk_dir(outdir)

        fdir = join(Trend_comparison().this_class_tif,'temp_trend')
        product_list = copy.copy(self.temp_product_list)
        product_list.append(self.temp_x_product)
        dict_all = {}
        for product in product_list:
            fpath = join(fdir,product+'_k.tif')
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            dict_all[product] = spatial_dict
        df = T.spatial_dics_to_df(dict_all)

        for product in self.temp_product_list:
            vals_x = df[self.temp_x_product].tolist()
            vals_y = df[product].tolist()
            # plt.figure(figsize=(8.8*centimeter_factor,8.8*centimeter_factor))
            fig,ax = plt.subplots(figsize=(8.8*centimeter_factor,8.8*centimeter_factor))
            # a,b,r,p  = KDE_plot().plot_scatter(vals_x,vals_y,is_plot_1_1_line=True,plot_fit_line=True,s=4,is_equal=True,ax=ax,alpha=1,cmap='gist_yarg')
            KDE_plot().plot_scatter(vals_x,vals_y,s=4,is_plot_1_1_line=True,plot_fit_line=True,
                                               is_equal=True,ax=ax,alpha=.7,cmap='gray_r',marker='o',edgecolors='k',facecolors='none')
            a,b,r,p = T.nan_line_fit(vals_x,vals_y)
            plt.title(product+' r2='+str(round(r*r,2))+' p='+str(round(p,2)))
            # plt.axis('equal')
            plt.xlim(0,.09)
            plt.ylim(0,.09)
            # plt.show()

            outf = join(outdir,product+'.png')
            plt.savefig(outf,dpi=300)
            plt.close()
        T.open_path_and_file(outdir)

class SPEI_SM:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'SPEI_SM',
            result_root_this_script, mode=2)
        self.product_list = [
            'GLEAM-SMRoot-origin',
            'ERA-SM-origin',
            'CCI-SM-origin',
        ]
        pass

    def run(self):
        self.SM_trend()
        pass

    def SM_trend(self):
        land_pix = self.land_pix()
        product_list = self.product_list
        outdir = join(self.this_class_tif,'SM_trend')
        T.mk_dir(outdir)
        gs = global_gs
        for product in product_list:
            data_dict = Meta_information().load_data(product)
            spatial_dict_k = {}
            spatial_dict_p = {}
            for pix in tqdm(data_dict,desc=product):
                r, c = pix
                if r > 121:
                    continue
                if not pix in land_pix:
                    continue
                vals = data_dict[pix]
                vals = np.array(vals)
                vals[vals<0] = np.nan
                # print(vals)
                vals_gs = T.monthly_vals_to_annual_val(vals, grow_season=gs)
                try:
                    a, b, r, p = T.nan_line_fit(list(range(len(vals_gs))), vals_gs)
                except:
                    continue
                spatial_dict_k[pix] = a
                spatial_dict_p[pix] = p
            outf_k = join(outdir,product+'_k.tif')
            outf_p = join(outdir,product+'_p.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict_k,outf_k)
            DIC_and_TIF().pix_dic_to_tif(spatial_dict_p,outf_p)

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

def main():
    # Trend_comparison().run()
    # Correlation_comparison().run()
    # Time_series_comparison().run()
    # Scatter_comparison().run()
    SPEI_SM().run()
    pass

if __name__ == '__main__':
    main()
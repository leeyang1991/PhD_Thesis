# coding=utf-8

from meta_info import *
result_root_this_script = join(results_root, 'Chapter4/statistic')

class Drought_events:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Drought_events', result_root_this_script, mode=2)
        pass

    def run(self):
        # self.tif_drought_events()
        self.histogram_drought_events()
        # self.plot_drought_events()

        # self.drought_events_detail_df()

        # self.tif_drought_events_with_timing()
        # self.png_drought_events_with_timing()

        # self.tif_drought_events_duration()
        # self.png_drought_events_duration()

        # self.drought_affected_area_time_series()
        # self.png_drought_events_duration()
        pass

    def tif_drought_events(self):
        from Chapter4 import analysis
        outdir = join(self.this_class_tif,'drought_events')
        T.mk_dir(outdir)
        fpath = join(analysis.Main_flow_Pick().this_class_arr,'events_with_timing_spei/spei03/events_with_timing.npy')
        events_spatial_dict = T.load_npy(fpath)
        spatial_dict = {}
        for pix in events_spatial_dict:
            events = events_spatial_dict[pix]
            spatial_dict[pix] = len(events)
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        outf = join(outdir,'drought_events.tif')
        DIC_and_TIF().arr_to_tif(arr, outf)

    def histogram_drought_events(self):
        from Chapter4 import analysis
        outdir = join(self.this_class_png,'histogram_drought_events')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        fpath = join(analysis.Main_flow_Pick().this_class_arr,'events_with_timing_spei/spei03/events_with_timing.npy')
        events_spatial_dict = T.load_npy(fpath)
        events_number_list = []
        for pix in events_spatial_dict:
            events = events_spatial_dict[pix]
            # spatial_dict[pix] = len(events)
            events_number_list.append(len(events))
        plt.figure(figsize=(8*centimeter_factor, 6*centimeter_factor))
        plt.hist(events_number_list, bins=6, range=(1, 7), rwidth=0.8, align='left', density=True)
        plt.xlabel('Drought events number')
        plt.ylabel('Frequency')
        outf = join(outdir, 'histogram_drought_events.pdf')
        plt.savefig(outf)
        plt.close()
        # plt.show()

    def plot_drought_events(self):
        fdir = join(self.this_class_tif,'drought_events')
        outdir = join(self.this_class_png,'drought_events')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        fpath = join(fdir,'drought_events.tif')
        plt.figure(figsize=(5, 5))
        Plot().plot_ortho(fpath,vmin=0,vmax=6)
        outf = join(outdir,'drought_events.png')
        plt.savefig(outf,dpi=600)
        plt.close()

    def drought_events_detail_df(self):
        from Chapter4 import analysis
        outdir = join(self.this_class_arr,'drought_events_detail')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        fpath = join(analysis.Main_flow_Pick().this_class_arr,'events_with_timing_spei/spei03/events_with_timing.npy')
        events_spatial_dict = T.load_npy(fpath)
        pix_list = list(events_spatial_dict.keys())
        pix_list.sort()
        spatial_dict = {}
        flag = 0
        for pix in pix_list:
            events = events_spatial_dict[pix]
            flag1 = 0
            for timing,evt in events:
                evt = np.array(evt,dtype=int)
                dict_i = {}
                dict_i['pix'] = pix
                dict_i['drought_index_in_this_pix'] = flag1
                dict_i['timing'] = timing
                dict_i['evt'] = evt
                dict_i['duration'] = len(evt)
                spatial_dict[flag] = dict_i
                flag += 1
                flag1 += 1
        df = T.dic_to_df(spatial_dict,key_col_str='index')
        outf = join(outdir,'drought_events_detail.df')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

    def tif_drought_events_with_timing(self):
        outdir = join(self.this_class_tif,'drought_events_with_timing')
        T.mk_dir(outdir)
        dff = join(self.this_class_arr,'drought_events_detail/drought_events_detail.df')
        df = T.load_df(dff)
        drought_timing_list = global_drought_timing_list
        for timing in drought_timing_list:
            df_timing = df[df['timing']==timing]
            df_group = T.df_groupby(df_timing,'pix')
            spatial_dict = {}
            for pix in df_group:
                df_pix = df_group[pix]
                drought_number = len(df_pix)
                spatial_dict[pix] = drought_number
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
            outf = join(outdir,f'{timing}.tif')
            DIC_and_TIF().arr_to_tif(arr, outf)

    def png_drought_events_with_timing(self):
        fdir = join(self.this_class_tif,'drought_events_with_timing')
        outdir = join(self.this_class_png,'drought_events_with_timing')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        drought_timing_list = global_drought_timing_list
        plt.figure(figsize=(10, 4))
        flag = 1
        for timing in drought_timing_list:
            fpath = join(fdir,f'{timing}.tif')
            ax = plt.subplot(1, 3, flag)
            flag += 1
            Plot().plot_ortho(fpath,vmin=1,vmax=3,ax=ax)
        plt.tight_layout()
        outf = join(outdir,'drought_events_with_timing.png')
        plt.savefig(outf,dpi=600)
        plt.close()

    def tif_drought_events_duration(self):
        outdir = join(self.this_class_tif,'drought_events_duration')
        T.mk_dir(outdir)
        dff = join(self.this_class_arr,'drought_events_detail/drought_events_detail.df')
        df = T.load_df(dff)
        df_group = T.df_groupby(df,'pix')
        spatial_dict = {}
        for pix in df_group:
            df_pix = df_group[pix]
            evt_list = df_pix['evt'].tolist()
            evt_len_list = []
            for evt in evt_list:
                evt_len = len(evt)
                evt_len_list.append(evt_len)
            evt_len_mean = np.mean(evt_len_list)
            spatial_dict[pix] = evt_len_mean
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)

        outf = join(outdir,f'drought_events_duration.tif')
        DIC_and_TIF().arr_to_tif(arr, outf)

    def png_drought_events_duration(self):
        fdir = join(self.this_class_tif,'drought_events_duration')
        outdir = join(self.this_class_png,'drought_events_duration')
        fpath = join(fdir,'drought_events_duration.tif')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        plt.figure(figsize=(5, 5))
        ax = plt.subplot(1, 1, 1)
        Plot().plot_ortho(fpath,vmin=1,vmax=3,ax=ax)
        plt.tight_layout()
        outf = join(outdir,'drought_events_duration.png')
        plt.savefig(outf,dpi=600)
        plt.close()

        plt.figure()
        dff = join(self.this_class_arr,'drought_events_detail/drought_events_detail.df')
        df = T.load_df(dff)
        evt_list = df['evt'].tolist()
        evt_len_list = []
        for evt in evt_list:
            evt_len = len(evt)
            evt_len_list.append(evt_len)
        plt.hist(evt_len_list,bins=6,range=(1,7),rwidth=0.8,align='left',density=True)
        plt.xlabel('Drought duration (month)')
        plt.ylabel('Frequency')
        outf = join(outdir,'drought_events_duration_hist.pdf')
        plt.savefig(outf)
        plt.close()


    def __add_drought_year_to_df(self,df):
        drought_year_list = []
        for i,row in df.iterrows():
            evt = row['evt']
            drought_start_month_index = evt[0]
            drought_start_year = drought_start_month_index // 12 + global_start_year
            drought_year_list.append(drought_start_year)
        df['drought_year'] = drought_year_list

        return df

    def drought_affected_area_time_series(self):
        land_total_area_in_km2 = 50542483.19037567 # vegetated land, above 30 N
        dff = join(self.this_class_arr, 'drought_events_detail/drought_events_detail.df')
        df = T.load_df(dff)
        df = self.__add_drought_year_to_df(df)
        pixel_area = DIC_and_TIF().calculate_pixel_area()
        df = T.add_spatial_dic_to_df(df,pixel_area,'area_m2')
        df['area_km2'] = df['area_m2'] / 1000 / 1000
        outdir = join(self.this_class_png, 'drought_affected_area_time_series')
        T.mk_dir(outdir, force=True)
        # T.open_path_and_file(outdir)

        year_list = list(range(global_start_year,global_end_year+1))
        plt.figure(figsize=(10, 5))
        flag = 1
        df_drt_group_dict = T.df_groupby(df,'drought_year')
        area_ratio_list = []
        for year in year_list:
            if not year in df_drt_group_dict:
                area_ratio_list.append(np.nan)
                continue
            df_i = df_drt_group_dict[year]
            area_sum = df_i['area_km2'].sum()
            ratio = area_sum / land_total_area_in_km2 * 100
            area_ratio_list.append(ratio)
        a,b,r,p = T.nan_line_fit(year_list,area_ratio_list)
        function = 'y = %.2f x + %.2f'%(a,b) + f', r = {r:.2f}, p = {p:.2f}'
        plt.bar(year_list,area_ratio_list)
        plt.xlabel('year')
        plt.ylabel('area ratio (%)')
        plt.text(1999, 0.5*flag, function)
        flag += 2
        outf = join(outdir,'drought_affected_area_time_series.pdf')
        plt.show()
        # plt.savefig(outf)
        # plt.close()

class SPEI_ts:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('SPEI_ts', result_root_this_script, mode=2)
        pass

    def run(self):

        # self.cal_affected_area()
        # self.gen_spei_df()
        # self.cal_annual_spei()
        # self.plot_area_and_annual_spei()

        pass

    def cal_affected_area(self):
        import preprocess
        outdir = join(self.this_class_arr,'spei_affected_area')
        T.mk_dir(outdir)

        outf = join(outdir,'spei_affected_area.df')
        land_total_area_in_km2 = 50542483.19037567  # vegetated land, above 30 N
        pixel_area = DIC_and_TIF().calculate_pixel_area()
        spei_tif_dir = join(preprocess.SPEI().datadir,'tif/spei03')
        date_list = []
        y = []
        for f in tqdm(T.listdir(spei_tif_dir)):
            date_str = f.split('.')[0]
            year = int(date_str[:4])
            month = int(date_str[4:6])
            date = datetime.datetime(year,month,1)
            date_list.append(date)
            fpath = join(spei_tif_dir,f)
            spatial_dic = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dic_affected = {}
            area_sum = 0
            for pix in spatial_dic:
                r,c = pix
                if r > 120:
                    continue
                val = spatial_dic[pix]
                if val > 10:
                    continue
                if val > -1.5:
                    continue
                area = pixel_area[pix]
                area = area / 1000 / 1000
                area_sum += area
            percentage = area_sum / land_total_area_in_km2 * 100
            y.append(percentage)
        df = pd.DataFrame()
        df['date'] = date_list
        df['percentage'] = y
        T.save_df(df,outf)
        T.df_to_excel(df,outf)
        T.open_path_and_file(outdir)

    def gen_spei_df(self):
        import preprocess
        from Chapter5 import statistic
        outdir = join(self.this_class_arr, 'spei_df')
        T.mk_dir(outdir)
        outf = join(outdir, 'spei_df.df')
        spei_tif_dir = join(preprocess.SPEI().datadir, 'per_pix/1982-2015/spei03')
        spei_spatial_dict = T.load_npy_dir(spei_tif_dir)
        vals_list = []
        pix_list = []
        for pix in tqdm(spei_spatial_dict):
            vals = spei_spatial_dict[pix]
            vals = np.array(vals,dtype=float)
            vals[vals>10] = np.nan
            if T.is_all_nan(vals):
                continue
            pix_list.append(pix)
            vals_list.append(vals)
        df = pd.DataFrame()
        df['pix'] = pix_list
        df['spei'] = vals_list
        df = statistic.Dataframe_func(df).df
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

    def cal_annual_spei(self):
        dff = join(self.this_class_arr, 'spei_df/spei_df.df')
        df = T.load_df(dff)
        T.print_head_n(df,10)
        gs = global_gs
        spei_list = df['spei'].tolist()
        annual_spei_list = []
        for spei in tqdm(spei_list):
            annual_spei = T.monthly_vals_to_annual_val(spei,gs)
            annual_spei_list.append(annual_spei)
        df['annual_spei'] = annual_spei_list
        T.print_head_n(df,10)
        T.save_df(df,dff)
        T.df_to_excel(df,dff)


    def plot_area_and_annual_spei(self):
        outdir = join(self.this_class_png,'drought_area_and_annual_spei')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        outf = join(outdir,'drought_area_and_annual_spei.pdf')
        dff_ts = join(self.this_class_arr, 'spei_df/spei_df.df')
        dff_area = join(self.this_class_arr, 'spei_affected_area/spei_affected_area.df')
        df_ts = T.load_df(dff_ts)
        df_area = T.load_df(dff_area)
        # T.print_head_n(df_ts,10)
        annual_spei_list = df_ts['annual_spei'].tolist()
        monthly_date_list = df_area['date'].tolist()
        area_list = df_area['percentage'].tolist()
        annual_spei_mean = np.nanmean(annual_spei_list,axis=0)
        annual_date_list = []
        for year in global_year_range_list:
            date = datetime.datetime(year,6,1)
            annual_date_list.append(date)
        plt.figure(figsize=(18*centimeter_factor,6*centimeter_factor))
        # plt.plot(monthly_date_list,area_list,zorder=-99)
        plt.fill_between(monthly_date_list, area_list, color='r', alpha=0.7, zorder=-99, edgecolor='none')
        a,b,r,p = T.nan_line_fit(list(range(len(area_list))),area_list)
        print(a,b,r,p)
        plt.plot(monthly_date_list, a * np.array(list(range(len(area_list)))) + b,
                    color='k', linewidth=1, zorder=10, linestyle='--')
        plt.ylim(-30, 30)
        plt.ylabel('Area Affected by Drought (%)')
        plt.twinx()
        plt.bar(annual_date_list, annual_spei_mean, width=300, color='none', edgecolor='k', linewidth=1, zorder=10)
        a,b,r,p = T.nan_line_fit(list(range(len(annual_spei_mean))),annual_spei_mean)
        print(a,b,r,p)
        plt.plot(annual_date_list, a * np.array(list(range(len(annual_spei_mean)))) + b,
                 color='k', linewidth=1, zorder=10, linestyle='--')
        annual_spei_mean_smooth = SMOOTH().smooth_convolve(annual_spei_mean, window_len=9)
        plt.plot(annual_date_list, annual_spei_mean_smooth, color='k', linewidth=1, zorder=10)
        plt.ylim(-.25, .25)
        plt.xticks(annual_date_list[3::5], [str(i) for i in global_year_range_list[3::5]])
        plt.ylabel('SPEI 3')
        # plt.savefig(outf, dpi=300)
        # plt.close()
        plt.show()

class Correlation_matrix:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Correlation_matrix', result_root_this_script, mode=2)
        pass

    def run(self):
        self.foo()
        pass

    def foo(self):
        from Chapter4 import analysis
        dff = join(analysis.Dataframe().this_class_arr,'Dataframe.df')
        df = T.load_df(dff)
        # T.print_head_n(df,10)
        # vars_list = self.__Rsgs()
        vars_list = self.__Rmgs()
        df = df[df['product']=='spei03']
        df = df[vars_list]
        data = df.corr()
        fig, ax = plt.subplots(1, 1)
        m = self.plot_corr_ellipses(data, ax=ax, cmap='RdBu_r', clim=[-.3, .3])
        cb = fig.colorbar(m)
        cb.set_label('Correlation coefficient')
        ax.margins(0.1)
        plt.show()


    def plot_corr_ellipses(self, data, ax=None, **kwargs):
        from matplotlib.collections import EllipseCollection
        M = np.array(data)
        if not M.ndim == 2:
            raise ValueError('data must be a 2D array')
        if ax is None:
            fig, ax = plt.subplots(1, 1, subplot_kw={'aspect': 'equal'})
            ax.set_xlim(-0.5, M.shape[1] - 0.5)
            ax.set_ylim(-0.5, M.shape[0] - 0.5)

        # xy locations of each ellipse center
        xy = np.indices(M.shape)[::-1].reshape(2, -1).T

        # set the relative sizes of the major/minor axes according to the strength of
        # the positive/negative correlation
        w = np.ones_like(M).ravel()
        h = 1 - np.abs(M).ravel()
        a = 45 * np.sign(M).ravel()

        ec = EllipseCollection(widths=w, heights=h, angles=a, units='x', offsets=xy,
                               transOffset=ax.transData, array=M.ravel(), **kwargs)
        ax.add_collection(ec)

        # if data is a DataFrame, use the row/column names as tick labels
        if isinstance(data, pd.DataFrame):
            ax.set_xticks(np.arange(M.shape[1]))
            ax.set_xticklabels(data.columns, rotation=90)
            ax.set_yticks(np.arange(M.shape[0]))
            ax.set_yticklabels(data.index)

        return ec

    def __Rsgs(self):
        i = 6
        variables = \
            '''
            post_PRE
            post_VPD
            NDVI_pre_{}_mean
            MA_TMP
            current_sos_anomaly
            VPD_lag_mean
            PRE_pre_{}_mean
            VPD_pre_{}_mean
            PRE_lag_mean
            water_balance
            MA_PRE
            CV_PRE
            CV_TMP
            correlation
            sand
            '''.format(i, i, i)
        vars_list = variables.split()
        return vars_list

    def __Rmgs(self):
        i = 6
        variables = \
            '''
            current_sos_std_anomaly
            post_PRE
            post_VPD
            dormant_length
            dormant_SWE_Terra
            dormant_TMP
            NDVI_pre_{}_mean
            MA_TMP
            lag
            PRE_pre_{}_mean
            drought_year_sos_anomaly
            water_balance
            MA_PRE
            CV_PRE
            CV_TMP
            VPD_pre_{}_mean
            PRE_lag_mean
            VPD_lag_mean
            correlation
            sand
            '''.format(i, i, i)
        vars_list = variables.split()
        return vars_list


def main():
    # Drought_events().run()
    # SPEI_ts().run()
    Pairplot().run()
    pass

if __name__ == '__main__':
    main()

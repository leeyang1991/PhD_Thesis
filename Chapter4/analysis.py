# coding=utf-8

from meta_info import *
result_root_this_script = join(results_root, 'Chapter4/analysis')
land_tif = global_land_tif

class Global_vars:

    def __init__(self):

        self.dff = Dataframe().dff

        pass

    def load_df(self):
        dff = Global_vars().dff
        df = T.load_df(dff)
        df = df[df['lat'] > 30]
        df = df[df['CV_PRE'] < 100]
        df = df[df['current_sos_std_anomaly'] < 10]
        df = df[df['current_sos_std_anomaly'] > -10]
        # df = df[df['current_sos_anomaly'] < 10]
        # df = df[df['current_sos_anomaly'] > -10]
        # df = df.dropna(subset=['timing'])
        return df

    def load_df_all(self):
        dff = Global_vars().dff
        df = T.load_df(dff)
        return df

    def clean_df(self, df, winter):
        if winter == 'with_winter':
            x_list = Global_vars().vars_RF_with_winter()
            df = df[df['dormant_SWE_Terra'] > -2]
            df = df[df['dormant_SWE_Terra'] < 2]
            df = df[df['current_sos_std_anomaly'] > -2]
            df = df[df['current_sos_std_anomaly'] < 2]
        elif winter == 'no_winter':
            x_list = Global_vars().vars_RF_no_winter()
        else:
            raise UserWarning

        for i in x_list:
            try:
                df = df[df[i] < 99999]
            except:
                pass

        df = df[df['post_PRE'] > -2]
        df = df[df['post_PRE'] < 2]
        #
        df = df[df['post_TMP'] > -2]
        df = df[df['post_TMP'] < 2]
        #
        df = df[df['post_VPD'] > -2]
        df = df[df['post_VPD'] < 2]
        #

        #
        df = df[df['NDVI_pre_6_mean'] > -2]
        df = df[df['NDVI_pre_6_mean'] < 2]

        return df

    def lc_list(self):
        lc_list = [
            'evergreen',
            'deciduous',
            'Shrublands',
            'Grasslands',
        ]
        return lc_list

    def timing_list(self):

        timing_list = ['early', 'mid', 'late', ]

        return timing_list
        pass

    def koppen_landuse(self):
        kl_list = [u'Forest.A', u'Forest.B', u'Forest.Cf', u'Forest.Csw', u'Forest.Df', u'Forest.Dsw', u'Forest.E',
                   u'Grasslands.A', u'Grasslands.B', u'Grasslands.Cf', u'Grasslands.Csw', u'Grasslands.Df',
                   u'Grasslands.Dsw',
                   u'Grasslands.E', u'Shrublands.A', u'Shrublands.B', u'Shrublands.Cf', u'Shrublands.Csw',
                   u'Shrublands.Df',
                   u'Shrublands.Dsw', u'Shrublands.E']
        return kl_list

    def koppen_list(self):
        koppen_list = [u'A', u'B', u'Cf', u'Csw', u'Df', u'Dsw', u'E', ]
        return koppen_list
        pass

    def lc_colors(self, lc):
        lc_colors_dic = {
            'Shrublands': "b",
            'Forest': "c",
            'Grasslands': "r",
            'evergreen': "g",
            'deciduous': "m",
        }
        return lc_colors_dic[lc]

    def vars_RF_with_winter(self):
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
            row
            col
            '''.format(i, i, i)
        vars_list = variables.split()
        return vars_list

    def vars_RF_no_winter(self):
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
            row
            col
            '''.format(i, i, i)
        vars_list = variables.split()
        return vars_list

    def marker_dic(self):
        markers_dic = {
            'Shrublands': "s",  # square
            'evergreen': "D",  # diamond
            'deciduous': "d",  # thin diamond
            'Grasslands': "X",  # x filled
        }
        return markers_dic


class Main_flow_Early_Peak_Late_Dormant:

    def __init__(self):
        self.this_class_arr = result_root_this_script + 'arr/Main_flow_Early_Peak_Late_Dormant/'
        self.this_class_tif = result_root_this_script + 'tif/Main_flow_Early_Peak_Late_Dormant/'
        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        pass

    def run(self):
        self.annual_phelogy()
        # self.long_term_pheology()
        pass

    def annual_phelogy(self):
        # calculate annual phenology
        # 1
        # self.tif_bi_weekly_annual()
        # 2 data transform
        self.data_transform_annual()
        # 3 smooth bi-weekly to daily
        self.hants_smooth_annual()
        # 4 calculate phenology
        self.early_peak_late_dormant_period_annual()
        # 5 transform daily to monthly
        self.transform_early_peak_late_dormant_period_annual()
        # 99 check get_early_peak_late_dormant_period_long_term
        # self.check_get_early_peak_late_dormant_period_long_term()
        pass

    def long_term_pheology(self):
        # calculate long term phenology
        # 1 calculate long term NDVI mean ,24 images
        # self.tif_bi_weekly_mean_long_term()
        # 2 transform to per pixel
        # self.data_transform()
        # 3 smooth bi-weekly to daily
        # self.hants_smooth()
        # 4 calculate phenology
        self.early_peak_late_dormant_period_long_term()
        # 99 check phenology
        self.check_early_peak_late_dormant_period_long_term()

        pass

    def return_phenology(self):
        f = self.this_class_arr + 'early_peak_late_dormant_period_long_term/early_peak_late_dormant_period_long_term.npy'
        dic = T.load_npy(f)
        return dic

    def return_gs(self):
        f = self.this_class_arr + 'early_peak_late_dormant_period_long_term/early_peak_late_dormant_period_long_term.npy'
        dic = T.load_npy(f)
        gs_dic = {}
        for pix in dic:
            val = dic[pix]['GS_mon']
            gs_dic[pix] = val
        return gs_dic
        pass

    def check_early_peak_late_dormant_period_long_term(self):
        outtifdir = self.this_class_tif + 'early_peak_late_dormant_period_long_term/'
        T.mk_dir(outtifdir)
        f = self.this_class_arr + 'early_peak_late_dormant_period_long_term/early_peak_late_dormant_period_long_term.npy'
        dic = T.load_npy(f)

        # result = {
        #     'early_length': 'early_period',
        #     'mid_length': 'peak_period',
        #     'late_length': 'late_period',
        #     'dormant_length': 'dormant_period',
        #     'early_start': '.',
        #     'early_end': '.',
        #     'peak': '.',
        #     'late_start': '.',
        #     'late_end': '.',
        # }
        # for var in result:
        #     spatial_dic = {}
        #     for pix in dic:
        #         val = dic[pix][var]
        #         spatial_dic[pix] = val
        #     print var
        #     arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        #     DIC_and_TIF().arr_to_tif(arr,outtifdir+var+'.tif')
        #######################################################
        #######################################################
        #######################################################
        spatial_1 = {}
        spatial_2 = {}
        spatial_3 = {}
        for pix in dic:
            GS_start_mon = dic[pix]['GS_mon'][0]
            GS_end_mon = dic[pix]['GS_mon'][-1]
            GS_length_mon = len(dic[pix]['GS_mon'])
            spatial_1[pix] = GS_start_mon
            spatial_2[pix] = GS_end_mon
            spatial_3[pix] = GS_length_mon

        arr1 = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_1)
        arr2 = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_2)
        arr3 = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_3)

        DIC_and_TIF().arr_to_tif(arr1, outtifdir + 'GS_start_mon.tif')
        DIC_and_TIF().arr_to_tif(arr2, outtifdir + 'GS_end_mon.tif')
        DIC_and_TIF().arr_to_tif(arr3, outtifdir + 'GS_length_mon.tif')

    def check_get_early_peak_late_dormant_period_long_term(self):
        fdir = self.this_class_arr + 'transform_early_peak_late_dormant_period_annual/'
        for var in T.listdir(fdir):
            print(var)
            dic = T.load_npy(fdir + var)
            spatial_dic = {}
            for pix in tqdm(dic):
                val = dic[pix]
                if len(val) == 0:
                    continue
                # plt.plot(val)
                # plt.show()
                meanarr = np.nanmean(val)
                spatial_dic[pix] = meanarr
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            plt.imshow(arr)
            plt.colorbar()
            plt.show()

        pass

    def __day_to_month(self, doy):
        base = datetime.datetime(2000, 0o1, 0o1)
        time_delta = datetime.timedelta(int(doy))
        date = base + time_delta
        month = date.month
        day = date.day

        if day > 15:
            month = month + 1
        if month >= 12:
            month = 12
        return month

    def transform_early_peak_late_dormant_period_annual(self):
        vars_dic = {
            'early_length': 'early_period',
            'mid_length': 'peak_period',
            'late_length': 'late_period',
            'dormant_length': 'dormant_period',
            'early_start': 'early_start',
            'early_start_mon': 'self.__day_to_month(early_start)',

            'early_end': 'early_end',
            'early_end_mon': 'self.__day_to_month(early_end)',

            'peak': 'peak',
            'peak_mon': 'self.__day_to_month(peak)',

            'late_start': 'late_start',
            'late_start_mon': 'self.__day_to_month(late_start)',

            'late_end': 'late_end',
            'late_end_mon': 'self.__day_to_month(late_end)',
        }

        outdir = self.this_class_arr + 'transform_early_peak_late_dormant_period_annual/'
        T.mk_dir(outdir)
        fdir = self.this_class_arr + 'early_peak_late_dormant_period_annual/'
        #
        for var in vars_dic:
            spatial_dic = DIC_and_TIF().void_spatial_dic()
            for y in tqdm(list(range(1982, 2016)), desc=var):
                f = fdir + '{}.npy'.format(y)
                dic = T.load_npy(f)
                for pix in dic:
                    var_val = dic[pix][var]
                    spatial_dic[pix].append(var_val)
            np.save(outdir + var, spatial_dic)

        ############### Dormant Mons #############
        spatial_dic = DIC_and_TIF().void_spatial_dic()
        for y in tqdm(list(range(1982, 2016)), desc='dormant_mons_list'):
            f = fdir + '{}.npy'.format(y)
            dic = T.load_npy(f)
            for pix in dic:
                sos = dic[pix]['early_start_mon']
                eos = dic[pix]['late_end_mon']
                gs = list(range(sos, eos + 1))
                winter_mons = []
                for m in range(1, 13):
                    if m in gs:
                        continue
                    winter_mons.append(m)
                spatial_dic[pix].append(winter_mons)
        np.save(outdir + 'dormant_mons', spatial_dic)

        ############### GS Mons #############
        spatial_dic = DIC_and_TIF().void_spatial_dic()
        for y in tqdm(list(range(1982, 2016)), desc='dormant_mons_list'):
            f = fdir + '{}.npy'.format(y)
            dic = T.load_npy(f)
            for pix in dic:
                sos = dic[pix]['early_start_mon']
                eos = dic[pix]['late_end_mon']
                if sos >= eos:
                    continue
                gs_mons = list(range(sos, eos + 1))
                spatial_dic[pix].append(gs_mons)
        np.save(outdir + 'gs_mons', spatial_dic)

        pass

    def hants_smooth_annual(self):
        outdir = self.this_class_arr + 'hants_smooth_annual/'
        T.mk_dir(outdir)
        gs_f = Phenology_based_on_Temperature_NDVI().this_class_arr + 'growing_season_index.npy'
        gs_dic = T.load_npy(gs_f)
        per_pix_dir = self.this_class_arr + 'data_transform_annual/'
        for year in T.listdir(per_pix_dir):
            print(year)
            per_pix_dir_i = per_pix_dir + year + '/'
            outf = outdir + year
            ndvi_dic = {}
            for f in T.listdir(per_pix_dir_i):
                dic = T.load_npy(per_pix_dir_i + f)
                ndvi_dic.update(dic)
            hants_dic = {}
            for pix in tqdm(ndvi_dic):
                if not pix in gs_dic:
                    continue
                vals = ndvi_dic[pix]
                smoothed_vals = self.__kernel_hants(vals)
                hants_dic[pix] = np.array(smoothed_vals)
            np.save(outf, hants_dic)

    def data_transform_annual(self):

        fdir = self.this_class_tif + 'tif_bi_weekly_annual/'
        outdir = self.this_class_arr + 'data_transform_annual/'
        T.mk_dir(outdir)
        for year in T.listdir(fdir):
            print(year, '\n')
            # for f in T.listdir(fdir + year):
            #     print f
            outdir_i = outdir + year + '/'
            T.mk_dir(outdir_i)
            Pre_Process().data_transform(fdir + year + '/', outdir_i)
        pass

    def early_peak_late_dormant_period_annual(self, threshold_i=0.2):
        hants_smooth_dir = self.this_class_arr + 'hants_smooth_annual/'
        outdir = self.this_class_arr + 'early_peak_late_dormant_period_annual/'
        T.mk_dir(outdir)

        for f in T.listdir(hants_smooth_dir):
            outf_i = outdir + f
            year = int(f.split('.')[0])
            hants_smooth_f = hants_smooth_dir + f
            hants_dic = T.load_npy(hants_smooth_f)
            result_dic = {}
            for pix in tqdm(hants_dic, desc=str(year)):
                vals = hants_dic[pix]
                peak = np.argmax(vals)
                if peak == 0 or peak == (len(vals) - 1):
                    continue
                try:
                    early_start = self.__search_left(vals, peak, threshold_i)
                    late_end = self.__search_right(vals, peak, threshold_i)
                except:
                    early_start = 60
                    late_end = 130
                    print(vals)
                    plt.plot(vals)
                    plt.show()
                # method 1
                # early_end, late_start = self.__slope_early_late(vals,early_start,late_end,peak)
                # method 2
                early_end, late_start = self.__median_early_late(vals, early_start, late_end, peak)

                early_period = early_end - early_start
                peak_period = late_start - early_end
                late_period = late_end - late_start
                dormant_period = 365 - (late_end - early_start)

                result = {
                    'early_length': early_period,
                    'mid_length': peak_period,
                    'late_length': late_period,
                    'dormant_length': dormant_period,
                    'early_start': early_start,
                    'early_start_mon': self.__day_to_month(early_start),

                    'early_end': early_end,
                    'early_end_mon': self.__day_to_month(early_end),

                    'peak': peak,
                    'peak_mon': self.__day_to_month(peak),

                    'late_start': late_start,
                    'late_start_mon': self.__day_to_month(late_start),

                    'late_end': late_end,
                    'late_end_mon': self.__day_to_month(late_end),
                }
                result_dic[pix] = result
            np.save(outf_i, result_dic)

    def early_peak_late_dormant_period_long_term(self, threshold_i=0.2):
        hants_smooth_f = self.this_class_arr + 'hants_smooth/hants_smooth.npy'
        outdir = self.this_class_arr + 'early_peak_late_dormant_period_long_term/'
        T.mk_dir(outdir)
        outf = outdir + 'early_peak_late_dormant_period_long_term'
        hants_dic = T.load_npy(hants_smooth_f)
        result_dic = {}
        for pix in tqdm(hants_dic):
            vals = hants_dic[pix]
            peak = np.argmax(vals)
            try:
                # start = self.__search_left(vals, maxind, threshold_i)
                # end = self.__search_right(vals, maxind, threshold_i)
                # dormant_length = 365 - (end - start)
                # spatial_dic[pix] = dormant_length/30.

                early_start = self.__search_left(vals, peak, threshold_i)
                late_end = self.__search_right(vals, peak, threshold_i)
                # method 1
                # early_end, late_start = self.__slope_early_late(vals,early_start,late_end,peak)
                # method 2
                early_end, late_start = self.__median_early_late(vals, early_start, late_end, peak)

                early_period = early_end - early_start
                peak_period = late_start - early_end
                late_period = late_end - late_start
                dormant_period = 365 - (late_end - early_start)
                GS_mon = list(range(self.__day_to_month(early_start), self.__day_to_month(late_end) + 1))
                result = {
                    'early_length': early_period,
                    'mid_length': peak_period,
                    'late_length': late_period,
                    'dormant_length': dormant_period,
                    'early_start': self.__day_to_month(early_start),
                    'early_end': self.__day_to_month(early_end),
                    'peak': self.__day_to_month(peak),
                    'late_start': self.__day_to_month(late_start),
                    'late_end': self.__day_to_month(late_end),
                    'GS_mon': np.array(GS_mon),
                }
                result_dic[pix] = result
            except:
                pass
        np.save(outf, result_dic)

    def hants_smooth(self):
        outdir = self.this_class_arr + 'hants_smooth/'
        outf = outdir + 'hants_smooth'
        T.mk_dir(outdir)
        gs_f = Phenology_based_on_Temperature_NDVI().this_class_arr + 'growing_season_index.npy'
        gs_dic = T.load_npy(gs_f)
        per_pix_dir = self.this_class_arr + 'NDVI_bi_weekly_per_pix/'
        ndvi_dic = {}
        for f in T.listdir(per_pix_dir):
            dic = T.load_npy(per_pix_dir + f)
            ndvi_dic.update(dic)

        hants_dic = {}
        for pix in tqdm(ndvi_dic):
            if not pix in gs_dic:
                continue
            vals = ndvi_dic[pix]
            smoothed_vals = self.__kernel_hants(vals)
            hants_dic[pix] = np.array(smoothed_vals)
        np.save(outf, hants_dic)

    def tif_bi_weekly_annual(self):
        fdir = data_root + 'NDVI/tif_05deg_bi_weekly/'
        outdir = self.this_class_tif + 'tif_bi_weekly_annual/'
        T.mk_dir(outdir)
        date_list = []
        for y in range(1982, 2016):
            temp = []
            for m in range(1, 13):
                for d in [0o1, 15]:
                    temp.append('{}{:02d}{:02d}.tif'.format(y, m, d))
            date_list.append(temp)
        for annual in date_list:
            yyyy = annual[0].split('.')[0][:4]
            print(yyyy)
            outdir_i = outdir + yyyy + '/'
            T.mk_dir(outdir_i)
            for mon in annual:
                f = fdir + mon
                shutil.copy(f, outdir_i + mon)

    def tif_bi_weekly_mean_long_term(self):
        fdir = data_root + 'NDVI/tif_05deg_bi_weekly/'
        outdir = self.this_class_tif + 'tif_bi_weekly_mean/'
        T.mk_dir(outdir)
        date_list = []
        for m in range(1, 13):
            for d in [0o1, 15]:
                temp = []
                for y in range(1982, 2016):
                    temp.append('{}{:02d}{:02d}.tif'.format(y, m, d))
                date_list.append(temp)
        template_f = DIC_and_TIF().tif_template
        template_arr = to_raster.raster2array(template_f)[0]
        for mon in date_list:
            fname = mon[0].split('.')[0][-4:] + '.tif'
            print(fname)
            arrs = np.zeros_like(template_arr)
            for y in mon:
                f = fdir + y
                arr = to_raster.raster2array(f)[0]
                arrs += arr
            mon_mean_arr = arrs / len(mon)
            T.mask_999999_arr(mon_mean_arr)
            DIC_and_TIF().arr_to_tif(mon_mean_arr, outdir + fname)

    def data_transform(self):
        fdir = self.this_class_tif + 'tif_bi_weekly_mean/'
        outdir = self.this_class_arr + 'NDVI_bi_weekly_per_pix/'
        Pre_Process().data_transform(fdir, outdir)

    def __interp__(self, vals):

        # x_new = np.arange(min(inx), max(inx), ((max(inx) - min(inx)) / float(len(inx))) / float(zoom))

        inx = list(range(len(vals)))
        iny = vals
        x_new = np.linspace(min(inx), max(inx), 365)
        func = interpolate.interp1d(inx, iny)
        y_new = func(x_new)

        return x_new, y_new

    def __search_left(self, vals, maxind, threshold_i):
        left_vals = vals[:maxind]
        left_min = np.min(left_vals)
        max_v = vals[maxind]
        if left_min < 2000:
            left_min = 2000
        threshold = (max_v - left_min) * threshold_i + left_min

        ind = 999999
        for step in range(365):
            ind = maxind - step
            if ind >= 365:
                break
            val_s = vals[ind]
            if val_s <= threshold:
                break

        return ind

    def __search_right(self, vals, maxind, threshold_i):
        right_vals = vals[maxind:]
        right_min = np.min(right_vals)
        max_v = vals[maxind]
        if right_min < 2000:
            right_min = 2000
        threshold = (max_v - right_min) * threshold_i + right_min

        ind = 999999
        for step in range(365):
            ind = maxind + step
            if ind >= 365:
                break
            val_s = vals[ind]
            if val_s <= threshold:
                break

        return ind

    def __kernel_hants(self, vals_bi_week):
        vals = np.array(vals_bi_week)
        std = np.std(vals)
        if std == 0:
            return None
        xnew, ynew = self.__interp__(vals)
        ynew = np.array([ynew])
        results = HANTS(sample_count=365, inputs=ynew, low=-10000, high=10000,
                        fit_error_tolerance=std)
        result = results[0]

        # plt.plot(result)
        # plt.plot(range(len(ynew[0])),ynew[0])
        # plt.show()
        return result

    def __slope_early_late(self, vals, sos, eos, peak):
        # 1 slope最大和最小分别作为early late 的结束和开始
        # 问题：early late 时间太短
        slope_left = []
        for i in range(sos, peak):
            if i - 1 < 0:
                slope_i = vals[1] - vals[0]
            else:
                slope_i = vals[i] - vals[i - 1]
            slope_left.append(slope_i)

        slope_right = []
        for i in range(peak, eos):
            if i - 1 < 0:
                slope_i = vals[1] - vals[0]
            else:
                slope_i = vals[i] - vals[i - 1]
            slope_right.append(slope_i)

        max_ind = np.argmax(slope_left) + sos
        min_ind = np.argmin(slope_right) + peak

        return max_ind, min_ind

    def __median_early_late(self, vals, sos, eos, peak):
        # 2 使用sos-peak peak-eos中位数作为sos和eos的结束和开始

        median_left = int((peak - sos) / 2.)
        median_right = int((eos - peak) / 2)
        max_ind = median_left + sos
        min_ind = median_right + peak
        return max_ind, min_ind


class Main_flow_Pick:
    '''
    分月份
    '''

    def __init__(self):
        self.this_class_arr = result_root_this_script + 'arr/Main_flow_Pick/'
        self.this_class_tif = result_root_this_script + 'tif/Main_flow_Pick/'
        self.this_class_png = result_root_this_script + 'png/Main_flow_Pick/'
        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        pass

    def run(self):
        # # ***********************0 prepare spei pick******************************
        # self.compose_spei_n_to_one_file()
        # # ***********************1 pick******************************
        # #############################################################
        # f_path = data_root + 'CWD/per_pix_1982_2015_detrend/CWD.npy'
        # outdir = self.this_class_arr + 'CWD_events/'
        # # # ------------------------CWD--------------------------------
        # self.pick(f_path, outdir)
        # # # ----------------------SPEI 1-12----------------------------
        # spei_compose_dir = data_root + 'SPEI/compose_spei_n_to_one_file/'
        # outdir = self.this_class_arr + 'compose_spei_n_to_one_file/'
        # T.mk_dir(outdir)
        # for f in T.listdir(spei_compose_dir):
        #     drought_index_f = spei_compose_dir+f
        #     outdir_i = outdir + f.split('.')[0] + '/'
        #     self.pick(drought_index_f,outdir_i)
        # #############################################################
        # # **********************2 add timing*************************
        # #############################################################
        # # ------------------------CWD--------------------------------
        events_f = self.this_class_arr + 'CWD_events/single_events.npy'
        drought_index_f = data_root + 'CWD/per_pix_1982_2015_detrend/CWD.npy'
        outdir = self.this_class_arr + 'events_with_timing_cwd/'
        growing_season_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + \
                           'early_peak_late_dormant_period_long_term/early_peak_late_dormant_period_long_term.npy'
        self.events_with_timing(events_f, drought_index_f, outdir, growing_season_f)
        # # # ----------------------SPEI 1-12----------------------------
        event_dir = self.this_class_arr + 'compose_spei_n_to_one_file/'
        for folder in tqdm(T.listdir(event_dir)):
            events_f = event_dir + folder + '/single_events.npy'
            spei_f = data_root + 'SPEI/compose_spei_n_to_one_file/{}.npy'.format(folder)
            outdir = self.this_class_arr + 'events_with_timing_spei/{}/'.format(folder)
            growing_season_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + \
                               'early_peak_late_dormant_period_long_term/early_peak_late_dormant_period_long_term.npy'
            self.events_with_timing(events_f, spei_f, outdir, growing_season_f)

        # ----------------------compose----------------------------
        self.compose_cwd_spei_events()
        pass

    def compose_cwd_spei_events(self):
        outdir = self.this_class_arr + 'compose_cwd_spei_events/'
        outf = outdir + 'events'
        T.mk_dir(outdir)
        cwd_f = self.this_class_arr + 'events_with_timing_cwd/events_with_timing.npy'
        spei_dir = self.this_class_arr + 'events_with_timing_spei/'
        cwd_dic = T.load_npy(cwd_f)
        events_dic_all = DIC_and_TIF().void_spatial_dic()
        for pix in cwd_dic:
            events = cwd_dic[pix]
            for event in events:
                events_dic_all[pix].append(event)

        for n in T.listdir(spei_dir):
            print(n)
            f = spei_dir + n + '/events_with_timing.npy'
            dic = T.load_npy(f)
            for pix in dic:
                events = dic[pix]
                for event in events:
                    events_dic_all[pix].append(event)
        np.save(outf, events_dic_all)

    def compose_spei_n_to_one_file(self):
        fdir = data_root + 'SPEI/per_pix_clean_smooth_detrend/'
        outdir = data_root + 'SPEI/compose_spei_n_to_one_file/'
        T.mk_dir(outdir)
        for spei in tqdm(T.listdir(fdir)):
            dic = {}
            for f in T.listdir(fdir + spei):
                dic_i = T.load_npy(os.path.join(fdir, spei, f))
                dic.update(dic_i)
            np.save(outdir + spei, dic)
        pass

    def check_EGS_LGS_events(self):
        event_dic_f = self.this_class_arr + 'EGS_LGS_events/EGS_LGS_events.npy'
        event_dic = T.load_npy(event_dic_f)

        pre_dic = {}
        early_dic = {}
        late_dic = {}
        for pix in event_dic:
            pre_num = 0
            early_num = 0
            late_num = 0
            for eln, drought_range in event_dic[pix]:
                if eln == 'pre':
                    pre_num += 1
                elif eln == 'early':
                    early_num += 1
                elif eln == 'late':
                    late_num += 1
                else:
                    pass
            if pre_num > 0:
                pre_dic[pix] = pre_num
            if early_num > 0:
                early_dic[pix] = early_num
            if late_num > 0:
                late_dic[pix] = late_num
        pre_arr = DIC_and_TIF().pix_dic_to_spatial_arr(pre_dic)
        early_arr = DIC_and_TIF().pix_dic_to_spatial_arr(early_dic)
        late_arr = DIC_and_TIF().pix_dic_to_spatial_arr(late_dic)

        plt.figure()
        DIC_and_TIF().plot_back_ground_arr()
        plt.imshow(pre_arr)
        plt.colorbar()
        plt.title('pre_arr')

        plt.figure()
        DIC_and_TIF().plot_back_ground_arr()
        plt.imshow(early_arr)
        plt.title('early_arr')
        plt.colorbar()

        plt.figure()
        DIC_and_TIF().plot_back_ground_arr()
        plt.imshow(late_arr)
        plt.title('late_arr')
        plt.colorbar()

        plt.show()

        pass

    def events_with_timing(self, events_f, drought_index_f, outdir, growing_season_f):
        # outdir = self.this_class_arr + 'EGS_LGS_events/'
        outf = outdir + 'events_with_timing'
        T.mk_dir(outdir, force=True)
        # growing_season_f = Winter().this_class_arr + 'gen_grow_season_index/growing_season_index.npy'
        # growing_season_f = Phenology_based_on_Temperature_NDVI().this_class_arr + 'growing_season_index.npy'
        growing_season_dic = T.load_npy(growing_season_f)
        ############### plot growing season spatial #######################
        # spatial_dic = {}
        # for pix in growing_season_dic:
        #     start = growing_season_dic[pix][0]
        #     spatial_dic[pix] = start
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr)
        # plt.colorbar()
        # plt.show()
        ############### plot growing season spatial #######################

        event_dic = T.load_npy(events_f)

        events_num = 0.
        for pix in event_dic:
            events = event_dic[pix]
            events_num += len(events)
        print('events_num', events_num)

        spei_dic = T.load_npy(drought_index_f)
        event_with_marks_dic = {}
        for pix in event_dic:
            if not pix == (246, 239):
                continue
            if not pix in growing_season_dic:
                continue
            gs = growing_season_dic[pix]['GS_mon']
            if len(gs) < 3:
                continue
            early_start = growing_season_dic[pix]['early_start']
            early_end = growing_season_dic[pix]['early_end']
            late_start = growing_season_dic[pix]['late_start']
            late_end = growing_season_dic[pix]['late_end']

            early_gs = list(range(early_start, early_end + 1))
            peak_gs = list(range(early_end + 1, late_start))
            late_gs = list(range(late_start, late_end + 1))

            if len(gs) == 3:
                early_gs = [gs[0]]
                peak_gs = [gs[1]]
                late_gs = [gs[2]]
            if len(peak_gs) == 0:
                if len(late_gs) != 1:
                    peak_gs.append(late_gs[0])
                    late_gs.pop(0)
                else:
                    peak_gs.append(early_gs[-1])
                    early_gs.pop(-1)

            spei = spei_dic[pix]
            events = event_dic[pix]
            if len(events) > 0:
                event_with_marks = []
                for event in events:
                    min_indx = T.pick_min_indx_from_1darray(spei, event)
                    mon = min_indx % 12 + 1
                    if mon in early_gs:
                        mark = 'early'
                    elif mon in peak_gs:
                        mark = 'mid'
                    elif mon in late_gs:
                        mark = 'late'
                    else:
                        mark = 'dormant'
                    print(mark)
                    print(event)
                    print(min_indx)
                    print(mon)
                    print(early_gs)
                    print(peak_gs)
                    print(late_gs)
                    pause()
                    event_with_marks.append([mark, event])
                event_with_marks_dic[pix] = event_with_marks
        np.save(outf, event_with_marks_dic)
        pass

    def pick(self, f, outdir):
        # 前n个月和后n个月无极端干旱事件
        n = 24.
        T.mk_dir(outdir, force=True)
        single_event_dic = {}
        dic = T.load_npy(f)
        for pix in tqdm(dic, desc='picking {}'.format(f)):
            vals = dic[pix]
            # print list(vals)
            # f = '{}_{}.txt'.format(pix[0],pix[1])
            # fw = open(f,'w')
            # fw.write(str(list(vals)))
            # fw.close()
            # pause()
            mean = np.nanmean(vals)
            std = np.std(vals)
            threshold = mean - 2 * std
            # threshold = -1.5
            # threshold = np.quantile(vals, 0.05)
            event_list, key = self.kernel_find_drought_period([vals, pix, threshold])
            if len(event_list) == 0:
                continue
            events_4 = []
            for i in event_list:
                level, drought_range = i
                events_4.append(drought_range)

            single_event = []
            for i in range(len(events_4)):
                if i - 1 < 0:  # 首次事件
                    if events_4[i][0] - n < 0 or events_4[i][-1] + n >= len(vals):  # 触及两边则忽略
                        continue
                    if len(events_4) == 1:
                        single_event.append(events_4[i])
                    elif events_4[i][-1] + n <= events_4[i + 1][0]:
                        single_event.append(events_4[i])
                    continue

                # 最后一次事件
                if i + 1 >= len(events_4):
                    if events_4[i][0] - events_4[i - 1][-1] >= n and events_4[i][-1] + n <= len(vals):
                        single_event.append(events_4[i])
                    break

                # 中间事件
                if events_4[i][0] - events_4[i - 1][-1] >= n and events_4[i][-1] + n <= events_4[i + 1][0]:
                    single_event.append(events_4[i])
            # print single_event
            # sleep(0.1)
            single_event_dic[pix] = single_event
            # for evt in single_event:
            #     picked_vals = T.pick_vals_from_1darray(vals,evt)
            #     plt.scatter(evt,picked_vals,color='r')
            # plt.plot(vals)
            # plt.show()
        np.save(outdir + 'single_events', single_event_dic)
        # spatial_dic = {}
        # for pix in single_event_dic:
        #     evt_num = len(single_event_dic[pix])
        #     if evt_num == 0:
        #         continue
        #     spatial_dic[pix] = evt_num
        # DIC_and_TIF().plot_back_ground_arr()
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr)
        # plt.colorbar()
        # plt.show()

    def kernel_find_drought_period(self, params):
        # 根据不同干旱程度查找干旱时期
        pdsi = params[0]
        key = params[1]
        threshold = params[2]
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
            # print(new_i)
            # exit()
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
            if min_val < threshold:
                level = 4
            # if -1 <= min_val < -.5:
            #     level = 1
            # elif -1.5 <= min_val < -1.:
            #     level = 2
            # elif -2 <= min_val < -1.5:
            #     level = 3
            # elif min_val <= -2.:
            #     level = 4
            else:
                level = 0

            events_list.append([level, new_i])
            # print(min_val)
            # plt.plot(vals)
            # plt.show()
        # for key in events_dic:
        #     # print key,events_dic[key]
        #     if 0 in events_dic[key][1]:
        #         print(events_dic[key])
        # exit()
        return events_list, key


class Recovery_time:

    def __init__(self):
        self.this_class_arr = result_root_this_script + 'arr/Recovery_time/'
        self.this_class_tif = result_root_this_script + 'tif/Recovery_time/'
        self.this_class_png = result_root_this_script + 'png/Recovery_time/'

        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        pass

    def run(self):
        # self.valid_pix()
        # # self.composite_NDVI()
        # # calculate func
        # ndvi_dir = data_root+'NDVI/per_pix_clean_anomaly/'
        # # ndvi_dir = data_root+'CSIF/per_pix_anomaly/'
        #
        # ## 1 #################################################
        # self.pick_class = Main_flow_Pick()
        # growing_season_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + \
        #                    'transform_early_peak_late_dormant_period_annual/gs_mons.npy'
        # winter_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + \
        #            'transform_early_peak_late_dormant_period_annual/dormant_mons.npy'
        # # # # 1 CWD
        # events_f = self.pick_class.this_class_arr+'events_with_timing_cwd/events_with_timing.npy'
        # # events_f = this_root + 'Main_flow_results/arr/Main_flow_Pick/events_with_timing_cwd/events_with_timing_trans.npy'
        # # drought_index_f = data_root + '/CWD/per_pix_2001_2015_detrend/CWD.npy'
        # drought_index_f = data_root + '/CWD/per_pix_1982_2015_detrend/CWD.npy'
        # out_dir = self.this_class_arr + 'recovery_time_CWD/'
        # lc_dic = self.__gen_lc_dic()
        # self.gen_recovery_time(events_f,drought_index_f,ndvi_dir,out_dir,growing_season_f,winter_f,lc_dic)
        # ####################################################
        #
        # # # SPEI 1 - 12
        # ## 2 ################################################
        # event_dir = self.pick_class.this_class_arr + 'events_with_timing_spei/'
        # out_dir = self.this_class_arr + 'recovery_time_spei/'
        # T.mk_dir(out_dir)
        # for folder in T.listdir(event_dir):
        #     # print folder
        #     events_f = event_dir + folder + '/events_with_timing.npy'
        #     # events_f = event_dir + folder + '/events_with_timing_trans.npy'
        #     # drought_index_f = data_root + 'SPEI/compose_spei_n_to_one_file_2001_2015/{}.npy'.format(folder)
        #     drought_index_f = data_root + 'SPEI/compose_spei_n_to_one_file/{}.npy'.format(folder)
        #     out_dir_i = out_dir + folder + '/'
        #     self.gen_recovery_time(events_f,drought_index_f,ndvi_dir,out_dir_i,growing_season_f,winter_f,lc_dic)
        ####################################################
        # compose CWD and SPEI events
        ## 3 ################################################
        # self.compose_CWD_SPEI_1_12()

        # self.real_data_plot()
        # self.real_data_plot2()
        self.real_data_plot_to_excel()

    def composite_NDVI(self):
        fdir = data_root + 'NDVI/per_pix_clean_anomaly_smooth_detrend/'
        outf = self.this_class_arr + 'NDVI'
        dic = {}
        for f in T.listdir(fdir):
            dic_i = T.load_npy(fdir + f)
            dic.update(dic_i)
        np.save(outf, dic)
        pass

    def valid_pix(self):
        ndvi_valid_arr = np.load(NDVI().this_class_arr + 'NDVI_invalid_mask.npy')
        tropical_arr = np.load(NDVI().this_class_arr + 'NDVI_tropical.npy')

        ndvi_valid_dic = D.spatial_arr_to_dic(ndvi_valid_arr)
        tropical_dic = D.spatial_arr_to_dic(tropical_arr)
        valid_mask_dic = {}
        for pix in ndvi_valid_dic:
            val = ndvi_valid_dic[pix]
            if np.isnan(val):
                continue
            valid_mask_dic[pix] = 1

        tropical_mask_dic = {}
        for pix in tropical_dic:
            val = tropical_dic[pix]
            if val == 1:
                tropical_mask_dic[pix] = 1

        self.valid_mask_dic = valid_mask_dic
        self.tropical_mask_dic = tropical_mask_dic

    def __gen_lc_dic(self):

        tif = data_root + 'landcover/Tiff/glc2000_0_5.tif'
        arr = to_raster.raster2array(tif)[0]
        arr = np.array(arr, dtype=int)
        evergreen_list = [1, 4]
        deciduous_list = [2, 3, 5]
        lc_dic = {}
        for r in range(len(arr)):
            for c in range(len(arr[0])):
                pix = (r, c)
                val = arr[pix]
                if val in evergreen_list:
                    forest_type = 'evergreen'
                elif val in deciduous_list:
                    forest_type = 'deciduous'
                else:
                    forest_type = 'other'
                lc_dic[pix] = forest_type
        return lc_dic

    def gen_recovery_time(self, events_f, drought_index_f, ndvi_dir, out_dir, gs_f, winter_f, lc_dic):
        '''
        生成全球恢复期
        :param interval: SPEI_{interval}
        :return:
        '''

        # pre_dic = Main_flow_Prepare().load_X_anomaly('PRE')

        growing_season_daterange = T.load_npy(gs_f)
        # print growing_season_daterange
        # exit()
        Tools().mk_dir(out_dir, force=True)
        outf = out_dir + 'recovery_time'
        # 1 加载事件
        # interval = '%02d' % interval
        events = T.load_npy(events_f)
        # 2 加载NDVI 与 SPEI
        ndvi_dic = T.load_npy_dir(ndvi_dir)
        spei_dic = T.load_npy(drought_index_f)
        winter_dic = T.load_npy(winter_f)
        recovery_time_dic = {}
        for pix in tqdm(ndvi_dic):
            # print type(pix)
            # print pix in events
            # print pix in growing_season_daterange
            # print '...'
            forest_type = lc_dic[pix]
            if pix in events and pix in growing_season_daterange:
                # growing_date_range = [5,6,7,8,9,10]
                # print growing_date_range
                # pre = pre_dic[pix]
                # pre = np.array(pre)
                winter_mon_list = winter_dic[pix]
                winter_index = []
                for year, mons in enumerate(winter_mon_list):
                    for mon in mons:
                        winter_index_i = year * 12 + mon
                        winter_index.append(winter_index_i)
                # 连续值分组
                # winter_ranges = []
                # for _, group in groupby(enumerate(winter_index), lambda (index, item): index - item):
                #     group = map(itemgetter(1), group)
                #     if len(group) > 1:
                #         winter_ranges.append(range(group[0], group[-1] + 1))
                #     else:
                #         winter_ranges.append([group[0]])

                # gs_index = []
                # gs_mon_list = growing_season_daterange[pix]
                # for year, mons in enumerate(gs_mon_list):
                #     for mon in mons:
                #         gs_index_i = year * 12 + mon
                #         gs_index.append(gs_index_i)
                #
                # print winter_dic[pix]
                # print growing_season_daterange[pix]
                # print winter_dic[pix]
                # print growing_season_daterange[pix]
                # exit()

                ndvi = ndvi_dic[pix]
                ndvi = np.array(ndvi)
                # ndvi = SMOOTH().forward_window_smooth(ndvi,window=6)
                spei = spei_dic[pix]
                spei = np.array(spei)
                event = events[pix]
                recovery_time_result = []
                for timing, date_range in event:
                    # 1 挑出此次干旱事件的NDVI和SPEI值 （画图需要）
                    spei_picked_vals = Tools().pick_vals_from_1darray(spei, date_range)
                    # 2 挑出此次干旱事件SPEI最低的索引
                    # 在当前生长季搜索
                    # event_start_index = date_range[0]
                    event_start_index = T.pick_min_indx_from_1darray(spei, date_range)
                    # print date_range
                    year_indx = event_start_index // 12
                    if len(growing_season_daterange[pix]) != (2015 - 1982 + 1):
                        continue
                    # print year_indx
                    growing_date_range = growing_season_daterange[pix][year_indx]  ################

                    # event_start_index = date_range[-1]

                    # 4 搜索恢复期
                    # 4.1 获取growing season NDVI的最小值
                    # 4.3 搜索恢复到正常情况的时间，recovery_time：恢复期； mark：'in', 'out', 'tropical'
                    # mark: In Out Tropical
                    recovery_range, lag, recovery_start_gs, recovery_start, winter_mark, post_3_month_greater_than_threshold = \
                        self.search1(ndvi, spei, event_start_index, growing_date_range, forest_type)

                    # recovery_time, lag, recovery_start_gs, recovery_start, 'undefined'
                    ###########################################
                    ###########################################
                    ###########################################
                    if recovery_range == None:
                        continue
                    recovery_range = np.array(recovery_range)
                    date_range = np.array(date_range)
                    recovery_time = len(recovery_range)
                    recovery_time_result.append({
                        'recovery_time': recovery_time,
                        'recovery_date_range': recovery_range,
                        'drought_event_date_range': date_range,
                        'timing': timing,
                        'lag': lag,
                        'recovery_start_gs': recovery_start_gs,
                        'winter_mark': winter_mark,
                        # 'post_3_month_greater_than_threshold': post_3_month_greater_than_threshold,
                    })
                    #
                    # ################# plot ##################

                    # print('pix',pix)
                    # print('events_f',events_f)
                    # print('gs_mon',growing_date_range)
                    # print('post_3_month_greater_than_threshold',post_3_month_greater_than_threshold)
                    # # print('recovery_time',recovery_time, 'timing',timing,'lag',lag,'recovery_start_gs',recovery_start_gs,'winter_mark',winter_mark)
                    # # print('growing_date_range',growing_date_range)
                    # print('recovery_range',recovery_range)
                    # recovery_date_range = recovery_range
                    # recovery_ndvi = Tools().pick_vals_from_1darray(ndvi, recovery_date_range)
                    #
                    # tmp_pre_date_range = []
                    # for i in recovery_date_range:
                    #     tmp_pre_date_range.append(i)
                    # for i in date_range:
                    #     tmp_pre_date_range.append(i)
                    # tmp_pre_date_range = list(set(tmp_pre_date_range))
                    # tmp_pre_date_range.sort()
                    # # pre_picked_vals = Tools().pick_vals_from_1darray(pre, tmp_pre_date_range)
                    # # tmp_picked_vals = Tools().pick_vals_from_1darray(tmp, tmp_pre_date_range)
                    # # if len(swe) == 0:
                    # #     continue
                    # # swe_picked_vals = Tools().pick_vals_from_1darray(swe, tmp_pre_date_range)
                    #
                    # plt.figure(figsize=(8, 6))
                    # # plt.plot(tmp_pre_date_range, pre_picked_vals, '--', color='blue', label='precipitation')
                    # # plt.plot(tmp_pre_date_range, tmp_picked_vals, '--', color='cyan', label='temperature')
                    # # plt.plot(tmp_pre_date_range, swe_picked_vals, '--', color='black', linewidth=2, label='SWE',
                    # #          zorder=99)
                    # # plt.plot(recovery_date_range, recovery_ndvi, color='g', linewidth=6, label='Recovery Period')
                    # plt.scatter(recovery_date_range, recovery_ndvi, color='g', label='Recovery Period')
                    # plt.plot(date_range, spei_picked_vals, color='r', linewidth=6,
                    #          label='drought Event')
                    # plt.scatter(date_range, spei_picked_vals, color='r', zorder=99)
                    #
                    # plt.plot(range(len(ndvi)), ndvi, '--', color='g', zorder=99, label='ndvi')
                    # plt.plot(range(len(spei)), spei, '--', color='r', zorder=99, label='drought index')
                    # # plt.plot(range(len(pre)), pre, '--', color='blue', zorder=99, label='Precip')
                    # # pre_picked = T.pick_vals_from_1darray(pre,recovery_date_range)
                    # # pre_mean = np.nanmean(pre_picked)
                    # # plt.plot(recovery_date_range,[pre_mean]*len(recovery_date_range))
                    # plt.legend()
                    #
                    # minx = 9999
                    # maxx = -9999
                    #
                    # for ii in recovery_date_range:
                    #     if ii > maxx:
                    #         maxx = ii
                    #     if ii < minx:
                    #         minx = ii
                    #
                    # for ii in date_range:
                    #     if ii > maxx:
                    #         maxx = ii
                    #     if ii < minx:
                    #         minx = ii
                    # # print date_range[0]-5,recovery_date_range[-1]+5
                    #
                    # xtick = []
                    # for iii in np.arange(len(ndvi)):
                    #     year = 1982 + iii / 12
                    #     year = int(year)
                    #     mon = iii % 12 + 1
                    #     mon = '%02d' % mon
                    #     xtick.append('{}.{}'.format(year, mon))
                    # # # plt.xticks(range(len(xtick))[::3], xtick[::3], rotation=90)
                    # plt.xticks(range(len(xtick)), xtick, rotation=90)
                    # plt.grid()
                    # plt.xlim(minx - 5, maxx + 5)
                    #
                    # lon, lat, address = Tools().pix_to_address(pix)
                    # try:
                    #     plt.title('lon:{:0.2f} lat:{:0.2f} address:{}\n'.format(lon, lat, address) +
                    #               'recovery_time:'+str(recovery_time)+
                    #               ' timing:'+ str(timing)+ ' lag:'+ str(lag)+ ' recovery_start_gs:'+ str(recovery_start_gs)
                    #               )
                    #
                    # except:
                    #     plt.title('lon:{:0.2f} lat:{:0.2f}\n'.format(lon, lat)+
                    #               'recovery_time:' + str(recovery_time) +
                    #               ' timing:' + str(timing) + ' lag:' + str(lag) + ' recovery_start_gs:' + str(
                    #         recovery_start_gs)
                    #               )
                    # plt.show()

                    # #################plot##################

                recovery_time_dic[pix] = recovery_time_result
            else:
                recovery_time_dic[pix] = []
        # print recovery_time_dic
        T.save_dict_to_binary(recovery_time_dic, outf)
        pass

    def kernel_search1(self, recovery_start, event_start_index, growing_date_range, ndvi, normal_ndvi_threshold,
                       forest_type):
        success = 0
        winter_mark = 0
        recovery_range = []
        greater_flag = 0
        # = 'evergreen'
        for i in range(recovery_start, event_start_index + 36):
            mon = i % 12 + 1
            if not mon in growing_date_range:
                winter_mark = 1
                continue

            if event_start_index + 36 >= len(ndvi):
                break
            ndvi_i = ndvi[i]
            mon = i % 12 + 1
            if forest_type == 'evergreen':
                if ndvi_i > normal_ndvi_threshold:
                    greater_flag += 1
                    if greater_flag >= 2:
                        success = 1
                        break
                else:
                    greater_flag = 0
            else:
                if ndvi_i > normal_ndvi_threshold and mon in growing_date_range:
                    greater_flag += 1
                    if greater_flag >= 2:
                        success = 1
                        break
                else:
                    greater_flag = 0
            recovery_range.append(i)
        # NDVI needs to exceed the threshold for more than 3 consecutive months

        if success == 0:
            return None, None, None
        if len(recovery_range) == 0:
            post_recovery_range = list(range(recovery_start + 1, recovery_start + 12))
        else:
            post_recovery_range = list(range(recovery_range[-1] + 1, recovery_range[-1] + 12))
        post_recovery_range_new = []
        for i in post_recovery_range:
            mon = i % 12 + 1
            if not mon in growing_date_range:
                continue
            post_recovery_range_new.append(i)
            if len(post_recovery_range_new) == 3:
                break
        post_recovery_range_val = T.pick_vals_from_1darray(ndvi, post_recovery_range_new)
        is_exceed_threshold = post_recovery_range_val > normal_ndvi_threshold
        if False in is_exceed_threshold:
            post_3_month_greater_than_threshold = 0
        else:
            post_3_month_greater_than_threshold = 1
        recovery_range.pop(-1)
        return recovery_range, winter_mark, post_3_month_greater_than_threshold

    def search1(self, ndvi, drought_indx, event_start_index, growing_date_range, forest_type):
        normal_ndvi_threshold = -0.05

        # print 'drought_indx',drought_indx
        # print 'ndvi',ndvi
        # print 'event_start_index',event_start_index
        # print 'growing_date_range',growing_date_range
        # print 'timing',timing

        # seasonal cycle
        picked_ndvi_vals = []
        picked_ndvi_vals_i = []

        picked_ndvi_index = []
        picked_ndvi_index_i = []
        # 从event start index 开始，选取后36个月的NDVI
        for i in range(36):
            if (event_start_index + i) >= len(ndvi):  # 到头了
                break
            search_ = event_start_index + i
            search_mon = search_ % 12 + 1
            if not search_mon in growing_date_range:
                if len(picked_ndvi_vals_i) != 0:
                    picked_ndvi_vals.append(picked_ndvi_vals_i)
                    picked_ndvi_index.append(picked_ndvi_index_i)
                picked_ndvi_vals_i = []
                picked_ndvi_index_i = []
            else:
                ndvi_i = ndvi[search_]
                picked_ndvi_vals_i.append(ndvi_i)
                picked_ndvi_index_i.append(search_)
            #################################
        if len(picked_ndvi_vals) == 0:
            return None, None, None, None, None, None

        first_gs_min_ndvi = min(picked_ndvi_vals[0])
        if len(picked_ndvi_vals) == 1:
            second_gs_min_ndvi = 999
        else:
            second_gs_min_ndvi = min(picked_ndvi_vals[1])
        # lag is in the first GS
        if first_gs_min_ndvi < normal_ndvi_threshold:
            min_ind = T.pick_min_indx_from_1darray(picked_ndvi_vals[0], range(len(picked_ndvi_vals[0])))
            recovery_start = picked_ndvi_index[0][min_ind]
            lag = recovery_start - event_start_index
            recovery_range, winter_mark, post_3_month_greater_than_threshold = \
                self.kernel_search1(recovery_start, event_start_index, growing_date_range, ndvi, normal_ndvi_threshold,
                                    forest_type)
            # if winter_mark == 1:
            #     recovery_start = picked_ndvi_index[1][0]
            #     recovery_range, _ = \
            #         self.kernel_search1(recovery_start,event_start_index,growing_date_range,ndvi,normal_ndvi_threshold)
            recovery_start_gs = 'first'

        # lag is in the second GS
        elif second_gs_min_ndvi < normal_ndvi_threshold:
            recovery_start = picked_ndvi_index[1][0]
            recovery_range, _, post_3_month_greater_than_threshold = \
                self.kernel_search1(recovery_start, event_start_index, growing_date_range, ndvi, normal_ndvi_threshold,
                                    forest_type)
            winter_mark = 1
            lag = None
            recovery_start_gs = 'second'
        else:
            recovery_start = None
            lag = None
            recovery_range = None
            recovery_start_gs = None
            winter_mark = None
            post_3_month_greater_than_threshold = None

        return recovery_range, lag, recovery_start_gs, recovery_start, winter_mark, post_3_month_greater_than_threshold

        pass

    def search(self, ndvi, drought_indx, event_start_index, growing_date_range, timing, winter_indx):
        normal_ndvi_threshold = -0.05

        # seasonal cycle
        picked_ndvi_vals = []
        picked_ndvi_vals_i = []

        picked_ndvi_index = []
        picked_ndvi_index_i = []
        # 从event start index 开始，选取后36个月的NDVI
        for i in range(36):
            if (event_start_index + i) >= len(ndvi):  # 到头了
                break
            search_ = event_start_index + i
            search_mon = search_ % 12 + 1
            if not search_mon in growing_date_range:
                if len(picked_ndvi_vals_i) != 0:
                    picked_ndvi_vals.append(picked_ndvi_vals_i)
                    picked_ndvi_index.append(picked_ndvi_index_i)
                picked_ndvi_vals_i = []
                picked_ndvi_index_i = []
            else:
                ndvi_i = ndvi[search_]
                picked_ndvi_vals_i.append(ndvi_i)
                picked_ndvi_index_i.append(search_)
            #################################
        if len(picked_ndvi_vals) == 0:
            return None, None, None, None, None

        first_gs_min_ndvi = min(picked_ndvi_vals[0])
        if len(picked_ndvi_vals) == 1:
            second_gs_min_ndvi = 999
        else:
            second_gs_min_ndvi = min(picked_ndvi_vals[1])
        recovery_start_gs = 'other'
        min_ind = np.nan
        if first_gs_min_ndvi < normal_ndvi_threshold:
            min_ind = T.pick_min_indx_from_1darray(picked_ndvi_vals[0], range(len(picked_ndvi_vals[0])))
            recovery_start = picked_ndvi_index[0][min_ind]
            recovery_start_gs = 'first'
        elif second_gs_min_ndvi < normal_ndvi_threshold:
            min_ind = T.pick_min_indx_from_1darray(picked_ndvi_vals[1], range(len(picked_ndvi_vals[1])))
            recovery_start = picked_ndvi_index[1][min_ind]
            recovery_start_gs = 'second'
        else:
            recovery_start = np.nan
            # print 'drought has no effect on NDVI'
        if np.isnan(recovery_start):
            return None, None, None, None, None
        winter_mark = 0
        lag = recovery_start - event_start_index
        recovery_time = 0
        success = 0

        for i in range(recovery_start, event_start_index + 36):
            mon = i % 12 + 1
            if not mon in growing_date_range:
                winter_mark = 1
            if event_start_index + 36 >= 408:
                break
            ndvi_i = ndvi[i]
            mon = i % 12 + 1
            recovery_time += 1
            if ndvi_i > normal_ndvi_threshold and mon in growing_date_range:
                success = 1
                break
        if success == 0:
            return None, None, None, None, None
        recovery_time = recovery_time + 1
        return recovery_time, lag, recovery_start_gs, recovery_start, winter_mark

    def compose_CWD_SPEI_1_12(self):
        outdir = self.this_class_arr + 'compose_CWD_SPEI_recovery_time/'
        cwd_f = self.this_class_arr + 'recovery_time_CWD/recovery_time.pkl'
        spei_dir = self.this_class_arr + 'recovery_time_spei/'
        T.mk_dir(outdir)
        outf = outdir + 'compose'
        # cwd_f = self.reovery_time_class.this_class_arr + 'recovery/recovery_time.npy'

        void_dic = DIC_and_TIF().void_spatial_dic()
        cwd_dic = T.load_dict_from_binary(cwd_f)
        for pix in cwd_dic:
            for event in cwd_dic[pix]:
                void_dic[pix].append(event)

        for f in tqdm(T.listdir(spei_dir)):
            # print f
            scale = int(f[-2:])
            if scale > 6:
                continue
            # exit()
            dic_i = T.load_dict_from_binary(spei_dir + f + '/recovery_time.pkl')
            for pix in dic_i:
                for event in dic_i[pix]:
                    void_dic[pix].append(event)
        T.save_dict_to_binary(void_dic, outf)

    def vals_to_time_series(self, vals, datetime_obj_list, picked_index=None):

        data = dict(zip(datetime_obj_list, vals))
        Series = pd.Series(data)
        if picked_index is not None:
            Series = Series.iloc[picked_index]
        return Series

    def real_data_plot2(self):
        '''
        picked sites
        '''
        outdir = join(self.this_class_png, 'real_data_plot/')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        # exit()
        picked_fdir = join(self.this_class_png, 'picked_sites/')
        lon_lat_str_list = []
        for f in T.listdir(picked_fdir):
            lon_lat_str_list.append(f)

        recovery_time_f = join(self.this_class_arr, 'recovery_time_spei/spei01/recovery_time.pkl')
        drought_index_f = join(data_root, 'SPEI/compose_spei_n_to_one_file/spei01.npy')
        ndvi_dir = join(data_root, 'NDVI/per_pix_clean_anomaly')
        growing_season_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + \
                           'transform_early_peak_late_dormant_period_annual/gs_mons.npy'
        winter_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + \
                   'transform_early_peak_late_dormant_period_annual/dormant_mons.npy'
        df = Global_vars().load_df()
        pixels_list = T.get_df_unique_val_list(df, 'pix')
        growing_season_mons_dict = T.load_npy(growing_season_f)
        winter_mons_dict = T.load_npy(winter_f)
        recovery_dict = T.load_dict_from_binary(recovery_time_f)
        drought_index_dict = T.load_npy(drought_index_f)
        ndvi_dict = T.load_npy_dir(ndvi_dir)
        start_date = datetime.datetime(1982, 1, 1)
        end_date = datetime.datetime(2015, 12, 1)
        date_range = pd.date_range(start_date, end_date, freq='MS')
        date_obj_list = []
        for i in date_range:
            date_obj = datetime.datetime(i.year, i.month, 1)
            date_obj_list.append(date_obj)
        for pix in drought_index_dict:
            # if not pix == (66, 416):
            #     continue
            if not pix in ndvi_dict:
                continue
            lon, lat = DIC_and_TIF().pix_to_lon_lat(pix)
            lon_lat_str = f'{lon}_{lat}'
            if not lon_lat_str in lon_lat_str_list:
                continue
            in_list = [[lon, lat, {'lon': lon, 'lat': lat}]]

            if lon < 0:
                continue
            if lat > 70:
                continue
            if lat < 30:
                continue
            events = recovery_dict[pix]

            gs_mons = growing_season_mons_dict[pix]
            winter_mons = winter_mons_dict[pix]
            # print(winter_mons)
            # print(len(winter_mons))
            # print(len(gs_mons))
            # exit()
            drought_index_vals = drought_index_dict[pix]
            ndvi = ndvi_dict[pix]
            if len(events) < 2:
                continue
            # print(len(events))
            winter_mark_list = []
            for event in events:
                recovery_time = event['recovery_time']
                recovery_date_range = event['recovery_date_range']
                drought_event_date_range = event['drought_event_date_range']
                timing = event['timing']
                lag = event['lag']
                recovery_start_gs = event['recovery_start_gs']
                winter_mark = event['winter_mark']
                winter_mark_list.append(winter_mark)
            set_winter_mark_list = set(winter_mark_list)
            if len(set_winter_mark_list) < 2:
                continue
            r, c = pix

            for event in events:
                recovery_time = event['recovery_time']
                recovery_date_range = event['recovery_date_range']
                drought_event_date_range = event['drought_event_date_range']
                timing = event['timing']
                lag = event['lag']
                recovery_start_gs = event['recovery_start_gs']
                winter_mark = event['winter_mark']
                if timing == None:
                    continue
                # if winter_mark == 0:
                #     continue
                if recovery_time == 0:
                    continue
                print(pix, recovery_time, recovery_date_range, drought_event_date_range, timing, lag, recovery_start_gs,
                      winter_mark)
                outdir_i = join(outdir, f'{lon}_{lat}')
                T.mk_dir(outdir_i)
                outf = join(outdir_i, f'{str(drought_event_date_range[0])}_{recovery_date_range[0]}_{timing}' + '.pdf')
                outf_shp = join(outdir_i, f'{lon}_{lat}.shp')
                T.point_to_shp(in_list, outf_shp)
                # drought_year_index = drought_event_date_range[0] // 12
                # drought_year = drought_event_date_range[0] // 12 + 1982
                # dormant_mon = winter_mons[drought_year_index]
                dormant_mon_obj = []
                for year, dm in enumerate(winter_mons):
                    dormant_mon_obj_i = []
                    for dm_i in dm:
                        dm_obj = datetime.datetime(year + 1982 + 1, dm_i, 1)
                        dormant_mon_obj.append(dm_obj)

                ## get consecutive dormant_mon_obj dates
                dormant_mon_obj_consecutive = []
                dormant_mon_obj_consecutive_i = []
                for i in range(len(dormant_mon_obj)):
                    if i + 1 == len(dormant_mon_obj):
                        continue
                    if dormant_mon_obj[i + 1] - dormant_mon_obj[i] < datetime.timedelta(days=32):
                        dormant_mon_obj_consecutive_i.append(dormant_mon_obj[i])
                    else:
                        dormant_mon_obj_consecutive_i.append(dormant_mon_obj[i])
                        dormant_mon_obj_consecutive.append(dormant_mon_obj_consecutive_i)
                        dormant_mon_obj_consecutive_i = []

                # recovery_date_range_vals = T.pick_vals_from_1darray(ndvi,recovery_date_range)
                # drought_event_date_range_vals = T.pick_vals_from_1darray(drought_index_vals,drought_event_date_range)
                ndvi_ts = self.vals_to_time_series(ndvi, date_obj_list)
                drought_index_vals_ts = self.vals_to_time_series(drought_index_vals, date_obj_list)
                recovery_date_range_vals_ts = self.vals_to_time_series(ndvi, date_obj_list, recovery_date_range)
                drought_event_date_range_vals_ts = self.vals_to_time_series(drought_index_vals, date_obj_list,
                                                                            drought_event_date_range)
                # plt.fill_betweenx(date_obj_list,ndvi_ts,drought_index_vals_ts,alpha=0.5)

                try:
                    if winter_mark == 1:
                        winter_mark_str = 'MR'
                    else:
                        winter_mark_str = 'SR'
                    plt.figure(figsize=(7, 7))
                    title = f'lon:{lon} lat:{lat} mode:{winter_mark_str}'
                    plt.suptitle(title)
                    plt.subplot(2, 1, 1)
                    plt.plot(ndvi_ts, label='NDVI anomaly', c='g')
                    plt.scatter(recovery_date_range_vals_ts.index, recovery_date_range_vals_ts, s=60, c='g', marker='o')
                    # print(winter_mons)
                    # print(dormant_mon_obj_consecutive)
                    for i in range(len(dormant_mon_obj_consecutive)):
                        dormant_mon_obj_consecutive_i = dormant_mon_obj_consecutive[i]
                        plt.fill_betweenx((-3.5, 3.5), dormant_mon_obj_consecutive_i[0],
                                          dormant_mon_obj_consecutive_i[-1],
                                          alpha=0.3, zorder=-1, color='#1f77b4')

                    plt.subplot(2, 1, 1)
                    plt.plot(drought_index_vals_ts, label='SPEI01', c='r')
                    plt.scatter(drought_event_date_range_vals_ts.index, drought_event_date_range_vals_ts, s=60, c='r',
                                marker='o')
                    plt.ylim(-3.5, 3.5)
                    plt.xlim(ndvi_ts.index[drought_event_date_range[0]] - datetime.timedelta(days=365),
                             ndvi_ts.index[recovery_date_range[-1]] + datetime.timedelta(days=365))
                    plt.xticks(rotation=45, ha='right')
                    plt.grid()

                    plt.subplot(2, 1, 2)
                    land_tif = join(this_root, 'conf/land.tif')
                    DIC_and_TIF().plot_back_ground_arr_north_sphere(land_tif)
                    r, c = pix
                    plt.scatter(c, r, s=60, c='r', marker='o')
                    plt.xticks([])
                    plt.yticks([])

                    plt.tight_layout()

                    # plt.show()
                    plt.savefig(outf)
                    plt.close()
                    # exit()
                except Exception as e:
                    print('error')
                    print(e)
                # pause()

        pass

    def real_data_plot(self):
        outdir = join(self.this_class_png, 'real_data_plot/')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        # exit()

        recovery_time_f = join(self.this_class_arr, 'recovery_time_spei/spei01/recovery_time.pkl')
        drought_index_f = join(data_root, 'SPEI/compose_spei_n_to_one_file/spei01.npy')
        ndvi_dir = join(data_root, 'NDVI/per_pix_clean_anomaly')
        growing_season_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + \
                           'transform_early_peak_late_dormant_period_annual/gs_mons.npy'
        winter_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + \
                   'transform_early_peak_late_dormant_period_annual/dormant_mons.npy'
        df = Global_vars().load_df()
        pixels_list = T.get_df_unique_val_list(df, 'pix')
        growing_season_mons_dict = T.load_npy(growing_season_f)
        winter_mons_dict = T.load_npy(winter_f)
        recovery_dict = T.load_dict_from_binary(recovery_time_f)
        drought_index_dict = T.load_npy(drought_index_f)
        ndvi_dict = T.load_npy_dir(ndvi_dir)
        start_date = datetime.datetime(1982, 1, 1)
        end_date = datetime.datetime(2015, 12, 1)
        date_range = pd.date_range(start_date, end_date, freq='MS')
        date_obj_list = []
        for i in date_range:
            date_obj = datetime.datetime(i.year, i.month, 1)
            date_obj_list.append(date_obj)
        for pix in drought_index_dict:
            # if not pix == (66, 416):
            #     continue
            if not pix in ndvi_dict:
                continue
            lon, lat = DIC_and_TIF().pix_to_lon_lat(pix)
            in_list = [[lon, lat, {'col1': 'value1'}]]

            if lon < 0:
                continue
            if lat > 70:
                continue
            if lat < 30:
                continue
            events = recovery_dict[pix]

            gs_mons = growing_season_mons_dict[pix]
            winter_mons = winter_mons_dict[pix]
            # print(winter_mons)
            # print(len(winter_mons))
            # print(len(gs_mons))
            # exit()
            drought_index_vals = drought_index_dict[pix]
            ndvi = ndvi_dict[pix]
            if len(events) < 2:
                continue
            # print(len(events))
            winter_mark_list = []
            for event in events:
                recovery_time = event['recovery_time']
                recovery_date_range = event['recovery_date_range']
                drought_event_date_range = event['drought_event_date_range']
                timing = event['timing']
                lag = event['lag']
                recovery_start_gs = event['recovery_start_gs']
                winter_mark = event['winter_mark']
                winter_mark_list.append(winter_mark)
            set_winter_mark_list = set(winter_mark_list)
            if len(set_winter_mark_list) < 2:
                continue
            r, c = pix

            for event in events:
                recovery_time = event['recovery_time']
                recovery_date_range = event['recovery_date_range']
                drought_event_date_range = event['drought_event_date_range']
                timing = event['timing']
                lag = event['lag']
                recovery_start_gs = event['recovery_start_gs']
                winter_mark = event['winter_mark']
                if timing == None:
                    continue
                # if winter_mark == 0:
                #     continue
                if recovery_time == 0:
                    continue
                print(pix, recovery_time, recovery_date_range, drought_event_date_range, timing, lag, recovery_start_gs,
                      winter_mark)
                outdir_i = join(outdir, f'{lon}_{lat}')
                T.mk_dir(outdir_i)
                outf = join(outdir_i, f'{str(drought_event_date_range[0])}_{recovery_date_range[0]}_{timing}' + '.pdf')
                outf_shp = join(outdir_i, f'{lon}_{lat}.shp')
                T.point_to_shp(in_list, outf_shp)
                # drought_year_index = drought_event_date_range[0] // 12
                # drought_year = drought_event_date_range[0] // 12 + 1982
                # dormant_mon = winter_mons[drought_year_index]
                dormant_mon_obj = []
                for year, dm in enumerate(winter_mons):
                    dormant_mon_obj_i = []
                    for dm_i in dm:
                        dm_obj = datetime.datetime(year + 1982 + 1, dm_i, 1)
                        dormant_mon_obj.append(dm_obj)

                ## get consecutive dormant_mon_obj dates
                dormant_mon_obj_consecutive = []
                dormant_mon_obj_consecutive_i = []
                for i in range(len(dormant_mon_obj)):
                    if i + 1 == len(dormant_mon_obj):
                        continue
                    if dormant_mon_obj[i + 1] - dormant_mon_obj[i] < datetime.timedelta(days=32):
                        dormant_mon_obj_consecutive_i.append(dormant_mon_obj[i])
                    else:
                        dormant_mon_obj_consecutive_i.append(dormant_mon_obj[i])
                        dormant_mon_obj_consecutive.append(dormant_mon_obj_consecutive_i)
                        dormant_mon_obj_consecutive_i = []

                # recovery_date_range_vals = T.pick_vals_from_1darray(ndvi,recovery_date_range)
                # drought_event_date_range_vals = T.pick_vals_from_1darray(drought_index_vals,drought_event_date_range)
                ndvi_ts = self.vals_to_time_series(ndvi, date_obj_list)
                drought_index_vals_ts = self.vals_to_time_series(drought_index_vals, date_obj_list)
                recovery_date_range_vals_ts = self.vals_to_time_series(ndvi, date_obj_list, recovery_date_range)
                drought_event_date_range_vals_ts = self.vals_to_time_series(drought_index_vals, date_obj_list,
                                                                            drought_event_date_range)
                # plt.fill_betweenx(date_obj_list,ndvi_ts,drought_index_vals_ts,alpha=0.5)

                try:
                    if winter_mark == 1:
                        winter_mark_str = 'MR'
                    else:
                        winter_mark_str = 'SR'
                    plt.figure(figsize=(7, 7))
                    title = f'lon:{lon} lat:{lat} mode:{winter_mark_str}'
                    plt.suptitle(title)
                    plt.subplot(2, 1, 1)
                    plt.plot(ndvi_ts, label='NDVI anomaly', c='g')
                    plt.scatter(recovery_date_range_vals_ts.index, recovery_date_range_vals_ts, s=60, c='g', marker='o')
                    # print(winter_mons)
                    # print(dormant_mon_obj_consecutive)
                    for i in range(len(dormant_mon_obj_consecutive)):
                        dormant_mon_obj_consecutive_i = dormant_mon_obj_consecutive[i]
                        plt.fill_betweenx((-3.5, 3.5), dormant_mon_obj_consecutive_i[0],
                                          dormant_mon_obj_consecutive_i[-1],
                                          alpha=0.3, zorder=-1, color='#1f77b4')

                    plt.subplot(2, 1, 1)
                    plt.plot(drought_index_vals_ts, label='SPEI01', c='r')
                    plt.scatter(drought_event_date_range_vals_ts.index, drought_event_date_range_vals_ts, s=60, c='r',
                                marker='o')
                    plt.ylim(-3.5, 3.5)
                    plt.xlim(ndvi_ts.index[drought_event_date_range[0]] - datetime.timedelta(days=365),
                             ndvi_ts.index[recovery_date_range[-1]] + datetime.timedelta(days=365))
                    plt.xticks(rotation=45, ha='right')
                    plt.grid()

                    plt.subplot(2, 1, 2)
                    land_tif = join(this_root, 'conf/land.tif')
                    DIC_and_TIF().plot_back_ground_arr_north_sphere(land_tif)
                    r, c = pix
                    plt.scatter(c, r, s=60, c='r', marker='o')
                    plt.xticks([])
                    plt.yticks([])

                    plt.tight_layout()

                    # plt.show()
                    plt.savefig(outf)
                    plt.close()
                    # exit()
                except Exception as e:
                    print('error')
                    print(e)
                # pause()

        pass

    def real_data_plot_to_excel(self):
        '''
        picked sites
        '''
        outdir = join(self.this_class_png, 'real_data_plot/')
        T.mk_dir(outdir)
        # T.open_path_and_file(outdir)
        # exit()
        picked_fdir = join(self.this_class_png, 'picked_sites/')
        lon_lat_str_list = []
        for f in T.listdir(picked_fdir):
            lon_lat_str_list.append(f)
        # print(lon_lat_str_list)
        # exit()

        recovery_time_f = join(self.this_class_arr, 'recovery_time_spei/spei01/recovery_time.pkl')
        drought_index_f = join(data_root, 'SPEI/compose_spei_n_to_one_file/spei01.npy')
        ndvi_dir = join(data_root, 'NDVI/per_pix_clean_anomaly')
        growing_season_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + \
                           'transform_early_peak_late_dormant_period_annual/gs_mons.npy'
        winter_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + \
                   'transform_early_peak_late_dormant_period_annual/dormant_mons.npy'
        df = Global_vars().load_df()
        pixels_list = T.get_df_unique_val_list(df, 'pix')
        growing_season_mons_dict = T.load_npy(growing_season_f)
        winter_mons_dict = T.load_npy(winter_f)
        recovery_dict = T.load_dict_from_binary(recovery_time_f)
        drought_index_dict = T.load_npy(drought_index_f)
        ndvi_dict = T.load_npy_dir(ndvi_dir)
        start_date = datetime.datetime(1982, 1, 1)
        end_date = datetime.datetime(2015, 12, 1)
        date_range = pd.date_range(start_date, end_date, freq='MS')
        date_obj_list = []
        for i in date_range:
            date_obj = datetime.datetime(i.year, i.month, 1)
            date_obj_list.append(date_obj)
        for pix in drought_index_dict:
            # if not pix == (66, 416):
            #     continue
            if not pix in ndvi_dict:
                continue
            lon, lat = DIC_and_TIF().pix_to_lon_lat(pix)
            lon_lat_str = f'{lon}_{lat}'
            if not lon_lat_str in lon_lat_str_list:
                continue
            in_list = [[lon, lat, {'lon': lon, 'lat': lat}]]

            # if lon < 0:
            #     continue
            if lat > 70:
                continue
            if lat < 30:
                continue
            events = recovery_dict[pix]
            # print(events)
            # exit()

            gs_mons = growing_season_mons_dict[pix]
            winter_mons = winter_mons_dict[pix]
            # print(winter_mons)
            # print(len(winter_mons))
            # print(len(gs_mons))
            # exit()
            drought_index_vals = drought_index_dict[pix]
            ndvi = ndvi_dict[pix]
            if len(events) < 2:
                continue
            # print(len(events))
            winter_mark_list = []
            for event in events:
                recovery_time = event['recovery_time']
                recovery_date_range = event['recovery_date_range']
                drought_event_date_range = event['drought_event_date_range']
                timing = event['timing']
                lag = event['lag']
                recovery_start_gs = event['recovery_start_gs']
                winter_mark = event['winter_mark']
                winter_mark_list.append(winter_mark)
            set_winter_mark_list = set(winter_mark_list)
            if len(set_winter_mark_list) < 2:
                continue
            r, c = pix

            for event in events:
                recovery_time = event['recovery_time']
                recovery_date_range = event['recovery_date_range']
                drought_event_date_range = event['drought_event_date_range']
                timing = event['timing']
                lag = event['lag']
                recovery_start_gs = event['recovery_start_gs']
                winter_mark = event['winter_mark']
                if timing == None:
                    continue
                # if winter_mark == 0:
                #     continue
                if recovery_time == 0:
                    continue
                print(pix, recovery_time, recovery_date_range, drought_event_date_range, timing, lag, recovery_start_gs,
                      winter_mark)
                outdir_i = join(outdir, f'{lon}_{lat}')
                T.mk_dir(outdir_i)
                outf = join(outdir_i, f'{str(drought_event_date_range[0])}_{recovery_date_range[0]}_{timing}' + '.pdf')
                outf_shp = join(outdir_i, f'{lon}_{lat}.shp')
                T.point_to_shp(in_list, outf_shp)
                # drought_year_index = drought_event_date_range[0] // 12
                # drought_year = drought_event_date_range[0] // 12 + 1982
                # dormant_mon = winter_mons[drought_year_index]
                dormant_mon_obj = []
                for year, dm in enumerate(winter_mons):
                    dormant_mon_obj_i = []
                    for dm_i in dm:
                        dm_obj = datetime.datetime(year + 1982 + 1, dm_i, 1)
                        dormant_mon_obj.append(dm_obj)

                ## get consecutive dormant_mon_obj dates
                dormant_mon_obj_consecutive = []
                dormant_mon_obj_consecutive_i = []
                for i in range(len(dormant_mon_obj)):
                    if i + 1 == len(dormant_mon_obj):
                        continue
                    if dormant_mon_obj[i + 1] - dormant_mon_obj[i] < datetime.timedelta(days=32):
                        dormant_mon_obj_consecutive_i.append(dormant_mon_obj[i])
                    else:
                        dormant_mon_obj_consecutive_i.append(dormant_mon_obj[i])
                        dormant_mon_obj_consecutive.append(dormant_mon_obj_consecutive_i)
                        dormant_mon_obj_consecutive_i = []

                # recovery_date_range_vals = T.pick_vals_from_1darray(ndvi,recovery_date_range)
                # drought_event_date_range_vals = T.pick_vals_from_1darray(drought_index_vals,drought_event_date_range)
                ndvi_ts = self.vals_to_time_series(ndvi, date_obj_list)
                drought_index_vals_ts = self.vals_to_time_series(drought_index_vals, date_obj_list)
                recovery_date_range_vals_ts = self.vals_to_time_series(ndvi, date_obj_list, recovery_date_range)
                drought_event_date_range_vals_ts = self.vals_to_time_series(drought_index_vals, date_obj_list,
                                                                            drought_event_date_range)
                # plt.fill_betweenx(date_obj_list,ndvi_ts,drought_index_vals_ts,alpha=0.5)
                # print(ndvi_ts)
                # exit()
                df_out = pd.DataFrame({'ndvi': ndvi_ts, 'drought_index': drought_index_vals_ts}, index=date_obj_list)
                df_out.to_excel(join(outdir_i, f'{lon}_{lat}.xlsx'))
                # print(df_out)
                # exit()

                try:
                    if winter_mark == 1:
                        winter_mark_str = 'MR'
                    else:
                        winter_mark_str = 'SR'
                    plt.figure(figsize=(7, 7))
                    title = f'lon:{lon} lat:{lat} mode:{winter_mark_str}'
                    plt.suptitle(title)
                    plt.subplot(2, 1, 1)
                    plt.plot(ndvi_ts, label='NDVI anomaly', c='g')
                    plt.scatter(recovery_date_range_vals_ts.index, recovery_date_range_vals_ts, s=60, c='g', marker='o')
                    # print(winter_mons)
                    # print(dormant_mon_obj_consecutive)
                    for i in range(len(dormant_mon_obj_consecutive)):
                        dormant_mon_obj_consecutive_i = dormant_mon_obj_consecutive[i]
                        plt.fill_betweenx((-3.5, 3.5), dormant_mon_obj_consecutive_i[0],
                                          dormant_mon_obj_consecutive_i[-1],
                                          alpha=0.3, zorder=-1, color='#1f77b4')

                    plt.subplot(2, 1, 1)
                    plt.plot(drought_index_vals_ts, label='SPEI01', c='r')
                    plt.scatter(drought_event_date_range_vals_ts.index, drought_event_date_range_vals_ts, s=60, c='r',
                                marker='o')
                    plt.ylim(-3.5, 3.5)
                    plt.xlim(ndvi_ts.index[drought_event_date_range[0]] - datetime.timedelta(days=365),
                             ndvi_ts.index[recovery_date_range[-1]] + datetime.timedelta(days=365))
                    plt.xticks(rotation=45, ha='right')
                    plt.grid()

                    plt.subplot(2, 1, 2)
                    land_tif = join(this_root, 'conf/land.tif')
                    DIC_and_TIF().plot_back_ground_arr_north_sphere(land_tif)
                    r, c = pix
                    plt.scatter(c, r, s=60, c='r', marker='o')
                    plt.xticks([])
                    plt.yticks([])

                    plt.tight_layout()

                    plt.show()
                    # plt.savefig(outf)
                    # plt.close()
                    # exit()
                except Exception as e:
                    print('error')
                    print(e)
                # pause()

        pass


class Dataframe:

    def __init__(self):
        self.this_class_arr = result_root_this_script + 'arr/Dataframe/'
        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + 'Dataframe.df'
        self.P_PET_fdir = join(self.this_class_arr, 'aridity_P_PET_dic')

    def run(self):
        df = self.__gen_df_init()

        # 1 add drought information
        # df = self.add_recovery_to_df(df)
        # df = self.winter_mark_new(df)
        # df = self.prepare_phenology_event(df)
        # df = self.add_current_season_SOS(df)
        # df = self.add_drought_year_SOS(df)

        # 2 add location information
        # df = self.add_lon_lat_to_df(df)
        # df = self.add_koppen_landuse_to_df(df)
        # df = self.add_landuse_to_df(df)
        # df = self.add_kp_humid_arid_area_to_df(df)
        # df = self.add_humid_arid_area_to_df(df)
        # df = self.add_forest_type_to_df(df)
        # df = self.add_new_lc_type_to_df(df)

        # 3 add climate conditions
        # df = self.prepare_prior_vars_n(df)
        # df = self.prepare_climate_var_in_recovery(df)
        # df = self.prepare_climate_var_in_recovery_VPD_CRU(df)
        # df = self.prepare_lag_vars(df)
        # df = self.add_dormant_variables_to_df(df)
        # df = self.add_drought_year_vars(df)
        # df = self.add_second_year_spring_vars(df)
        # 4 add constant variables
        # df = self.add_correlation_to_df(df)
        # df = self.prepare_MAT_MAP_cv(df)
        # df = self.add_soil_HI(df)
        # df = self.add_Humid_nonhumid(df)
        # df = self.add_col_row_to_df(df)
        # df = self.add_SR_sos_to_drought_start_variables_to_df(df)
        df = self.add_MR_drought_start_to_eos_variables_to_df(df)
        # df = self.add_drought_severity(df)
        T.save_df(df, self.dff)
        self.__df_to_excel(df, self.dff, random=True)

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

    def __df_to_excel(self, df, dff, n=1000, random=False):
        if n == None:
            df.to_excel('{}.xlsx'.format(dff))
        else:
            if random:
                df = df.sample(n=n, random_state=1)
                df.to_excel('{}.xlsx'.format(dff))
            else:
                df = df.head(n)
                df.to_excel('{}.xlsx'.format(dff))

    def __load_X_anomaly(self, x):
        anomaly_fdir = data_root + '{}/per_pix_clean_anomaly/'.format(x)
        # per_pix_dir = data_root + '/{}/per_pix_clean/'.format(x)
        # mean_dir = this_root + 'data/{}/mon_mean/'.format(x)

        x_anomaly_dic = {}
        for f in tqdm(T.listdir(anomaly_fdir), desc='loading anomaly ...'):
            dic = T.load_npy(anomaly_fdir + f)
            x_anomaly_dic.update(dic)

        return x_anomaly_dic

    def __cal_phe_anomaly(self, dic, pix):

        var_list = []
        for y in range(len(dic[pix])):
            var_i = dic[pix][y]
            var_list.append(var_i)

        std_anomaly = {}
        anomaly = {}
        for i in range(len(var_list)):
            std = np.std(var_list)
            mean = np.nanmean(var_list)
            anomaly_i = var_list[i] - mean
            std_anomaly_i = (var_list[i] - mean) / std
            anomaly[i] = anomaly_i
            std_anomaly[i] = std_anomaly_i
        return anomaly, std_anomaly

    def __load_soil(self):
        sand_tif = data_root + 'HWSD/T_SAND_resample.tif'
        silt_tif = data_root + 'HWSD/T_SILT_resample.tif'
        clay_tif = data_root + 'HWSD/T_CLAY_resample.tif'

        sand_arr = to_raster.raster2array(sand_tif)[0]
        silt_arr = to_raster.raster2array(silt_tif)[0]
        clay_arr = to_raster.raster2array(clay_tif)[0]

        sand_arr[sand_arr < -9999] = np.nan
        silt_arr[silt_arr < -9999] = np.nan
        clay_arr[clay_arr < -9999] = np.nan

        sand_dic = DIC_and_TIF().spatial_arr_to_dic(sand_arr)
        silt_dic = DIC_and_TIF().spatial_arr_to_dic(silt_arr)
        clay_dic = DIC_and_TIF().spatial_arr_to_dic(clay_arr)

        return sand_dic

        pass

    def __load_HI(self):
        HI_tif = this_root + 'results/tif/WaterBalance/HI/HI_difference.tif'
        HI_arr = to_raster.raster2array(HI_tif)[0]
        HI_arr[HI_arr < -9999] = np.nan
        HI_dic = DIC_and_TIF().spatial_arr_to_dic(HI_arr)
        return HI_dic

    def add_recovery_to_df(self, df):
        recovery_spei_dir = Recovery_time().this_class_arr + 'recovery_time_spei/'
        recovery_cwd_f = Recovery_time().this_class_arr + 'recovery_time_CWD/recovery_time.pkl'

        events_f_list = []
        product_f_list = []
        for folder in T.listdir(recovery_spei_dir):
            product_f_list.append(folder)
            for f in T.listdir(recovery_spei_dir + folder):
                fpath = os.path.join(recovery_spei_dir, folder, f)
                events_f_list.append(fpath)
        product_f_list.append('cwd')
        events_f_list.append(recovery_cwd_f)
        timing_int_dic = {
            None: None,
            'early': 1,
            'mid': 2,
            'late': 3,
        }
        pix_list = []
        recovery_date_range_list = []
        # post_3_month_greater_than_threshold_list = []
        winter_mark_list = []
        recovery_time_list = []
        lag_list = []
        recovery_start_gs_list = []
        drought_event_date_range_list = []
        event_start_index_list = []
        timing_list = []
        timing_int_list = []
        product_type_list = []

        for i in tqdm(range(len(events_f_list))):
            fpath = events_f_list[i]
            product = product_f_list[i]
            dic = T.load_dict_from_binary(fpath)
            for pix in dic:
                events = dic[pix]
                if len(events) == 0:
                    continue
                for evt in events:
                    recovery_date_range = tuple(evt['recovery_date_range'])
                    winter_mark = evt['winter_mark']
                    recovery_time = evt['recovery_time']
                    # post_3_month_greater_than_threshold = evt['post_3_month_greater_than_threshold']
                    lag = evt['lag']
                    recovery_start_gs = evt['recovery_start_gs']
                    drought_event_date_range = tuple(evt['drought_event_date_range'])
                    event_start_index = drought_event_date_range[0]
                    timing = evt['timing']
                    timing_int = timing_int_dic[timing]

                    pix_list.append(pix)
                    recovery_date_range_list.append(recovery_date_range)
                    winter_mark_list.append(winter_mark)
                    recovery_time_list.append(recovery_time)
                    # post_3_month_greater_than_threshold_list.append(post_3_month_greater_than_threshold)
                    lag_list.append(lag)
                    recovery_start_gs_list.append(recovery_start_gs)
                    drought_event_date_range_list.append(drought_event_date_range)
                    event_start_index_list.append(event_start_index)
                    timing_list.append(timing)
                    timing_int_list.append(timing_int)
                    product_type_list.append(product)

        df['pix'] = pix_list
        df['product'] = product_type_list
        df['timing'] = timing_list
        df['timing_int'] = timing_int_list
        df['drought_event_date_range'] = drought_event_date_range_list
        df['lag'] = lag_list
        df['event_start_index'] = event_start_index_list
        df['recovery_date_range'] = recovery_date_range_list
        df['recovery_time'] = recovery_time_list
        # df['post_3_month_greater_than_threshold'] = post_3_month_greater_than_threshold_list
        df['winter_mark'] = winter_mark_list
        df['recovery_start_gs'] = recovery_start_gs_list

        return df

    def winter_mark_new(self, df):
        '''
        winter_mark_new
        过冬标记，恢复年份和干旱年份在同一年
        winter_mark
        过冬标记，恢复年份和干旱年份可能不在同一年，即可能在第二年才开始恢复
        '''
        winter_mark_new_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            winter_mark = row['winter_mark']
            drought_year = row['event_start_index'] // 12
            recovery_date_range = row['recovery_date_range']
            if len(recovery_date_range) == 0:
                winter_mark_new_list.append(np.nan)
                continue
            recovery_year = row['recovery_date_range'][0] // 12
            if drought_year == recovery_year:
                winter_mark_recovery = 0
            else:
                winter_mark_recovery = 1
            winter_mark_new = winter_mark or winter_mark_recovery
            winter_mark_new_list.append(winter_mark_new)
        df['winter_mark_new'] = winter_mark_new_list
        return df
        pass

    def add_lon_lat_to_df(self, df):
        pix_to_lon_lat_dic_f = DIC_and_TIF().this_class_arr + 'pix_to_lon_lat_dic.npy'
        lon_lat_dic = T.load_npy(pix_to_lon_lat_dic_f)
        # print(pix)
        lon_list = []
        lat_list = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc='adding lon lat into df'):
            pix = row.pix
            lon, lat = lon_lat_dic[pix]
            lon_list.append(lon)
            lat_list.append(lat)
        df['lon'] = lon_list
        df['lat'] = lat_list

        return df

    def add_koppen_landuse_to_df(self, df):
        # df = T.load_df(df_f)
        koppen_landuse_f = this_root + 'results/arr/Koppen_new/cross_koppen_landuse_pix.npy'
        koppen_landuse_dic = T.load_npy(koppen_landuse_f)
        koppen_landuse_dic_set = {}
        for kl in koppen_landuse_dic:
            pixs = koppen_landuse_dic[kl][0]
            pixs = set(pixs)
            koppen_landuse_dic_set[kl] = pixs

        kl_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix_ = row.pix
            kl_ = None
            for kl in koppen_landuse_dic_set:
                pixs = koppen_landuse_dic_set[kl]
                if pix_ in pixs:
                    kl_ = kl
                    break
            kl_list.append(kl_)
        df['climate_zone'] = kl_list
        return df

    def add_landuse_to_df(self, df):
        # df_f = Prepare_CWD_X_pgs_egs_lgs2().this_class_arr + 'prepare/data_frame.df'
        # df = T.load_df(df_f)
        kp_list = []
        lc_list = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc='adding landuse and koppen to df columns'):
            kl = row.climate_zone
            if kl == None:
                kp_list.append(None)
                lc_list.append(None)
                continue
            lc, kp = kl.split('.')
            kp_list.append(kp)
            lc_list.append(lc)
        df['kp'] = kp_list
        df['pft'] = lc_list
        return df

    def add_kp_humid_arid_area_to_df(self, df):
        # df_f = Prepare_CWD_X_pgs_egs_lgs2().this_class_arr + 'prepare/data_frame.df'
        # df = T.load_df(df_f)
        humid_arid_list = []
        humid_arid_dic = {
            'A': None,
            'B': 'arid',
            'Cf': 'humid',
            'Csw': 'arid',
            'Df': 'humid',
            'Dsw': 'arid',
            'E': None,
        }
        for _, row in tqdm(df.iterrows(), total=len(df), desc='add_hot_cold_area_to_df'):
            kl = row.climate_zone
            if kl == None:
                humid_arid_list.append(None)
                continue
            lc, kp = kl.split('.')
            humid_arid = humid_arid_dic[kp]
            humid_arid_list.append(humid_arid)
        df['kp_humid_arid'] = humid_arid_list
        return df

    def add_humid_arid_area_to_df(self, df):
        humid_arid_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            hi = row['water_balance']
            if hi > 0:
                humid_arid = 'humid'
            elif hi < 0:
                humid_arid = 'arid'
            else:
                humid_arid = None
            humid_arid_list.append(humid_arid)
        df['humid_arid'] = humid_arid_list

        return df

    def add_forest_type_to_df(self, df):
        # df_f = Prepare_CWD_X_pgs_egs_lgs2().this_class_arr + 'prepare/data_frame.df'
        # df = T.load_df(df_f)
        tif = data_root + 'landcover/Tiff/glc2000_0_5.tif'
        arr = to_raster.raster2array(tif)[0]
        arr = np.array(arr, dtype=int)
        forest_list = []
        evergreen_list = [1, 4]
        deciduous_list = [2, 3, 5]
        for _, row in tqdm(df.iterrows(), total=len(df), desc='adding forests to df columns'):
            pix = row.pix
            val = arr[pix]
            if val in evergreen_list:
                forest_type = 'evergreen'
            elif val in deciduous_list:
                forest_type = 'deciduous'
            else:
                forest_type = None
            forest_list.append(forest_type)
        df['forest_type'] = forest_list
        return df

    def add_new_lc_type_to_df(self, df):
        lc_new = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pft = row.pft
            forest_type = row.forest_type
            # print(forest_type)
            if forest_type == None:
                lc_new.append(pft)
            else:
                lc_new.append(forest_type)
        df['lc'] = lc_new

        return df

    def prepare_phenology_event(self, df):
        fdir = Main_flow_Early_Peak_Late_Dormant().this_class_arr + 'transform_early_peak_late_dormant_period_annual/'
        for f in T.listdir(fdir):
            var = f.split('.')[0]
            dic = T.load_npy(fdir + f)
            val_list = []
            for i, row in tqdm(df.iterrows(), desc=var, total=len(df)):
                pix = row.pix
                event_start_index = row['event_start_index']
                event_start_year = event_start_index // 12
                vals = dic[pix]
                var_event_val = vals[event_start_year]
                val_list.append(var_event_val)
            df[var] = val_list
        return df

    def add_current_season_SOS(self, df):
        # second year SOS
        sos_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + \
                'transform_early_peak_late_dormant_period_annual/early_start.npy'
        sos_dic = T.load_npy(sos_f)
        sos_list = []
        sos_anomaly_list = []
        sos_std_anomaly_list = []
        for indx, row in tqdm(df.iterrows(), total=len(df), desc='add_current_season_SOS...'):
            pix = row['pix']
            event_start = row['event_start_index']
            winter_mark_new = row['winter_mark']
            if winter_mark_new == 1:
                recovery_start_year_index = event_start // 12 + 1
            else:
                recovery_start_year_index = event_start // 12
            sos = sos_dic[pix]
            if len(sos) != 34:
                sos_list.append(None)
                sos_anomaly_list.append(None)
                sos_std_anomaly_list.append(None)
                continue
            sos_anomaly, sos_std_anomaly = self.__cal_phe_anomaly(sos_dic, pix)
            sos_i = sos[recovery_start_year_index]
            sos_anomaly_i = sos_anomaly[recovery_start_year_index]
            sos_std_anomaly_i = sos_std_anomaly[recovery_start_year_index]

            sos_list.append(sos_i)
            sos_anomaly_list.append(sos_anomaly_i)
            sos_std_anomaly_list.append(sos_std_anomaly_i)

        df['current_sos'] = sos_list
        df['current_sos_anomaly'] = sos_anomaly_list
        df['current_sos_std_anomaly'] = sos_std_anomaly_list

        return df

    def add_drought_year_SOS(self, df):
        sos_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + \
                'transform_early_peak_late_dormant_period_annual/early_start.npy'
        sos_dic = T.load_npy(sos_f)
        sos_list = []
        sos_anomaly_list = []
        sos_std_anomaly_list = []
        for indx, row in tqdm(df.iterrows(), total=len(df), desc='add_current_season_SOS...'):
            pix = row['pix']
            drought_start = row['drought_event_date_range'][0]
            winter_mark = row['winter_mark']
            year_index = drought_start // 12
            # if winter_mark == 1:
            #     year_index += 1
            # else:
            #     sos_list.append(None)
            #     sos_anomaly_list.append(None)
            #     sos_std_anomaly_list.append(None)
            #     continue
            sos = sos_dic[pix]
            if len(sos) != 34:
                sos_list.append(None)
                sos_anomaly_list.append(None)
                sos_std_anomaly_list.append(None)
                continue
            sos_anomaly, sos_std_anomaly = self.__cal_phe_anomaly(sos_dic, pix)
            sos_i = sos[year_index]
            sos_anomaly_i = sos_anomaly[year_index]
            sos_std_anomaly_i = sos_std_anomaly[year_index]

            sos_list.append(sos_i)
            sos_anomaly_list.append(sos_anomaly_i)
            sos_std_anomaly_list.append(sos_std_anomaly_i)

        df['drought_year_sos'] = sos_list
        df['drought_year_sos_anomaly'] = sos_anomaly_list
        df['drought_year_sos_std_anomaly'] = sos_std_anomaly_list

        return df

    def cal_dormant_sos_anomaly(self, df):
        # dormant length is not calendar year
        fdir = Main_flow_Early_Peak_Late_Dormant().this_class_arr + 'transform_early_peak_late_dormant_period_annual/'
        early_start_f = fdir + 'early_start.npy'
        late_end_f = fdir + 'late_end.npy'
        early_start_dic = T.load_npy(early_start_f)
        late_end_dic = T.load_npy(late_end_f)
        dormant_length_anomaly_dic = {}
        dormant_length_std_anomaly_dic = {}
        early_start_anomaly_dic = {}
        early_start_std_anomaly_dic = {}
        dormant_length_dic = {}
        for pix in tqdm(early_start_dic, desc='cal anomaly'):
            early_start = early_start_dic[pix]
            late_end = late_end_dic[pix]
            if len(early_start) == 0:
                continue
            dormant_length = []
            for y in range(len(early_start)):
                if y + 1 >= len(early_start):
                    continue
                early_start_i = early_start[y + 1]
                late_end_i = late_end[y]
                dormant_length_i = 365 - late_end_i + early_start_i
                dormant_length.append(dormant_length_i)

            dormant_length_mean = np.nanmean(dormant_length)
            early_start_mean = np.nanmean(early_start)
            dormant_length_std = np.std(dormant_length)
            early_start_std = np.std(early_start)
            dormant_length_anomaly = []
            dormant_length_std_anomaly = []
            for i in dormant_length:
                dormant_length_anomaly_i = i - dormant_length_mean
                dormant_length_anomaly.append(dormant_length_anomaly_i)
                dormant_length_std_anomaly_i = dormant_length_anomaly_i / dormant_length_std
                dormant_length_std_anomaly.append(dormant_length_std_anomaly_i)

            early_start_anomaly = []
            early_start_std_anomaly = []
            for i in early_start:
                early_start_anomaly_i = i - early_start_mean
                early_start_anomaly.append(early_start_anomaly_i)
                early_start_std_anomaly_i = early_start_anomaly_i / early_start_std
                early_start_std_anomaly.append(early_start_std_anomaly_i)
            dormant_length_anomaly_dic[pix] = dormant_length_anomaly
            dormant_length_std_anomaly_dic[pix] = dormant_length_std_anomaly
            early_start_anomaly_dic[pix] = early_start_anomaly
            early_start_std_anomaly_dic[pix] = early_start_std_anomaly
            dormant_length_dic[pix] = dormant_length

        dormant_length_anomaly_list = []
        dormant_length_std_anomaly_list = []
        sos_anomaly_list = []
        sos_std_anomaly_list = []
        dormant_length_list = []

        for i, row in tqdm(df.iterrows(), desc='adding dormant sos anomaly to df..', total=len(df)):
            pix = row.pix
            event_start_index = row['event_start_index']
            event_start_year = event_start_index // 12

            dormant_length_anomaly = dormant_length_anomaly_dic[pix][event_start_year]
            dormant_length_std_anomaly = dormant_length_std_anomaly_dic[pix][event_start_year]
            sos_anomaly = early_start_anomaly_dic[pix][event_start_year]
            sos_std_anomaly = early_start_std_anomaly_dic[pix][event_start_year]
            dormant_length = dormant_length_dic[pix][event_start_year]

            dormant_length_anomaly_list.append(dormant_length_anomaly)
            dormant_length_std_anomaly_list.append(dormant_length_std_anomaly)
            sos_anomaly_list.append(sos_anomaly)
            sos_std_anomaly_list.append(sos_std_anomaly)
            dormant_length_list.append(dormant_length)

        df['dormant_length_anomaly'] = dormant_length_anomaly_list
        df['dormant_length_std_anomaly'] = dormant_length_std_anomaly_list
        df['dormant_length'] = dormant_length_list

        return df

    def prepare_climate_var_in_recovery(self, df):
        for x in ['PRE', 'TMP', 'VPD']:
            x_anomaly_dic = self.__load_X_anomaly(x)
            picked_vals_mean_list = []
            for indx, row in tqdm(df.iterrows(), total=len(df), desc='{} prepare_climate_var_in_recovery...'.format(x)):
                pix = row['pix']
                if not pix in x_anomaly_dic:
                    picked_vals_mean_list.append(np.nan)
                    continue
                if len(row['recovery_date_range']) == 0:
                    picked_vals_mean_list.append(np.nan)
                    continue
                recovery_date_range = row['recovery_date_range']

                vals = x_anomaly_dic[pix]
                vals = np.array(vals)

                picked_vals = T.pick_vals_from_1darray(vals, recovery_date_range)
                picked_vals_mean = np.nanmean(picked_vals)
                picked_vals_mean_list.append(picked_vals_mean)
            row_name = 'post_{}'.format(x)
            df[row_name] = picked_vals_mean_list

        return df

    def prepare_climate_var_in_recovery_VPD_CRU(self, df):
        # dict_f = data_root + 'CRU_VPD/VPD/per_pix_anomaly/VPD.npy'
        dict_f = data_root + 'CRU_VPD/VPD/per_pix/VPD.npy'
        x_anomaly_dic = T.load_npy(dict_f)
        picked_vals_mean_list = []
        for indx, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in x_anomaly_dic:
                picked_vals_mean_list.append(np.nan)
                continue
            if len(row['recovery_date_range']) == 0:
                picked_vals_mean_list.append(np.nan)
                continue
            recovery_date_range = row['recovery_date_range']

            vals = x_anomaly_dic[pix]
            vals = Pre_Process().z_score(vals)  ### z score
            vals = np.array(vals)
            picked_vals = T.pick_vals_from_1darray(vals, recovery_date_range)
            picked_vals_mean = np.nanmean(picked_vals)
            picked_vals_mean_list.append(picked_vals_mean)
        row_name = 'post_VPD_CRU'
        df[row_name] = picked_vals_mean_list

        return df

    def add_dormant_variables_to_df(self, df):
        for x in ['PRE', 'TMP', 'VPD', 'SWE_terra']:
            # for x in ['SWE_Terra']:
            winter_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + \
                       'transform_early_peak_late_dormant_period_annual/dormant_mons.npy'
            gs_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + \
                   'transform_early_peak_late_dormant_period_annual/gs_mons.npy'
            winter_dic = T.load_npy(winter_f)
            gs_dic = T.load_npy(gs_f)
            x_anomaly_dic = self.__load_X_anomaly(x)
            # x_anomaly_dic = {}
            picked_vals_mean_list = []
            for indx, row in tqdm(df.iterrows(), total=len(df), desc='{} add_dormant_variables_to_df...'.format(x)):
                pix = row['pix']
                gs = gs_dic[pix]
                winter = winter_dic[pix]
                event_start = row['drought_event_date_range'][0]
                event_year = event_start // 12
                part1 = []
                for m in range(gs[event_year][-1], 12):
                    if not m in gs:
                        part1.append(event_year * 12 + m)
                part2 = []
                if len(winter[event_year + 1]) == 0:
                    picked_vals_mean_list.append(None)
                    continue
                for m in range(winter[event_year + 1][0], winter[event_year + 1][-1]):
                    if m in gs[event_year + 1]:
                        break
                    part2.append(event_year * 12 + m + 11)
                date_range_include_winter = part1 + part2
                # print 'part1',part1
                # print 'part2',part2
                # print 'gs[event_year]',gs[event_year]
                # print 'gs[event_year+1]',gs[event_year+1]
                # print 'winter[event_year+1]',winter[event_year+1]
                # print 'event_start',event_start
                # print 'date_range_include_winter',date_range_include_winter
                # pause()
                winter_mark = row['winter_mark']
                if winter_mark == 0:
                    picked_vals_mean_list.append(None)
                    continue
                if not pix in x_anomaly_dic:
                    picked_vals_mean_list.append(None)
                    continue
                vals = x_anomaly_dic[pix]

                picked_vals = T.pick_vals_from_1darray(vals, date_range_include_winter)
                if len(picked_vals) == 0:
                    picked_vals_mean_list.append(None)
                    continue
                picked_vals_mean = np.nanmean(picked_vals)
                picked_vals_mean_list.append(picked_vals_mean)
            row_name = 'dormant_{}'.format(x)
            df[row_name] = picked_vals_mean_list
        return df

    def add_drought_year_vars(self, df):
        for x in ['PRE', 'TMP', 'VPD']:
            x_anomaly_dic = self.__load_X_anomaly(x)
            picked_vals_mean_list = []
            for indx, row in tqdm(df.iterrows(), total=len(df), desc='{} prepare_climate_var_in_recovery...'.format(x)):
                pix = row['pix']
                if not pix in x_anomaly_dic:
                    picked_vals_mean_list.append(np.nan)
                    continue
                if len(row['recovery_date_range']) == 0:
                    picked_vals_mean_list.append(np.nan)
                    continue
                drought_event_date_range = row['drought_event_date_range']
                gs_mons = row['gs_mons']
                gs_mons = list(gs_mons)
                drought_year = np.mean(drought_event_date_range)
                drought_year = int(drought_year) // 12
                vals = x_anomaly_dic[pix]
                vals = np.array(vals)
                annual_vals = T.monthly_vals_to_annual_val(vals, gs_mons)
                picked_vals_mean_list.append(annual_vals[drought_year])
            row_name = '{}_drought_year'.format(x)
            df[row_name] = picked_vals_mean_list

        return df

    def add_second_year_spring_vars(self, df):
        early_start_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + \
                        'transform_early_peak_late_dormant_period_annual/early_start_mon.npy'
        early_end_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + \
                      'transform_early_peak_late_dormant_period_annual/early_end_mon.npy'
        early_start_dic = T.load_npy(early_start_f)
        early_end_dic = T.load_npy(early_end_f)
        for x in ['PRE', 'TMP', 'VPD']:
            x_anomaly_dic = self.__load_X_anomaly(x)
            picked_vals_mean_list = []
            for indx, row in tqdm(df.iterrows(), total=len(df), desc='{} prepare_climate_var_in_recovery...'.format(x)):
                pix = row['pix']
                if not pix in x_anomaly_dic:
                    picked_vals_mean_list.append(np.nan)
                    continue
                if len(row['recovery_date_range']) == 0:
                    picked_vals_mean_list.append(np.nan)
                    continue
                drought_event_date_range = row['drought_event_date_range']
                drought_year = np.mean(drought_event_date_range)
                drought_year = int(drought_year) // 12 + 1
                vals = x_anomaly_dic[pix]
                vals = np.array(vals)
                early_start = early_start_dic[pix]
                early_end = early_end_dic[pix]
                if len(early_start) != 34:
                    picked_vals_mean_list.append(np.nan)
                    continue
                spring_range = list(range(early_start[drought_year], early_end[drought_year] + 1))
                annual_vals = T.monthly_vals_to_annual_val(vals, spring_range)
                picked_vals_mean_list.append(annual_vals[drought_year])
            row_name = '{}_second_year_spring'.format(x)
            df[row_name] = picked_vals_mean_list

        return df

    def prepare_prior_vars_n(self, df):
        pre_threshold = 6
        for x in ['PRE', 'NDVI', 'TMP', 'VPD', ]:
            # print x
            x_anomaly_dic = self.__load_X_anomaly(x)

            pre_mean_list = []
            for indx, row in tqdm(df.iterrows(), total=len(df), desc='{} prepare_prior_vars...'.format(x)):
                recovery_start_index = row['drought_event_date_range'][0]
                pix = row['pix']
                success = 1
                if not pix in x_anomaly_dic:
                    success = 0
                # if drought_end_index + post_threshold >= 408:
                #     success = 0
                if recovery_start_index - pre_threshold < 0:
                    success = 0
                if success == 0:
                    pre_mean_list.append(np.nan)
                    continue

                vals = x_anomaly_dic[pix]
                pre_indx = list(range(recovery_start_index - pre_threshold, recovery_start_index + 1))
                #
                try:
                    pre_vals = T.pick_vals_from_1darray(vals, pre_indx)
                except:
                    pre_vals = np.nan
                #
                pre_mean = np.nanmean(pre_vals)
                #
                pre_mean_list.append(pre_mean)

            df['{}_pre_{}_mean'.format(x, pre_threshold)] = pre_mean_list

        return df

    def prepare_lag_vars(self, Y_df):
        for x in ['PRE', 'TMP', 'VPD']:
            x_anomaly_dic = self.__load_X_anomaly(x)
            pre_mean_list = []
            for indx, row in tqdm(Y_df.iterrows(), total=len(Y_df), desc='{} prepare_lag_vars...'.format(x)):
                recovery_start_index = row['drought_event_date_range'][0]
                lag = row['lag']
                # print(lag)
                if lag == None:
                    pre_mean_list.append(np.nan)
                    continue
                if np.isnan(lag):
                    pre_mean_list.append(np.nan)
                    continue
                lag = int(lag)
                pix = row['pix']
                success = 1
                if not pix in x_anomaly_dic:
                    success = 0
                # if drought_end_index + post_threshold >= 408:
                #     success = 0
                if recovery_start_index + lag < 0:
                    success = 0
                if success == 0:
                    pre_mean_list.append(np.nan)
                    continue

                vals = x_anomaly_dic[pix]
                pre_indx = list(range(recovery_start_index, recovery_start_index + lag + 1))
                #
                try:
                    pre_vals = T.pick_vals_from_1darray(vals, pre_indx)
                except:
                    pre_vals = np.nan
                #
                pre_mean = np.nanmean(pre_vals)
                #
                pre_mean_list.append(pre_mean)

            Y_df['{}_lag_mean'.format(x)] = pre_mean_list
        return Y_df

        pass

    def prepare_MAT_MAP_cv(self, df):

        gs_dic = Main_flow_Early_Peak_Late_Dormant().return_gs()
        for x in ['PRE', 'TMP', 'VPD']:
            print(x)
            per_pix_dir = data_root + '/{}/per_pix_clean/'.format(x)
            anomaly_dir = data_root + '{}/per_pix_clean_anomaly/'.format(x)
            x_dic = {}
            for f in tqdm(T.listdir(per_pix_dir), desc='loading {} vals ...'.format(x)):
                dic = T.load_npy(per_pix_dir + f)
                x_dic.update(dic)

            # x_anomaly_dic = {}
            # for f in tqdm(T.listdir(anomaly_dir), desc='loading anomaly ...'):
            #     dic = T.load_npy(anomaly_dir + f)
            #     x_anomaly_dic.update(dic)
            x_anomaly_dic = x_dic
            mean_dic = {}
            for pix in tqdm(x_dic, desc='cal mean {}...'.format(x)):
                if not pix in gs_dic:
                    continue
                # print pix
                vals = x_dic[pix]
                gs = gs_dic[pix]

                selected_vals = []
                for i in range(len(vals)):
                    mon = i % 12 + 1
                    if mon in gs:
                        selected_vals.append(vals[i])
                mean = np.nanmean(selected_vals)
                mean_dic[pix] = mean

            cv_dic = {}  # mean annual / std annual
            for pix in tqdm(x_anomaly_dic, desc='cal CV {}...'.format(x)):
                if not pix in gs_dic:
                    continue
                vals = x_anomaly_dic[pix]
                gs = gs_dic[pix]

                selected_vals = []
                for i in range(len(vals)):
                    mon = i % 12 + 1
                    if mon in gs:
                        selected_vals.append(vals[i])
                selected_vals = np.array(selected_vals)
                selected_vals_annual = np.reshape(selected_vals, (34, len(gs)))
                year_mean_list = []
                for y in selected_vals_annual:
                    year_mean_list.append(np.nanmean(y))

                std = np.std(year_mean_list)
                mean = np.nanmean(year_mean_list)
                cv = mean / std
                cv_dic[pix] = cv
                pass

            mean_list = []
            cv_list = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc='adding {} to df...'.format(x)):
                pix = row['pix']
                if not pix in mean_dic:
                    mean_list.append(np.nan)
                    cv_list.append(np.nan)
                    continue
                mean = mean_dic[pix]
                cv = cv_dic[pix]
                mean_list.append(mean)
                cv_list.append(cv)
            df['MA_{}'.format(x)] = mean_list
            df['CV_{}'.format(x)] = cv_list

        return df
        pass

    def add_soil_HI(self, df):
        sand_dic = self.__load_soil()
        HI_dic = self.__load_HI()

        sand_list = []
        HI_list = []
        for indx, row in tqdm(df.iterrows(), total=len(df), desc='picking sand WB sos...'):
            pix = row['pix']
            event_start_index = row['event_start_index']
            event_start_year = event_start_index // 12
            sand = sand_dic[pix]
            HI = HI_dic[pix]
            HI_list.append(HI)
            sand_list.append(sand)

        df['sand'] = sand_list
        df['water_balance'] = HI_list

        return df

    def add_correlation_to_df(self, df):
        '''
        drought sensitivity
        '''
        ndvi_dir = data_root + 'NDVI/per_pix_clean_anomaly_smooth_detrend/'
        cwd_f = data_root + 'CWD/per_pix_clean_anomaly_smooth/CWD.npy'
        ndvi_dic = {}
        for f in T.listdir(ndvi_dir):
            dic = T.load_npy(ndvi_dir + f)
            ndvi_dic.update(dic)
        cwd_dic = T.load_npy(cwd_f)

        spatial_dic = DIC_and_TIF().void_spatial_dic_nan()

        for pix in tqdm(ndvi_dic, desc='calculating correlation...'):
            try:
                ndvi = ndvi_dic[pix]
                cwd = cwd_dic[pix]
                r, p = stats.pearsonr(ndvi, cwd)
                spatial_dic[pix] = r
            except:
                pass

        corr_list = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc='add_correlation_to_df'):
            pix = row.pix
            r = spatial_dic[pix]
            corr_list.append(r)
        df['correlation'] = corr_list
        return df

    def drop_n_std(self, vals, n=1):
        vals = np.array(vals)
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        up = mean + n * std
        down = mean - n * std
        vals[vals > up] = np.nan
        vals[vals < down] = np.nan
        return vals

    def P_PET_ratio(self, P_PET_fdir):
        # fdir = '/Volumes/NVME2T/wen_proj/20220111/aridity_P_PET_dic'
        fdir = P_PET_fdir
        dic = T.load_npy_dir(fdir)
        dic_long_term = {}
        for pix in dic:
            vals = dic[pix]
            vals = np.array(vals)
            vals = T.mask_999999_arr(vals, warning=False)
            vals[vals == 0] = np.nan
            if T.is_all_nan(vals):
                continue
            vals = self.drop_n_std(vals)
            long_term_vals = np.nanmean(vals)
            dic_long_term[pix] = long_term_vals
        return dic_long_term

    def P_PET_class(self):
        outdir = join(self.this_class_arr, 'P_PET_class')
        T.mkdir(outdir)
        outf = join(outdir, 'P_PET_class.npy')
        if isfile(outf):
            return T.load_npy(outf)
        dic = self.P_PET_ratio(self.P_PET_fdir)
        dic_reclass = {}
        for pix in dic:
            val = dic[pix]
            label = None
            # label = np.nan
            if val > 0.65:
                label = 'Humid'
                # label = 3
            elif val < 0.2:
                label = 'Arid'
                # label = 0
            elif val > 0.2 and val < 0.5:
                label = 'Semi Arid'
                # label = 1
            elif val > 0.5 and val < 0.65:
                label = 'Semi Humid'
                # label = 2
            dic_reclass[pix] = label
        T.save_npy(dic_reclass, outf)
        return dic_reclass

    def add_Humid_nonhumid(self, df):
        AI_dict = self.P_PET_ratio(self.P_PET_fdir)
        P_PET_dic_reclass = self.P_PET_class()
        df = T.add_spatial_dic_to_df(df, P_PET_dic_reclass, 'AI_reclass')
        df = T.add_spatial_dic_to_df(df, P_PET_dic_reclass, 'AI_class')
        df = df.dropna(subset=['AI_class'])
        df.loc[df['AI_reclass'] != 'Humid', ['AI_reclass']] = 'Dryland'
        df = T.add_spatial_dic_to_df(df, AI_dict, 'Aridity_index')
        return df

    def add_col_row_to_df(self, df):
        pix = df['pix'].values
        col_list = []
        row_list = []
        for row, col in pix:
            col_list.append(col)
            row_list.append(row)
        df['col'] = col_list
        df['row'] = row_list
        return df

    def add_SR_sos_to_drought_start_variables_to_df(self, df):
        # for SR
        sos_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + \
                'transform_early_peak_late_dormant_period_annual/early_start_mon.npy'
        sos_dic = T.load_npy(sos_f)
        # var_name = 'SM'
        var_name = 'NDVI'
        var_dir = join(data_root, f'{var_name}/per_pix_clean_anomaly_smooth')

        # var_name = 'spei03'
        # var_dir = join(data_root,f'SPEI/per_pix_clean_smooth_detrend/{var_name}/')
        var_dict = T.load_npy_dir(var_dir)
        SR_sos_to_drought_start_list = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc='add_sos_to_drought_variables_to_df'):
            pix = row.pix
            if not pix in var_dict:
                SR_sos_to_drought_start_list.append(np.nan)
                continue
            winter_mark = row.winter_mark_new
            if winter_mark == 1:
                SR_sos_to_drought_start_list.append(np.nan)
                continue
            timing = row.timing
            if timing == None:
                SR_sos_to_drought_start_list.append(np.nan)
                continue
            drought_event_date_range = row.drought_event_date_range
            drought_start = drought_event_date_range[0]
            drought_start_year_index = drought_start // 12
            sos_mon = sos_dic[pix][drought_start_year_index]
            sos_mon_index = drought_start_year_index * 12 + sos_mon - 1
            picked_time_index = list(range(sos_mon_index, drought_start + 1))
            vals = var_dict[pix]
            picked_vals = [vals[i] for i in picked_time_index]
            mean = np.nanmean(picked_vals)
            SR_sos_to_drought_start_list.append(mean)

        df[f'{var_name}_SR_sos_to_drought_start'] = SR_sos_to_drought_start_list
        return df

    def add_MR_drought_start_to_eos_variables_to_df(self, df):
        # for SR
        eos_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + \
                'transform_early_peak_late_dormant_period_annual/late_end_mon.npy'
        eos_dic = T.load_npy(eos_f)
        # var_name = 'SM'
        var_list = ['PRE', 'VPD', 'TMP']
        for var_name in var_list:
            if var_name == 'VPD':
                var_dir = join(data_root, f'{var_name}/per_pix_clean')
            else:
                var_dir = join(data_root, f'{var_name}/per_pix_clean_anomaly_smooth')
            # var_name = 'spei03'
            # var_dir = join(data_root,f'SPEI/per_pix_clean_smooth_detrend/{var_name}/')
            var_dict = T.load_npy_dir(var_dir)
            MR_drought_start_to_eos_list = []
            for i, row in tqdm(df.iterrows(), total=len(df), desc='add_MR_drought_start_to_eos_variables_to_df'):
                pix = row.pix
                if not pix in var_dict:
                    MR_drought_start_to_eos_list.append(np.nan)
                    continue
                winter_mark = row.winter_mark_new
                if winter_mark == 0:
                    MR_drought_start_to_eos_list.append(np.nan)
                    continue
                timing = row.timing
                if timing == None:
                    MR_drought_start_to_eos_list.append(np.nan)
                    continue
                drought_event_date_range = row.drought_event_date_range
                drought_start = drought_event_date_range[0]
                drought_start_year_index = drought_start // 12
                eos_mon = eos_dic[pix][drought_start_year_index]
                eos_mon_index = drought_start_year_index * 12 + eos_mon - 1
                picked_time_index = list(range(drought_start, eos_mon_index))
                vals = var_dict[pix]
                picked_vals = [vals[i] for i in picked_time_index]
                mean = np.nanmean(picked_vals)
                MR_drought_start_to_eos_list.append(mean)

            # df[f'{var_name}_MR_drought_start_to_eos_variables_to_df'] = MR_drought_start_to_eos_list
            df[f'{var_name}_MR_drought_start_to_eos'] = MR_drought_start_to_eos_list
        return df

    def add_drought_severity(self, df):
        # for SR
        sos_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + \
                'transform_early_peak_late_dormant_period_annual/early_start_mon.npy'
        sos_dic = T.load_npy(sos_f)

        var_name = 'spei03'
        var_dir = join(data_root, f'SPEI/per_pix_clean_smooth_detrend/{var_name}/')
        var_dict = T.load_npy_dir(var_dir)
        drought_severity = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc='add_sos_to_drought_variables_to_df'):
            pix = row.pix
            if not pix in var_dict:
                drought_severity.append(np.nan)
                continue
            product = row['product']
            if not product == var_name:
                drought_severity.append(np.nan)
                continue
            # print(1)

            timing = row.timing
            drought_event_date_range = row.drought_event_date_range
            vals = var_dict[pix]
            min_index = T.pick_min_indx_from_1darray(vals, drought_event_date_range)
            min_val = vals[min_index]
            drought_severity.append(min_val)
        # print(drought_severity)
        # exit()
        df[f'{var_name}_severity'] = drought_severity
        return df


class Analysis:

    def __init__(self):
        self.this_class_arr = result_root_this_script + 'arr/Analysis/'
        self.this_class_png = result_root_this_script + 'png/Analysis/'
        self.this_class_tif = result_root_this_script + 'tif/Analysis/'
        Tools().mk_dir(self.this_class_png, force=True)
        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)

    def run(self):
        # self.bar()
        # self.bar1()
        # self.export_bar_data()
        # self.bar_std()
        # self.gentine()
        # self.dormant_matrix()
        # self.spei_and_sos()
        # self.df_sos_partial_correlation()
        # self.partial_correlation()
        # self.plot_drought_and_sos()
        # self.drought_non_drought_sos()
        # self.swe_sos()
        # self.SR_biophysical_effect()
        # self.SR_biological_effect()
        # self.Aridity_index_analysis()
        # self.Aridity_index_analysis_equal_interval()
        # self.spatial_autocorrelation()
        self.latitude_recovery_output_excel()
        # self.difference_between_left_winter_and_drought_recovery()

        pass

    def SR_biophysical_effect(self):
        df = Global_vars().load_df()
        # var_name = 'spei03'
        # var_ = f'{var_name}_severity'
        var_ = f'SM_SR_sos_to_drought_start'
        # df = df[df['product']==var_name]
        df = df.dropna(subset=[var_])
        df = df.dropna(subset=['SM_SR_sos_to_drought_start'])
        df = df.dropna(subset=['winter_mark_new'])
        df = df[df['winter_mark_new'] == 0]
        df = df[df['winter_mark'] == 0]
        # df = df.dropna(subset=['timing'])
        # df = df.dropna(subset=['climate_zone'])
        df = df[df['timing'] != 'late']
        df = df[df['Aridity_index'] < 0.65]
        # df = df[df['Aridity_index']>0.65]
        # df = df.dropna()
        # winter_mark_new = df['winter_mark_new'].tolist()
        # winter_mark = df['winter_mark'].tolist()
        # print(winter_mark)
        # print(winter_mark_new)
        # exit()
        drought_year_sos_anomaly = df['drought_year_sos_anomaly'].tolist()
        # drought_year_sos_anomaly = df['drought_year_sos_anomaly'].tolist()
        SM_SR_sos_to_drought_start = df[var_].tolist()
        data = {'drought_year_sos_anomaly': drought_year_sos_anomaly, 'SM': SM_SR_sos_to_drought_start}
        # plt.hist(drought_year_sos_anomaly,bins=100)
        # plt.show()
        df_i = pd.DataFrame(data)
        sos_bins = np.arange(-40, 40, 8)
        sos_bins = [round(i, 2) for i in sos_bins]
        df_group, bins_name = self.df_bin(df_i, 'drought_year_sos_anomaly', sos_bins)
        # x_list = []
        y_list = []
        err_list = []
        for name, df_group_i in df_group:
            SM = df_group_i['SM'].tolist()
            mean = np.nanmean(SM)
            err, _, _ = self.uncertainty_err(SM)
            # x_list.append(name)
            y_list.append(mean)
            err_list.append(err)
        # print(bins_name)
        # exit()
        Plot().plot_line_with_error_bar(bins_name, y_list, err_list)
        plt.plot(bins_name, y_list, c='r')
        plt.xticks(rotation=90)
        plt.xlabel('SOS anomaly (days)')
        plt.ylabel('SM anomaly')
        # plt.title('Humid')
        plt.title('Arid')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def SR_biological_effect(self):
        df = Global_vars().load_df()
        var_name = 'spei03'
        # var_ = f'{var_name}_severity'
        df = df.dropna(subset=['NDVI_SR_sos_to_drought_start'])
        df = df.dropna(subset=['winter_mark_new'])
        df = df[df['winter_mark_new'] == 0]
        df = df[df['winter_mark'] == 0]
        df = df.dropna(subset=['timing'])
        df = df[df['timing'] != 'late']
        df = df.dropna(subset=['climate_zone'])
        df = df[df['recovery_time'] < 7]
        drought_year_sos_anomaly = df['NDVI_SR_sos_to_drought_start'].tolist()
        recovery = df['recovery_time'].tolist()
        data = {'drought_year_sos_anomaly': drought_year_sos_anomaly, 'recovery': recovery}
        df_plot = pd.DataFrame(data)
        recovery_list = T.get_df_unique_val_list(df_plot, 'recovery')
        bar_x = []
        mean_list = []
        err_list = []
        for recovery in recovery_list:
            df_plot_ = df_plot[df_plot['recovery'] == recovery]
            drought_year_sos_anomaly = df_plot_['drought_year_sos_anomaly'].tolist()
            drought_year_sos_anomaly = np.array(drought_year_sos_anomaly)
            mean = np.nanmean(drought_year_sos_anomaly)
            std = np.nanstd(drought_year_sos_anomaly)
            err, _, _ = self.uncertainty_err(drought_year_sos_anomaly)
            bar_x.append(recovery)
            mean_list.append(mean)
            err_list.append(err)

        plt.figure()
        Plot().plot_line_with_error_bar(bar_x, mean_list, err_list)
        plt.plot(bar_x, mean_list, '-o')
        plt.ylabel('NDVI anomaly')
        # plt.xticks(rotation=90)
        plt.xlabel('Drought recovery (months)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        # plt.plot(df_group)
        # plt.show()
        exit()
        # r,p = stats.pearsonr(drought_year_sos_anomaly,recovery)
        # print(r,p)
        # KDE_plot().plot_scatter(drought_year_sos_anomaly,recovery)
        # plt.ylabel('recovery time (months)')
        # plt.xlabel('NDVI_SR_sos_to_drought_start')

        plt.show()

    def __df_to_excel(self, df, dff, n=1000, random=False):
        if n == None:
            df.to_excel('{}.xlsx'.format(dff))
        else:
            if random:
                df = df.sample(n=n, random_state=1)
                df.to_excel('{}.xlsx'.format(dff))
            else:
                df = df.head(n)
                df.to_excel('{}.xlsx'.format(dff))

    def bar(self):
        outpngdir = self.this_class_png + 'bar/'
        T.mk_dir(outpngdir)
        df = Global_vars().load_df()
        # df = df[df['product']=='spei03']
        winter_mark = [0, 1]
        # winter_mark = [1]
        timing_list = ['early', 'mid', 'late', ]
        lim_list = [
            (1.3, 2.),
            (3, 7.)
        ]
        for w in winter_mark:
            df_w = df[df['winter_mark'] == w]
            for t in timing_list:
                df_t = df_w[df_w['timing'] == t]
                plt.figure()
                sns.barplot(data=df_t, x='lc', y='recovery_time', ci=99.999999, order=Global_vars().lc_list())
                mean_list = []
                for lc in Global_vars().lc_list():
                    df_lc = df_t[df_t['lc'] == lc]
                    y_val = df_lc['recovery_time'].to_list()
                    y_mean = np.nanmean(y_val)
                    mean_list.append(y_mean)
                print(mean_list)
                std = np.std(mean_list)
                print('std', std)
                # sns.barplot(data=df_t,x='lc',y='recovery_time',ci='sd',order=Global_vars().lc_list())
                title = '{}_{}'.format(w, t)
                plt.title(title)
                print(title)
                print('---')

                plt.ylim(lim_list[w])
                # plt.savefig(outpngdir + title + '.pdf')
                # plt.close()
        plt.show()
        pass

    def bar1(self):
        # add all lc
        outpngdir = self.this_class_png + 'bar1/'
        T.mk_dir(outpngdir)
        T.open_path_and_file(outpngdir)
        df = Global_vars().load_df()
        # df = df[df['product']=='spei03']
        winter_mark = [0, 1]
        # winter_mark = [1]
        timing_list = ['early', 'mid', 'late', ]
        lim_list = [
            (1.3, 2.),
            (3, 7.)
        ]
        for w in winter_mark:
            df_w = df[df['winter_mark'] == w]
            for t in timing_list:
                df_t = df_w[df_w['timing'] == t]
                plt.figure()
                # sns.barplot(data=df_t,x='lc',y='recovery_time',ci=99.999999,order=Global_vars().lc_list())
                mean_list = []
                std_list = []
                x_list = []
                for lc in Global_vars().lc_list():
                    df_lc = df_t[df_t['lc'] == lc]
                    y_val = df_lc['recovery_time'].to_list()
                    y_mean = np.nanmean(y_val)
                    # y_std = np.nanstd(y_val)
                    y_std = T.uncertainty_err(y_val)[0]
                    mean_list.append(y_mean)
                    std_list.append(y_std)
                    x_list.append(lc)
                y_val_all = df_t['recovery_time'].to_list()
                y_mean_all = np.nanmean(y_val_all)
                # y_std_all = np.nanstd(y_val_all)
                y_std_all = T.uncertainty_err(y_val_all)[0]
                mean_list.append(y_mean_all)
                std_list.append(y_std_all)
                x_list.append('all')
                title = '{}_{}'.format(w, t)
                plt.title(title)
                print(title)
                print('---')
                # plt.bar(x_list,mean_list,yerr=std_list)
                plt.bar(x_list, mean_list)
                for i in range(len(x_list)):
                    x = x_list[i]
                    y = mean_list[i]
                    plt.text(x, y, '{:.2f}'.format(y), ha='center', va='bottom')
                # plt.errorbar(x_list,mean_list,yerr=std_list)

                plt.ylim(lim_list[w])
                plt.savefig(outpngdir + title + '.pdf')
                plt.close()
        # plt.show()
        pass

    def uncertainty_err(self, vals):
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        up, bottom = stats.t.interval(0.95, len(vals) - 1, loc=mean, scale=std / np.sqrt(len(vals)))
        err = mean - bottom
        return err, up, bottom

    def df_bin(self, df, col, bins):
        df[f'{col}_bins'] = pd.cut(df[col], bins=bins)
        df_group = df.groupby([f'{col}_bins'])
        bins_name = df_group.groups.keys()
        bins_name_list = list(bins_name)
        bins_list_str = [str(i) for i in bins_name_list]
        # for name,df_group_i in df_group:
        #     vals = df_group_i[col].tolist()
        #     mean = np.nanmean(vals)
        #     err,_,_ = self.uncertainty_err(SM)
        #     # x_list.append(name)
        #     y_list.append(mean)
        #     err_list.append(err)
        return df_group, bins_list_str

    def Aridity_index_analysis(self):
        df = Global_vars().load_df()
        # AI_index = df['Aridity_index']
        df = df[df['product'] == 'spei03']
        # df = df[df['winter_mark']==1]
        AI_bins = [0, 0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 1, 1.25, 1.5, 2, 2.5, 3, np.inf]
        # AI_bins = np.arange(0.2,3,0.2)
        AI_bins = list(AI_bins)
        # AI_bins.append(np.inf)
        Aridity_index = df['Aridity_index'].tolist()
        pix_list = df['pix'].tolist()
        spatial_dict = dict(zip(pix_list, Aridity_index))
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        plt.imshow(arr, cmap='jet_r', vmin=0, vmax=2)

        df['AI_bin'] = pd.cut(df['Aridity_index'], bins=AI_bins)
        df_group = df.groupby(['AI_bin'])
        bins = df_group.groups.keys()
        # print(bins)
        # exit()
        bins_list = list(bins)
        mean_list = []
        err_list = []
        sample_number_list = []
        all_number_sum = 0
        for name, group in df_group:
            recovery_time = group['recovery_time'].to_list()
            recovery_time = np.array(recovery_time)
            mean = np.nanmean(recovery_time)
            std = np.nanstd(recovery_time)
            ## confidence interval
            err = 1.96 * std / np.sqrt(len(recovery_time))
            up, bottom = stats.t.interval(0.95, len(recovery_time) - 1, loc=mean,
                                          scale=std / np.sqrt(len(recovery_time)))
            err1 = mean - bottom
            print(name, len(recovery_time), mean, std)
            mean_list.append(mean)
            err_list.append(err1)
            sample_number_list.append(len(recovery_time))
            all_number_sum += len(recovery_time)
        sample_number_list = np.array(sample_number_list)
        sample_number_list = sample_number_list / all_number_sum
        plt.figure()
        bins_list_str = [str(i) for i in bins_list]
        # plt.errorbar(bins_list_str,mean_list,yerr=err_list,fmt='o')
        # print(bins_list)
        # exit()

        # plt.hist(Aridity_index, bins=180,range=(0,3),density=True)
        # x,y = Plot().plot_hist_smooth(Aridity_index, bins=100,range=(0,3),alpha=0)
        # plt.plot(x,y,color='red')
        width = []
        for i in range(len(AI_bins)):
            if i + 1 == len(AI_bins):
                break
            width_i = AI_bins[i + 1] - AI_bins[i]
            width.append(width_i)
        # print(width)
        # exit()
        print(len(width))
        print(len(AI_bins[:-1]))
        bar_x = []
        for i in range(len(AI_bins)):
            if i + 1 == len(AI_bins):
                break
            x_i = AI_bins[i] + width[i] / 2
            bar_x.append(x_i)
        plt.bar(AI_bins[:-1], sample_number_list, width=width, alpha=1, align='edge', color='gray', zorder=-1)
        # plt.twinx()
        plt.figure()
        Plot().plot_line_with_error_bar(bar_x, mean_list, err_list)
        plt.plot(bar_x, mean_list, '-o')
        plt.ylabel('recovery time')
        plt.xticks(rotation=90)
        plt.xlabel('Aridity index')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def Aridity_index_analysis_equal_interval(self):
        df = Global_vars().load_df()
        # AI_index = df['Aridity_index']
        df = df[df['product'] == 'spei03']
        # df = df[df['winter_mark']==1]
        # AI_bins = [0,0.05,0.2,0.35,0.5,0.65,0.8,1,1.25,1.5,2,2.5,3,np.inf]
        AI_bins = np.arange(0., 3, 0.2)
        AI_bins = list(AI_bins)
        AI_bins.append(np.inf)
        Aridity_index = df['Aridity_index'].tolist()
        pix_list = df['pix'].tolist()
        spatial_dict = dict(zip(pix_list, Aridity_index))
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        plt.imshow(arr, cmap='jet_r', vmin=0, vmax=2)

        df['AI_bin'] = pd.cut(df['Aridity_index'], bins=AI_bins)
        df_group = df.groupby(['AI_bin'])
        bins = df_group.groups.keys()
        # print(bins)
        # exit()
        bins_list = list(bins)
        mean_list = []
        err_list = []
        sample_number_list = []
        all_number_sum = 0
        for name, group in df_group:
            recovery_time = group['recovery_time'].to_list()
            recovery_time = np.array(recovery_time)
            mean = np.nanmean(recovery_time)
            std = np.nanstd(recovery_time)
            ## confidence interval
            err = 1.96 * std / np.sqrt(len(recovery_time))
            up, bottom = stats.t.interval(0.95, len(recovery_time) - 1, loc=mean,
                                          scale=std / np.sqrt(len(recovery_time)))
            err1 = mean - bottom
            print(name, len(recovery_time), mean, std)
            mean_list.append(mean)
            err_list.append(err1)
            sample_number_list.append(len(recovery_time))
            all_number_sum += len(recovery_time)
        sample_number_list = np.array(sample_number_list)
        sample_number_list = sample_number_list / all_number_sum
        plt.figure()
        bins_list_str = [str(i) for i in bins_list]
        # plt.errorbar(bins_list_str,mean_list,yerr=err_list,fmt='o')
        # print(bins_list)
        # exit()

        # plt.hist(Aridity_index, bins=180,range=(0,3),density=True)
        # x,y = Plot().plot_hist_smooth(Aridity_index, bins=100,range=(0,3),alpha=0)
        # plt.plot(x,y,color='red')
        width = []
        for i in range(len(AI_bins)):
            if i + 1 == len(AI_bins):
                break
            width_i = AI_bins[i + 1] - AI_bins[i]
            width.append(width_i)
        # print(width)
        # exit()
        print(len(width))
        print(len(AI_bins[:-1]))
        bar_x = []
        for i in range(len(AI_bins)):
            if i + 1 == len(AI_bins):
                break
            x_i = AI_bins[i] + width[i] / 2
            bar_x.append(x_i)
        plt.bar(AI_bins[:-1], sample_number_list, width=width, alpha=1, align='edge', color='gray', zorder=-1)
        # plt.twinx()
        plt.figure()
        Plot().plot_line_with_error_bar(bar_x, mean_list, err_list)
        plt.plot(bar_x, mean_list, '-o')
        plt.ylabel('recovery time')
        plt.xticks(rotation=90)
        plt.xlabel('Aridity index')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def export_bar_data(self):
        outdir = self.this_class_arr + 'export_bar_data/'
        T.mk_dir(outdir)
        col_list = ['winter_mark', 'timing', 'lc', 'recovery_time']
        df = Global_vars().load_df()
        df = df[col_list]
        df = df[df['lc'] != 'Forest']
        T.save_df(df, outdir + 'drought_recovery.df')
        # # df = df[df['product']=='spei03']
        # winter_mark = [0, 1]
        # # winter_mark = [1]
        # timing_list = ['early', 'mid', 'late', ]
        # lim_list = [
        #     (1.3, 2.),
        #     (3, 7.)
        # ]
        # for w in winter_mark:
        #     df_w = df[df['winter_mark'] == w]
        #     for t in timing_list:
        #         df_t = df_w[df_w['timing'] == t]
        #         plt.figure()
        #         sns.barplot(data=df_t, x='lc', y='recovery_time', ci=95, order=Global_vars().lc_list())
        #         mean_list = []
        #         for lc in Global_vars().lc_list():
        #             df_lc = df_t[df_t['lc'] == lc]
        #             y_val = df_lc['drought recovery'].to_list()
        #             y_mean = np.nanmean(y_val)
        #             mean_list.append(y_mean)
        #         print(mean_list)
        #         std = np.std(mean_list)
        #         print('std', std)
        #         # sns.barplot(data=df_t,x='lc',y='recovery_time',ci='sd',order=Global_vars().lc_list())
        #         title = '{}_{}'.format(w, t)
        #         plt.title(title)
        #         print(title)
        #         print('---')
        #
        #         plt.ylim(lim_list[w])
        #         # plt.savefig(outpngdir + title + '.pdf')
        #         # plt.close()
        # plt.show()
        pass

    def bar_std(self):
        outpngdir = self.this_class_png + 'bar/'
        T.mk_dir(outpngdir)
        df = Global_vars().load_df()
        # df = df[df['product']=='spei03']
        # product = T.get_df_unique_val_list(df,'product')
        # print(product)
        # exit()
        # lc_list = T.get_df_unique_val_list(df,'lc')
        # print(lc_list)
        # exit()
        winter_mark_list = [0, 1]
        # winter_mark = [1]
        timing_list = ['early', 'mid', 'late', ]
        lim_list = [
            (1.3, 2.),
            (3, 7.)
        ]
        for w in winter_mark_list:
            df_w = df[df['winter_mark'] == w]
            for t in timing_list:
                df_t = df_w[df_w['timing'] == t]
                recovery = df_t['recovery_time'].to_list()
                std = np.nanstd(recovery)
                mean = np.nanmean(recovery)
                title = '{}_{}'.format(w, t)
                print(title, mean, std)
        for w in winter_mark_list:
            df_w = df[df['winter_mark'] == w]
            recovery = df_w['recovery_time'].to_list()
            std = np.nanstd(recovery)
            mean = np.nanmean(recovery)
            title = '{}'.format(w)
            print(title, mean, std)
        df_new = pd.DataFrame()
        index_selected = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            timing = row['timing']
            winter_mark = row['winter_mark']
            if timing == 'late' and winter_mark == 0:
                continue
            index_selected.append(i)
        df_new = df.loc[index_selected]
        # df_new = df_new[df_new['recovery_time']<12]
        # T.print_head_n(df_new)
        lc_list = Global_vars().lc_list()
        for lc in lc_list:
            df_lc = df_new[df_new['lc'] == lc]
            rc = df_lc['recovery_time'].to_list()
            rc_mean = np.nanmean(rc)
            print(lc, rc_mean)
        print(lc_list)
        exit()
        for t in timing_list:
            df_t = df_new[df_new['timing'] == t]
            df_deciduous = df_t[df_t['lc'] == 'deciduous']
            df_non_deciduous = df_t[df_t['lc'] != 'deciduous']
            recovery_deciduous = df_deciduous['recovery_time'].to_list()
            recovery_non_deciduous = df_non_deciduous['recovery_time'].to_list()
            # std = np.nanstd(recovery)
            mean_deciduous = np.nanmean(recovery_deciduous)
            mean_non_deciduous = np.nanmean(recovery_non_deciduous)
            print(t, mean_deciduous)
            print(t, mean_non_deciduous)
            deciduous_random_sample = np.random.choice(recovery_deciduous, size=len(recovery_deciduous))
            non_deciduous_random_sample = np.random.choice(recovery_non_deciduous, size=len(recovery_deciduous))
            mse = sklearn.metrics.mean_squared_error(deciduous_random_sample, non_deciduous_random_sample)
            rmse = math.sqrt(mse)
            print('rmse', t, rmse)
        exit()
        for w in winter_mark_list:
            df_w = df[df['winter_mark'] == w]
            df_early = df_w[df_w['timing'] == 'early']
            df_other = df_w[df_w['timing'] != 'early']
            recovery_early = df_early['recovery_time'].tolist()
            recovery_non_early = df_other['recovery_time'].tolist()
            early_mean = np.nanmean(recovery_early)
            other_mean = np.nanmean(recovery_non_early)
            early_random_sample = np.random.choice(recovery_early, size=len(recovery_early))
            other_random_sample = np.random.choice(recovery_non_early, size=len(recovery_early))
            mse = sklearn.metrics.mean_squared_error(early_random_sample, other_random_sample)
            rmse = math.sqrt(mse)
            print(w, 'early mean', early_mean)
            print(w, 'other_mean', other_mean)
            print(w, 'rmse', rmse)

    def gentine(self):
        outdir = self.this_class_png + 'gentine/'
        T.mk_dir(outdir)
        outf = outdir + 'scatter.pdf'
        y_var = 'current_sos_anomaly'
        # y_var = 'drought_year_sos_anomaly'
        recovery_var = 'recovery_time'
        # x_var = 'dormant_SWE_Terra'
        # x_var = 'dormant_SWE_Terra'
        x_var = 'dormant_TMP'
        # x_var = 'dormant_length'
        df = Global_vars().load_df()
        humid_arid_list = ['humid', 'arid']
        humid_arid_color = {
            'humid': 'b',
            'arid': 'r'
        }
        df = df[df['winter_mark'] == 1]
        for humid_arid in humid_arid_list:
            df_humid_arid = df[df['humid_arid'] == humid_arid]
            # df = df[df['humid_arid']=='arid']

            x_bins = np.linspace(-2., 2., 21)
            # x_bins = np.linspace(-2.6,0,21)
            # x_bins = np.linspace(0,2.6,21)
            # x_bins = np.linspace(150,220,21)

            box = []
            scatter = []
            box_len = []
            for i in range(len(x_bins)):
                if i + 1 >= len(x_bins):
                    continue
                df_x = df_humid_arid[df_humid_arid[x_var] > x_bins[i]]
                df_x = df_x[df_x[x_var] < x_bins[i + 1]]
                # print(len(df_t))
                y = df_x[y_var].to_list()
                recovery = df_x[recovery_var].to_list()
                # print(len(y))
                box_len.append(len(y))
                r, p = T.nan_correlation(y, recovery)
                if p > 0.001:
                    r = np.nan
                    print(i)
                scatter.append(r)
                box.append(r)
            # print(box)
            # KDE_plot().plot_scatter(range(len(box)),box,plot_fit_line=True,s=50,cmap='jet')
            # t_bins = x_bins[:-1]
            t_bins = x_bins[1:]
            plt.scatter(t_bins, box, color=humid_arid_color[humid_arid], alpha=0.5)
            # plt.boxplot()
            sns.regplot(t_bins, box, scatter=False, color=humid_arid_color[humid_arid])
            # plt.xticks(range(len(box))[::5],t_bins[::5],)
            plt.xlabel(x_var)
            plt.ylabel(f'r: {y_var} vs {recovery_var}')
            # plt.savefig(outf)
            # plt.close()
            # plt.figure()
            # plt.plot(t_bins,box_len)
        # plt.ylim(0.05,0.35)
        plt.show()

    def dormant_matrix(self):
        # y1_var = 'dormant_SWE_Terra'
        # y1_var = 'dormant_PRE'
        y1_var = 'dormant_TMP'
        # y1_var = 'dormant_SWE_Terra'
        recovery_var = 'recovery_time'
        title = f'{y1_var}<<-->>{recovery_var}'
        t_var = 'MA_TMP'
        p_var = 'MA_PRE'
        df = Global_vars().load_df()
        df = df[df['winter_mark'] == 1]
        bin_number = 40
        mat_bins = np.linspace(-10, 30, bin_number)
        mat_bins = [round(i, 2) for i in mat_bins]
        map_bins = np.linspace(0, 200, bin_number)
        map_bins = [round(i, 2) for i in map_bins]
        matrix = []
        for i in tqdm(range(len(mat_bins))):
            if i + 1 >= len(mat_bins):
                continue
            df_t = df[df[t_var] > mat_bins[i]]
            df_t = df_t[df_t[t_var] < mat_bins[i + 1]]
            temp = []
            for j in range(len(map_bins)):
                if j + 1 >= len(map_bins):
                    continue
                df_p = df_t[df_t[p_var] > map_bins[j]]
                df_p = df_p[df_p[p_var] < map_bins[j + 1]]
                recovery = df_p[recovery_var].to_list()
                y1_val = df_p[y1_var].to_list()
                # print(y1_val)
                # exit()
                try:
                    # r,p = stats.pearsonr(y1_val,recovery)
                    r, p = T.nan_correlation(y1_val, recovery)
                    # if p < 0.05:
                    #     r = np.nan
                except Exception as e:
                    print(e)
                    r = np.nan
                temp.append(r)
                # temp.append(np.nanmean(recovery))
            matrix.append(temp)
        matrix = np.array(matrix)
        plt.imshow(matrix, vmin=-0.5, vmax=0.5, cmap='RdBu_r')
        plt.xlabel(p_var)
        plt.ylabel(t_var)
        plt.xticks(range(len(matrix))[1:][::10], map_bins[1:][::10])
        plt.yticks(range(len(matrix[0]))[1:][::10], mat_bins[1:][::10])
        plt.title(title)
        plt.colorbar()
        plt.show()

    def spei_and_sos(self):
        outtifdir = self.this_class_tif + 'drought_and_sos/'
        T.mk_dir(outtifdir)
        sos_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + '/transform_early_peak_late_dormant_period_annual/early_start.npy'
        spei_dir_all = data_root + 'SPEI/per_pix/'
        for spei_dir in T.listdir(spei_dir_all):
            spei_dic = T.load_npy_dir(spei_dir_all + spei_dir, condition='')
            sos_dic = T.load_npy(sos_f)
            gs = list(range(5, 10))
            spatial_dic = {}
            for pix in tqdm(sos_dic, desc=spei_dir):
                sos = sos_dic[pix]
                if len(sos) == 0:
                    continue
                if not pix in spei_dic:
                    continue
                sos_anomaly = self.__cal_anomaly(sos)
                spei = spei_dic[pix]
                spei_reshape = np.reshape(spei, (-1, 12))
                spei_annual_gs = []
                for y in spei_reshape:
                    # y_gs = T.pick_vals_from_1darray(y,gs)
                    y_gs_min_index = T.pick_min_indx_from_1darray(y, gs)
                    y_gs_min = y[y_gs_min_index]
                    # y_gs_mean = np.mean(y_gs)
                    # spei_annual_gs.append(y_gs_mean)
                    spei_annual_gs.append(y_gs_min)
                spei_annual_gs_pre = spei_annual_gs[:-1]
                sos_current = sos_anomaly[1:]
                try:
                    r, p = stats.pearsonr(spei_annual_gs_pre, sos_current)
                except:
                    r, p = np.nan, np.nan
                spatial_dic[pix] = r
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            DIC_and_TIF().arr_to_tif(arr, outtifdir + spei_dir + '.tif')
            # plt.imshow(arr)
            # plt.colorbar()
            # plt.show()

    def plot_drought_and_sos(self):
        fdir = self.this_class_tif + 'drought_and_sos/'
        flag = 0
        for tif in T.listdir(fdir):
            flag += 1
            arr = to_raster.raster2array(fdir + tif)[0]
            T.mask_999999_arr(arr)
            plt.subplot(3, 4, flag)
            plt.imshow(arr, vmin=-0.5, vmax=0.5)
            DIC_and_TIF().plot_back_ground_arr_north_sphere()
            plt.xticks([], [])
            plt.yticks([], [])
        plt.tight_layout()
        plt.show()

        pass

    def __cal_anomaly(self, val, method='anomaly'):
        mean = np.mean(val)
        std = np.std(val)
        if method == 'anomaly':
            anomaly = []
            for i in val:
                ii = i - mean
                anomaly.append(ii)
        elif method == 'std_anomaly':
            anomaly = []
            for i in val:
                ii = (i - mean) / std
                anomaly.append(ii)
        else:
            raise UserWarning
        return anomaly

        pass

    def drought_non_drought_sos(self):
        sos_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + '/transform_early_peak_late_dormant_period_annual/early_start.npy'
        sos_dic = T.load_npy(sos_f)
        spei_dir_all = data_root + 'SPEI/per_pix/'
        # for spei_dir in T.listdir(spei_dir_all):
        #     spei_dic = T.load_npy_dir(spei_dir_all + spei_dir, condition='')
        #     sos_dic = T.load_npy(sos_f)
        df = Global_vars().load_df_all()
        '''
        early
        mid
        late'''
        timing = 'early'
        # timing = 'mid'
        # timing = 'late'
        df = df[df['timing'] == timing]
        # df = df[df['humid_arid']=='arid']
        # df = df[df['humid_arid']=='humid']
        drought_sos_pdf = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            event_start_index = row['event_start_index']
            drought_year = event_start_index // 12
            sos = sos_dic[pix]
            sos_anomaly = self.__cal_anomaly(sos)
            sos_second_year = sos_anomaly[drought_year + 1]
            drought_sos_pdf.append(sos_second_year)

        drought_spatial_dic = DIC_and_TIF().void_spatial_dic()
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            event_start_index = row['event_start_index']
            drought_year = event_start_index // 12
            drought_spatial_dic[pix].append(drought_year)

        non_drought_year_dic = {}
        for pix in drought_spatial_dic:
            drought_year_list = drought_spatial_dic[pix]
            if len(drought_year_list) == 0:
                continue
            drought_year_list = list(set(drought_year_list))
            all_year_list = list(range(2015 - 1982 + 1))
            non_drought_year_list = []
            for i in all_year_list:
                if i in drought_year_list:
                    continue
                non_drought_year_list.append(i)
            non_drought_year_dic[pix] = non_drought_year_list

        non_drought_sos_pdf = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            non_drought_year_list = non_drought_year_dic[pix]
            # non_drought_sos_list = []
            sos = sos_dic[pix]
            sos_anomaly = self.__cal_anomaly(sos)
            for y in non_drought_year_list:
                if y + 1 >= len(sos_anomaly):
                    continue
                sos_second_year = sos_anomaly[y + 1]
                # non_drought_sos_list.append(sos_second_year)
                # non_drought_sos_mean = np.mean(non_drought_sos_list)
                non_drought_sos_pdf.append(sos_second_year)
        plt.hist(drought_sos_pdf, range=(-60, 60), bins=180, color='r', density=False, alpha=0.4,
                 weights=np.ones(len(drought_sos_pdf)) / len(drought_sos_pdf), label='drought')
        # plt.figure()
        plt.hist(non_drought_sos_pdf, range=(-60, 60), bins=180, color='b', density=False, alpha=0.4,
                 weights=np.ones(len(non_drought_sos_pdf)) / len(non_drought_sos_pdf), label='non-drought')
        plt.title(timing)
        plt.legend()
        plt.show()

    def df_sos_partial_correlation(self):
        outdir = self.this_class_arr + 'df_sos_partial_correlation/'
        T.mk_dir(outdir)
        df = Global_vars().load_df()
        pre_dir = data_root + 'PRE/per_pix_clean_anomaly/'
        tmp_dir = data_root + 'TMP/per_pix_clean_anomaly/'
        sos_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + 'transform_early_peak_late_dormant_period_annual/early_start.npy'
        early_start_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + 'transform_early_peak_late_dormant_period_annual/early_start_mon.npy'
        early_end_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + 'transform_early_peak_late_dormant_period_annual/early_end_mon.npy'
        spei_dir = data_root + 'SPEI/per_pix/spei03/'
        dir_dic = {
            'pre_dir': pre_dir,
            'tmp_dir': tmp_dir,
            'sos_f': sos_f,
            'early_start_f': early_start_f,
            'early_end_f': early_end_f,
            'spei_dir': spei_dir,
        }

        data_dic = {}
        for var_name in dir_dic:
            fdir = dir_dic[var_name]
            if os.path.isfile(fdir):
                dic = T.load_npy(fdir)
            else:
                dic = T.load_npy_dir(fdir)
            data_dic[var_name] = dic

        # for spei_dir in T.listdir(spei_dir_all):
        #     spei_dic = T.load_npy_dir(spei_dir_all + spei_dir, condition='')
        #     sos_dic = T.load_npy(sos_f)
        pre_mean_list = []
        tmp_mean_list = []
        spei_mean_list = []
        pix_list = []
        sos_anomaly_list = []
        sos_std_anomaly_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row.pix
            event_start_index = row.event_start_index
            first_year = event_start_index // 12
            next_year = event_start_index // 12 + 1
            spring_start_mon = data_dic['early_start_f'][pix]
            spring_end_mon = data_dic['early_end_f'][pix]

            first_year_spring_to_winter_index = list(range(spring_start_mon[first_year], 13))
            first_year_spring_to_winter_index = np.array(first_year_spring_to_winter_index)
            first_year_spring_to_winter_index = first_year_spring_to_winter_index - 1
            next_year_winter_to_spring_index = list(range(0, spring_start_mon[next_year] - 1))
            next_year_winter_to_spring_index = np.array(next_year_winter_to_spring_index)

            next_year_spring_index = list(range(spring_start_mon[next_year], spring_end_mon[next_year] + 1))
            next_year_spring_index = np.array(next_year_spring_index)
            next_year_spring_index = next_year_spring_index - 1

            # print(spring_start_mon[first_year])
            # print(spring_start_mon[next_year])
            # print('first_year_spring_to_winter_index',first_year_spring_to_winter_index)
            # print('next_year_winter_to_spring_index',next_year_winter_to_spring_index)
            # print('next_year_spring_index',next_year_spring_index)
            # print('---'*8)

            sos_day = data_dic['sos_f'][pix]
            spei = data_dic['spei_dir'][pix]
            pre = data_dic['pre_dir'][pix]
            tmp = data_dic['tmp_dir'][pix]

            sos_day_anomaly = self.__cal_anomaly(sos_day, method='anomaly')
            sos_day_std_anomaly = self.__cal_anomaly(sos_day, method='std_anomaly')
            spei_reshape = np.reshape(spei, (-1, 12))
            pre_reshape = np.reshape(pre, (-1, 12))
            tmp_reshape = np.reshape(tmp, (-1, 12))

            spei_first_year = spei_reshape[first_year]
            spei_next_year = spei_reshape[next_year]
            pre_next_year = pre_reshape[next_year]
            tmp_next_year = tmp_reshape[next_year]

            spei_first_year_spring_to_winter = T.pick_vals_from_1darray(spei_first_year,
                                                                        first_year_spring_to_winter_index)
            spei_next_year_winter_to_spring = T.pick_vals_from_1darray(spei_next_year, next_year_winter_to_spring_index)
            # spei_next_year_spring = T.pick_vals_from_1darray(spei_next_year,next_year_spring_index)
            spei_picked = []
            for ii in [spei_first_year_spring_to_winter, spei_next_year_winter_to_spring]:
                for iii in ii:
                    spei_picked.append(iii)
            pre_next_year_spring = T.pick_vals_from_1darray(pre_next_year, next_year_spring_index)
            tmp_next_year_spring = T.pick_vals_from_1darray(tmp_next_year, next_year_spring_index)

            pre_mean = np.mean(pre_next_year_spring)
            tmp_mean = np.mean(tmp_next_year_spring)
            spei_mean = np.mean(spei_picked)

            pre_mean_list.append(pre_mean)
            tmp_mean_list.append(tmp_mean)
            spei_mean_list.append(spei_mean)
            pix_list.append(pix)
            sos_std_anomaly_list.append(sos_day_std_anomaly[next_year])
            sos_anomaly_list.append(sos_day_anomaly[next_year])
        df_partial_corr = pd.DataFrame()
        df_partial_corr['pix'] = pix_list
        df_partial_corr['pre'] = pre_mean_list
        df_partial_corr['tmp'] = tmp_mean_list
        df_partial_corr['spei'] = spei_mean_list
        df_partial_corr['sos_std_anomaly'] = sos_std_anomaly_list
        df_partial_corr['sos_anomaly'] = sos_anomaly_list

        T.save_df(df_partial_corr, outdir + 'dataframe')
        self.__df_to_excel(df_partial_corr, outdir + 'dataframe')

    def partial_correlation(self):
        dff = self.this_class_arr + 'df_sos_partial_correlation/dataframe'
        df = T.load_df(dff)
        df = df[df['spei'] > -5]
        pre = df['pre']
        tmp = df['tmp']
        spei = df['spei']
        # corr = df.partial_corr()
        pcorr = pg.pcorr(df)
        corr = df.corr()
        # pcorr = df.pcorr()
        rcorr = df.rcorr()
        print(corr)
        print(pcorr)
        print(rcorr)
        # KDE_plot().plot_scatter(pre,tmp)
        # KDE_plot().plot_scatter(pre,spei)
        # plt.show()
        # r,p = self.partial_corr(df,x='spei',y='sos_std_anomaly',cov=['tmp','pre',])
        # print(r,p)
        pass

    def partial_corr(self, df, x, y, cov):
        df = pd.DataFrame(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        # print(df)
        df = df.dropna()
        print(df)
        # try:
        # print(x)
        # print(y)
        stats_result = pg.partial_corr(data=df, x=x, y=y, covar=cov, method='pearson').round(3)
        print(stats_result)
        r = float(stats_result['r'])
        p = float(stats_result['p-val'])
        return r, p
        # except Exception as e:
        #     # print(df)
        #     print(e)
        #     print(x)
        #     print(y)
        #     print(cov)
        #     print(df[cov])
        #     df_select = df[cov]
        #     df_select[x] = df[x]
        #     df_select[y] = df[y]
        #     out_excel = r'G:\project05_redo\results\arr\Partial_Correlation\test.xlsx'
        #     df_select.to_excel(out_excel)
        #     pause()
        #     return np.nan,np.nan

    def swe_sos(self):
        df = Global_vars().load_df()
        x = []
        y = []
        humid_arid_list = ['humid', 'arid']
        df = df[df['humid_arid'] == humid_arid_list[1]]
        df = df[df['winter_mark'] == 1]

        for i, row in tqdm(df.iterrows(), total=len(df)):
            # sos = row['current_sos_std_anomaly']
            sos = row['current_sos_std_anomaly']
            swe = row['dormant_SWE_Terra']
            x.append(sos)
            y.append(swe)
        KDE_plot().plot_scatter(x, y)
        plt.show()

    def spatial_autocorrelation(self):
        df = Global_vars().load_df()
        var_name = 'spei03'
        # var_ = f'{var_name}_severity'
        df = df[df['product'] == var_name]
        # df = df[df['recovery_time'] < 15]
        lon = df['lon'].tolist()
        lat = df['lat'].tolist()
        recovery = df['recovery_time'].tolist()
        data = {'lon': lon, 'recovery': recovery, 'lat': lat}
        # var_xy = 'lon'
        var_xy = 'lat'
        df_plot = pd.DataFrame(data)
        if var_xy == 'lon':
            bins = np.arange(-180, 190, step=10)
        else:
            bins = np.arange(30, 100, step=5)
        # print(bins)
        # exit()
        df_group, bins_list_str = T.df_bin(df_plot, var_xy, bins)
        y_list = []
        err_list = []
        for name, df_group_i in df_group:
            vals = df_group_i['recovery'].tolist()
            mean = np.nanmean(vals)
            err, _, _ = self.uncertainty_err(vals)
            # x_list.append(name)
            y_list.append(mean)
            err_list.append(err)

        plt.figure()
        Plot().plot_line_with_error_bar(bins_list_str, y_list, err_list)
        plt.plot(bins_list_str, y_list, '-o')
        plt.ylabel('Drought recovery')
        # plt.xticks(rotation=90)
        if var_xy == 'lon':
            plt.xlabel('Longitude')
        else:
            plt.xlabel('Latitude')
        plt.grid(True)
        # plt.title(f'{var_xy} spatial autocorrelation')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        # plt.plot(df_group)
        # plt.show()
        # exit()
        # r,p = stats.pearsonr(drought_year_sos_anomaly,recovery)
        # print(r,p)
        # KDE_plot().plot_scatter(drought_year_sos_anomaly,recovery)
        # plt.ylabel('recovery time (months)')
        # plt.xlabel('NDVI_SR_sos_to_drought_start')

    def latitude_recovery_output_excel(self):
        outdir = join(self.this_class_arr, 'latitude_recovery_output_excel')
        T.mk_dir(outdir)
        outf = join(outdir, 'latitude_recovery_output_excel.xlsx')
        df = Global_vars().load_df()
        var_name = 'spei03'
        # var_ = f'{var_name}_severity'
        df = df[df['product'] == var_name]
        # df = df[df['recovery_time'] < 15]
        lon = df['lon'].tolist()
        lat = df['lat'].tolist()
        recovery = df['recovery_time'].tolist()
        data = {'lon': lon, 'recovery': recovery, 'lat': lat}
        # var_xy = 'lon'
        var_xy = 'lat'
        df_plot = pd.DataFrame(data)
        if var_xy == 'lon':
            bins = np.arange(-180, 190, step=10)
        else:
            bins = np.arange(30, 100, step=5)
        # print(bins)
        # exit()
        df_group, bins_list_str = T.df_bin(df_plot, var_xy, bins)
        y_list = []
        err_list = []
        for name, df_group_i in df_group:
            vals = df_group_i['recovery'].tolist()
            mean = np.nanmean(vals)
            err, _, _ = self.uncertainty_err(vals)
            # x_list.append(name)
            y_list.append(mean)
            err_list.append(err)
        df_out = pd.DataFrame({'lat': bins_list_str, 'recovery': y_list, 'err': err_list})
        df_out.to_excel(outf)
        plt.figure()
        Plot().plot_line_with_error_bar(bins_list_str, y_list, err_list)
        plt.plot(bins_list_str, y_list, '-o')
        plt.ylabel('Drought recovery')
        # plt.xticks(rotation=90)
        if var_xy == 'lon':
            plt.xlabel('Longitude')
        else:
            plt.xlabel('Latitude')
        plt.grid(True)
        # plt.title(f'{var_xy} spatial autocorrelation')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        # plt.plot(df_group)
        # plt.show()
        # exit()
        # r,p = stats.pearsonr(drought_year_sos_anomaly,recovery)
        # print(r,p)
        # KDE_plot().plot_scatter(drought_year_sos_anomaly,recovery)
        # plt.ylabel('recovery time (months)')
        # plt.xlabel('NDVI_SR_sos_to_drought_start')

        pass

    def difference_between_left_winter_and_drought_recovery(self):
        # 发生在晚生长季的干旱时，植被恢复的用时，和实际前一个生长季剩余的时间的差值是多少
        df = Global_vars().load_df()
        # df = df[df['product'] == 'spei03']
        df = df[df['recovery_time'] < 20]
        df = df[df['winter_mark_new'] == 1]
        df = df[df['timing'] == 'late']
        df = df[df['recovery_start_gs'] == 'first']
        difference_list = []
        time_remain_list = []
        recovery_time_list = []
        for i, row in df.iterrows():
            pix = row['pix']
            drought_start = row['event_start_index']
            late_end_mon = row['late_end_mon']
            drought_start_mon = drought_start % 12 + 1
            recovery_time = row['recovery_time']
            time_remain = late_end_mon - drought_start_mon
            if time_remain >= 3:
                continue
            difference = recovery_time - time_remain
            difference_list.append(difference)
            time_remain_list.append(time_remain)
            recovery_time_list.append(recovery_time)
        mean_diff = np.nanmean(difference_list)
        err_diff, up, bottom = T.uncertainty_err(difference_list)
        std_diff = np.nanstd(difference_list)

        mean_time_remain = np.nanmean(time_remain_list)
        err_time_remain, up, bottom = T.uncertainty_err(time_remain_list)
        std_time_remain = np.nanstd(time_remain_list)

        mean_recovery_time = np.nanmean(recovery_time_list)
        err_recovery_time, up, bottom = T.uncertainty_err(recovery_time_list)
        std_recovery_time = np.nanstd(recovery_time_list)

        print(f'mean_diff: {mean_diff}')
        print(f'err_diff: {err_diff}')
        print(f'std_diff: {std_diff}')
        print(f'mean_time_remain: {mean_time_remain}')
        print(f'err_time_remain: {err_time_remain}')
        print(f'std_time_remain: {std_time_remain}')
        print(f'mean_recovery_time: {mean_recovery_time}')
        print(f'err_recovery_time: {err_recovery_time}')
        print(f'std_recovery_time: {std_recovery_time}')

    def difference_between_left_winter_and_drought_recovery_lc(self):
        # 发生在晚生长季的干旱时，植被恢复的用时，和实际前一个生长季剩余的时间的差值是多少
        df = Global_vars().load_df()
        df = df[df['product'] == 'spei03']
        df = df[df['recovery_time'] < 10]
        df = df[df['winter_mark_new'] == 1]
        df = df[df['timing'] == 'late']
        df = df[df['recovery_start_gs'] == 'first']
        lc_list = ['Grasslands', 'Shrublands', 'deciduous', 'evergreen']
        err_list = []
        mean_list = []
        for lc in lc_list:
            df_lc = df[df['lc'] == lc]
            difference_list = []
            for i, row in df_lc.iterrows():
                pix = row['pix']
                drought_start = row['event_start_index']
                late_end_mon = row['late_end_mon']
                drought_start_mon = drought_start % 12 + 1
                recovery_time = row['recovery_time']
                time_remain = late_end_mon - drought_start_mon
                if time_remain >= 3:
                    continue
                difference = recovery_time - time_remain
                difference_list.append(difference)
            mean = np.nanmean(difference_list)
            err, up, bottom = T.uncertainty_err(difference_list)
            print(up, bottom)
            mean_list.append(mean)
            err_list.append(err)
            print(lc, mean, err)
        plt.figure()
        plt.bar(lc_list, mean_list, yerr=err_list, capsize=5)
        plt.ylabel('Drought_recovery - time_remaining')
        plt.xlabel('LC')
        # plt.grid(True)
        plt.show()


class Tif:

    def __init__(self):
        self.this_class_tif = result_root_this_script + 'tif/Tif/'
        Tools().mk_dir(self.this_class_tif, force=True)
        pass

    def load_df(self):
        dff = Global_vars().dff
        df = T.load_df(dff)
        # df = df[df['lat']>30]
        # df = df[df['CV_PRE']<500]
        # df = df[df['current_sos_anomaly'] < 20]
        # df = df[df['current_sos_anomaly'] > -20]
        # df = df.dropna(subset=['timing'])
        return df

    def run(self):

        # self.plot_recovery()
        # self.plot_recovery_spei_1_12()
        # self.gen_recovery_spei_1_12_tif()
        # self.compose_gen_recovery_spei_1_12_tif()
        # self.phenology_doy()
        # self.NDVI_phenology_doy()
        # self.NDVI_gs_length()
        # self.CSIF_phenology_doy()
        # self.Koppen_GLC2000_tif()
        # self.Aridity_index_tif()
        # self.plot_drought_events_number()
        self.SPEI_NDVI_correlation_timing()

    def plot_recovery(self):
        outdir = self.this_class_tif + 'recovery_time/'
        T.mk_dir(outdir)
        df = Global_vars().load_df()
        # print(len(df))
        # exit()
        # df = df[df['post_3_month_greater_than_threshold']==1]
        # print(len(df))
        # exit()
        spatial_dic = DIC_and_TIF().void_spatial_dic()
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            val = row.recovery_time
            spatial_dic[pix].append(val)
        arr = DIC_and_TIF().pix_dic_to_spatial_arr_mean(spatial_dic)
        outtif = outdir + 'recovery1.tif'
        DIC_and_TIF().arr_to_tif(arr, outtif)
        # plt.imshow(arr,vmin=1,vmax=4)
        # plt.colorbar()
        # plt.show()

    def plot_recovery_spei_1_12(self):
        df = Global_vars().load_df()
        for scale in range(1, 13):
            plt.subplot(6, 2, scale)
            df_spei = df[df['product'] == 'spei{:02d}'.format(scale)]
            spatial_dic = DIC_and_TIF().void_spatial_dic()
            for i, row in tqdm(df_spei.iterrows(), total=len(df_spei)):
                pix = row.pix
                val = row.recovery_time
                spatial_dic[pix].append(val)
            arr = DIC_and_TIF().pix_dic_to_spatial_arr_mean(spatial_dic)
            # rasterized_world_tif = '/Volumes/Ugreen_4T_25/project05_redo/conf/land.tif'
            rasterized_world_tif = T.path_join(this_root, 'conf', 'land.tif')
            DIC_and_TIF().plot_back_ground_arr_north_sphere(rasterized_world_tif)
            arr = arr[:120]
            plt.imshow(arr, vmin=0, vmax=8)
            plt.xticks([], [])
            plt.title('spei{:02d}'.format(scale))
        plt.tight_layout()
        plt.show()

    def plot_drought_events_number_each_SPEI_scale(self):
        df = Global_vars().load_df()
        for scale in range(1, 13):
            plt.subplot(6, 2, scale)
            df_spei = df[df['product'] == 'spei{:02d}'.format(scale)]
            spatial_dic = DIC_and_TIF().void_spatial_dic()
            for i, row in tqdm(df_spei.iterrows(), total=len(df_spei)):
                pix = row.pix
                val = row.recovery_time
                spatial_dic[pix].append(val)
            spatial_dic_events_number = {}
            for pix in spatial_dic:
                spatial_dic_events_number[pix] = len(spatial_dic[pix])
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_events_number)
            arr[arr == 0] = np.nan
            # rasterized_world_tif = '/Volumes/Ugreen_4T_25/project05_redo/conf/land.tif'
            rasterized_world_tif = T.path_join(this_root, 'conf', 'land.tif')
            DIC_and_TIF().plot_back_ground_arr_north_sphere(rasterized_world_tif, aspect='auto')
            arr = arr[:120]
            plt.imshow(arr, vmin=1, vmax=3, aspect='auto')
            plt.colorbar()
            plt.xticks([], [])
            plt.title('spei{:02d}'.format(scale))
        # plt.tight_layout()
        plt.show()

    def plot_drought_events_number(self):
        df = Global_vars().load_df()
        spatial_dic = DIC_and_TIF().void_spatial_dic()
        for i, row in tqdm(df.iterrows()):
            pix = row.pix
            val = row.recovery_time
            spatial_dic[pix].append(val)
        spatial_dic_events_number = {}
        for pix in spatial_dic:
            spatial_dic_events_number[pix] = len(spatial_dic[pix])
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_events_number)
        arr[arr == 0] = np.nan
        # rasterized_world_tif = '/Volumes/Ugreen_4T_25/project05_redo/conf/land.tif'
        rasterized_world_tif = T.path_join(this_root, 'conf', 'land.tif')
        DIC_and_TIF().plot_back_ground_arr_north_sphere(rasterized_world_tif, aspect='auto')
        arr = arr[:120]
        plt.imshow(arr, vmin=1, vmax=20, aspect='auto')
        plt.colorbar()
        plt.xticks([], [])
        plt.tight_layout()
        plt.show()

    def gen_recovery_spei_1_12_tif(self):
        outdir = T.path_join(self.this_class_tif, 'gen_recovery_spei_1_12_tif')
        T.mk_dir(outdir)
        df = self.load_df()
        for scale in range(1, 13):
            df_spei = df[df['product'] == 'spei{:02d}'.format(scale)]
            spatial_dic = DIC_and_TIF().void_spatial_dic()
            for i, row in tqdm(df_spei.iterrows(), total=len(df_spei)):
                pix = row.pix
                val = row.recovery_time
                spatial_dic[pix].append(val)
            arr = DIC_and_TIF().pix_dic_to_spatial_arr_mean(spatial_dic)
            fpath = T.path_join(outdir, 'spei{:02d}.tif'.format(scale))
            DIC_and_TIF().arr_to_tif(arr, fpath)

    def compose_gen_recovery_spei_1_12_tif(self):
        fdir = T.path_join(self.this_class_tif, 'gen_recovery_spei_1_12_tif')
        outdir = T.path_join(self.this_class_tif, 'compose_gen_recovery_spei_1_12_tif')
        T.mk_dir(outdir)

        for scale in [3, 6, 9, 12]:
            outf = T.path_join(outdir, 'spei{}.tif'.format(scale))
            spei_flist = []
            for i in range(scale - 2, scale + 1):
                fpath = 'spei{:02d}.tif'.format(i)
                spei_flist.append(T.path_join(fdir, fpath))
            # print(spei_flist)
            Pre_Process().compose_tif_list(spei_flist, outf)

    def NDVI_phenology_doy(self):
        fdir = T.path_join(Main_flow_Early_Peak_Late_Dormant().this_class_arr,
                           'transform_early_peak_late_dormant_period_annual')
        outdir = T.path_join(self.this_class_tif, 'phenology_doy')
        T.mk_dir(outdir)
        all_dic = {
            'early_start_mon': T.path_join(fdir, 'early_start_mon.npy'),
            'early_start_doy': T.path_join(fdir, 'early_start.npy'),
            'late_end_mon': T.path_join(fdir, 'late_end.npy'),
            'late_end_doy': T.path_join(fdir, 'late_end_mon.npy'),
            # 'gs_length_mon' : T.path_join(fdir,'gs_mons.npy'),
            # 'gs_length_doy' : T.path_join(fdir,'gs_mons.npy')
        }
        for key in all_dic:
            f = all_dic[key]
            dic = T.load_npy(f)
            spatial_dic = {}
            for i in dic:
                vals = dic[i]
                # print(vals)
                mean = np.nanmean(vals)
                spatial_dic[i] = mean
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            outf = T.path_join(outdir, key + '.tif')
            DIC_and_TIF().arr_to_tif(arr, outf)

    def NDVI_gs_length(self):
        fdir = T.path_join(Main_flow_Early_Peak_Late_Dormant().this_class_arr,
                           'transform_early_peak_late_dormant_period_annual')
        outdir = T.path_join(self.this_class_tif, 'phenology_doy')
        T.mk_dir(outdir)
        all_dic = {
            # 'gs_length_mon' : T.path_join(fdir,'gs_mons.npy'),
            'gs_length_doy': T.path_join(fdir, 'dormant_length.npy')
        }
        # print(outdir)
        # exit()
        for key in all_dic:
            f = all_dic[key]
            dic = T.load_npy(f)
            spatial_dic = {}
            for i in dic:
                vals = dic[i]
                mean = np.nanmean(vals)
                gs_length = 365 - mean
                spatial_dic[i] = gs_length
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            outf = T.path_join(outdir, key + '.tif')
            DIC_and_TIF().arr_to_tif(arr, outf)

    def CSIF_phenology_doy(self):
        fdir = T.path_join(this_root, 'Main_flow_results_sif', 'arr', 'Main_flow_Early_Peak_Late_Dormant',
                           'transform_early_peak_late_dormant_period_annual')
        # fdir = T.path_join(Main_flow_Early_Peak_Late_Dormant().this_class_arr,'transform_early_peak_late_dormant_period_annual')
        outdir = T.path_join(self.this_class_tif, 'CSIF_phenology_doy')
        T.mk_dir(outdir)
        all_dic = {
            'early_start_mon': T.path_join(fdir, 'early_start_mon.npy'),
            'early_start_doy': T.path_join(fdir, 'early_start.npy'),
            'late_end_mon': T.path_join(fdir, 'late_end.npy'),
            'late_end_doy': T.path_join(fdir, 'late_end_mon.npy'),
            # 'gs_length_mon' : T.path_join(fdir,'gs_mons.npy'),
            # 'gs_length_doy' : T.path_join(fdir,'gs_mons.npy')
        }
        lat_set = 65
        D = DIC_and_TIF()
        for key in all_dic:
            f = all_dic[key]
            dic = T.load_npy(f)
            spatial_dic = {}
            for i in tqdm(dic):
                # r,c = i
                lon, lat = D.pix_to_lon_lat(i)
                if lat > lat_set:
                    # print(lat,i)
                    continue
                vals = dic[i]
                # print(vals)
                mean = np.nanmean(vals)
                spatial_dic[i] = mean
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            # plt.imshow(arr)
            # plt.show()
            outf = T.path_join(outdir, key + '.tif')
            DIC_and_TIF().arr_to_tif(arr, outf)

        pass

    def Koppen_GLC2000_tif(self):
        outdir = self.this_class_tif + 'Koppen_GLC2000_tif/'
        T.mk_dir(outdir)
        lc_name_dic = {'Grasslands': 3, 'Shrublands': 2, 'evergreen': 0, 'deciduous': 1}
        for lc_ in lc_name_dic:
            outf = outdir + 'kp_lc_{}.tif'.format(lc_)
            outf_text = outdir + 'label_{}.txt'.format(lc_)
            df = self.load_df()
            df = df[df['lc'] == lc_]
            kp_name_dic = {'B': 1, 'E': 6, 'Df': 4, 'Dsw': 5, 'Cf': 2, 'Csw': 3}
            label_dic = {}
            spatial_dic = {}
            for i, row in tqdm(df.iterrows(), total=len(df), desc=lc_):
                pix = row.pix
                lc = row.lc
                kp = row.kp
                if not lc in lc_name_dic:
                    continue
                if not kp in kp_name_dic:
                    continue
                val = lc_name_dic[lc] * len(kp_name_dic) + kp_name_dic[kp]
                label = lc + '_' + kp
                label_dic[label] = val
                spatial_dic[pix] = val
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            DIC_and_TIF().arr_to_tif(arr, outf)
            label_list = [i for i in label_dic]
            label_list.sort()
            fw = open(outf_text, 'w')
            for i in label_list:
                val = label_dic[i]
                content = i + '\t' + str(int(val)) + '\n'
                fw.write(content)
            fw.close()
            pass

    def Aridity_index_tif(self):
        outdir = self.this_class_tif + 'Aridity_index_tif/'
        T.mk_dir(outdir)
        aridity_index_dict = Dataframe().P_PET_ratio(Dataframe().P_PET_fdir)
        DIC_and_TIF().pix_dic_to_tif(aridity_index_dict, outdir + 'aridity_index.tif')

    def SPEI_NDVI_correlation_timing(self):
        outdir = join(self.this_class_tif, 'SPEI_NDVI_correlation_timing')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        ndvi_dir = join(data_root, 'NDVI', 'per_pix_clean_anomaly_smooth_detrend')
        spei_dir = join(data_root, 'SPEI/per_pix_clean_smooth_detrend/spei03')

        ndvi_dict = T.load_npy_dir(ndvi_dir)
        spei_dict = T.load_npy_dir(spei_dir)

        growing_season_f = Main_flow_Early_Peak_Late_Dormant().this_class_arr + \
                           'early_peak_late_dormant_period_long_term/early_peak_late_dormant_period_long_term.npy'
        gs_dict = T.load_npy(growing_season_f)
        early_range_dict = {}
        peak_range_dict = {}
        late_range_dict = {}
        for pix in tqdm(gs_dict):
            vals = gs_dict[pix]
            early_start = vals['early_start']
            early_end = vals['early_end']
            late_start = vals['late_start']
            late_end = vals['late_end']
            early_range = list(range(early_start, early_end + 1))
            peak_range = list(range(early_end, late_start + 1))
            late_range = list(range(late_start, late_end + 1))
            early_range_dict[pix] = early_range
            peak_range_dict[pix] = peak_range
            late_range_dict[pix] = late_range
        timing_dict = {'early': early_range_dict, 'peak': peak_range_dict, 'late': late_range_dict}
        for timing in timing_dict:
            gs_dict = timing_dict[timing]
            corr_dict = {}
            for pix in tqdm(gs_dict, desc=timing):
                gs = gs_dict[pix]
                if not pix in ndvi_dict:
                    continue
                if not pix in spei_dict:
                    continue
                ndvi = ndvi_dict[pix]
                spei = spei_dict[pix]
                ndvi_gs = T.monthly_vals_to_annual_val(ndvi, gs)
                spei_gs = T.monthly_vals_to_annual_val(spei, gs)
                r, p = stats.pearsonr(ndvi_gs, spei_gs)
                corr_dict[pix] = r
            outf = join(outdir, 'corr_{}.tif'.format(timing))
            DIC_and_TIF().pix_dic_to_tif(corr_dict, outf)

        pass


class RF:

    def __init__(self):
        self.this_class_arr = result_root_this_script + 'arr/RF/'
        self.this_class_png = result_root_this_script + 'png/RF/'
        Tools().mk_dir(self.this_class_png, force=True)
        Tools().mk_dir(self.this_class_arr, force=True)
        pass

    def run(self):
        lc_list = Global_vars().lc_list()
        outdir = self.this_class_arr + 'RF_results/'
        T.mk_dir(outdir)
        # for lc in lc_list:
        #     print(lc)
        self.plot_importance_with_winter()
        # self.plot_importance_no_winter()
        _, importances_dic_withwinter, r_X_dic_withwinter, _, _, score_withwinter, _, _ = self.plot_importance_with_winter()
        # _, importances_dic_nowinter, r_X_dic_nowinter, _, _, score_nowinter, _, _ = self.plot_importance_no_winter()
        # with_winter_result_path = outdir + f'{lc}_with_winter'
        # no_winter_result_path = outdir + f'{lc}_no_winter'

        # with_winter_dic = {
        #     'importance':importances_dic_withwinter,
        #     'correlation':r_X_dic_withwinter,
        #     'model_score':score_withwinter,
        # }
        # no_winter_dic = {
        #     'importance': importances_dic_nowinter,
        #     'correlation': r_X_dic_nowinter,
        #     'model_score': score_nowinter,
        # }
        #
        # T.save_npy(with_winter_dic,with_winter_result_path)
        # T.save_npy(no_winter_dic,no_winter_result_path)

        pass

    def __load_df(self):
        df = Global_vars().load_df()
        # df = df[df['product']=='spei04']
        df = df.drop_duplicates(subset=['pix', 'recovery_time', 'recovery_date_range'])

        return df

    def __xlist(self):
        x_list = []

        return x_list
        pass

    # def __x_variables(self):
    #     periods = [
    #         # 'first_drought',
    #         'spell',
    #         # 'second_drought',
    #         'post',
    #     ]
    #     x_var_list = [
    #         'spell_length',
    #     ]
    #     season_list = Global_vars().season_months()
    #     for product in ['precip_anomaly','vpd','sm_anomaly',]:
    #         for period in periods:
    #             x_var = '{}_{}'.format(product,period)
    #             x_var_list.append(x_var)
    #
    #     for product in ['precip_anomaly_mean','vpd_mean','sm_anomaly_mean',]:
    #         for season in season_list:
    #             x_var = '{}_{}_spell'.format(product,season)
    #             x_var_list.append(x_var)
    #     return x_var_list

    def plot_importance_with_winter(self, ):
        outpngdir = self.this_class_png + 'plot_importance/'
        y_var = 'recovery_time'
        T.mk_dir(outpngdir)
        x_var_list = Global_vars().vars_RF_with_winter()
        df = self.__load_df()
        df = df[df['winter_mark'] == 1]
        lat_list = np.linspace(30, 90, 5)
        for i in range(len(lat_list)):
            if i + 1 >= len(lat_list):
                continue
            df_lat = df[df['lat'] > lat_list[i]]
            df_lat = df_lat[df_lat['lat'] < lat_list[i + 1]]
            # df = df[df['product'] == 'cwd']
            # df = df[df['lc']=='Forest']
            # df = df[df['lc']=='Grasslands']
            # df = df[df['MA_PRE']<50]
            X1 = df_lat[x_var_list]
            Y1 = df_lat[y_var]
            new_df = pd.DataFrame()
            new_df[x_var_list] = X1
            new_df[y_var] = Y1
            new_df = new_df.dropna()
            new_df = Global_vars().clean_df(new_df)
            # print(len(new_df))
            # exit()
            # new_df = pd.DataFrame(new_df)
            # new_df = new_df.drop_duplicates(subset=[y_var])
            # print(len(new_df))

            # T.print_head_n(new_df)
            # exit()
            X = new_df[x_var_list]
            Y = new_df[y_var]
            # exit()
            selected_feature = x_var_list
            # selected_feature = self.discard_hierarchical_clustering(df, xvar_list=x_var_list,
            #                                                         yvar=y_var,
            #                                                         isplot=False,
            #                                                         t=1.0)
            outf = outpngdir + '{}.pdf'.format(y_var)
            # print(outf)
            # print(X)
            # print(Y)
            # exit()
            print(lat_list[i], '-->', lat_list[i + 1])
            clf, importances_dic, r_X_dic, mse, r_model, score, Y_test, y_pred = self.random_forest_train(X, Y,
                                                                                                          selected_feature,
                                                                                                          isplot=True,
                                                                                                          is_save_png=False,
                                                                                                          outpngf=outf, )
            # return clf, importances_dic, r_X_dic, mse, r_model,score, Y_test, y_pred

    def plot_importance_no_winter(self, ):
        outpngdir = self.this_class_png + 'plot_importance/'
        y_var = 'recovery_time'
        T.mk_dir(outpngdir)
        x_var_list = Global_vars().vars_RF_no_winter()
        df = self.__load_df()
        df = df[df['winter_mark'] == 0]
        # df_lc = df[df['lc']==lc]
        # print(len(df))
        # T.print_head_n(df)
        # df_lc = df[df['lc']==lc]
        # print(df)
        # exit()
        # print(x_var_list)
        # exit()
        X1 = df[x_var_list]
        Y1 = df[y_var]
        new_df = pd.DataFrame()
        new_df[x_var_list] = X1
        new_df[y_var] = Y1
        # new_df = new_df.dropna()
        new_df = Global_vars().clean_df(new_df)
        # print(len(new_df))
        # exit()
        # new_df = pd.DataFrame(new_df)
        # new_df = new_df.drop_duplicates(subset=[y_var])
        # print(len(new_df))

        # T.print_head_n(new_df)
        # exit()
        X = new_df[x_var_list]
        Y = new_df[y_var]
        # exit()
        selected_feature = x_var_list
        # selected_feature = self.discard_hierarchical_clustering(df, xvar_list=x_var_list,
        #                                                         yvar=y_var,
        #                                                         isplot=False,
        #                                                         t=1.0)
        outf = outpngdir + '{}.pdf'.format(y_var)
        # print(outf)
        # print(X)
        # print(Y)
        # exit()
        clf, importances_dic, r_X_dic, mse, r_model, score, Y_test, y_pred = self.random_forest_train(X, Y,
                                                                                                      selected_feature,
                                                                                                      isplot=True,
                                                                                                      is_save_png=False,
                                                                                                      outpngf=outf, )
        return clf, importances_dic, r_X_dic, mse, r_model, score, Y_test, y_pred

    def discard_hierarchical_clustering(self, df, xvar_list, t=0.0, isplot=False):
        '''
        url:
        https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
        '''
        from collections import defaultdict

        from scipy.stats import spearmanr
        from scipy.cluster import hierarchy

        # print(df)
        # exit()
        df = df.dropna()
        X = df[xvar_list]
        corr = np.array(X.corr())
        corr_linkage = hierarchy.ward(corr)
        cluster_ids = hierarchy.fcluster(corr_linkage, t=t, criterion='distance')
        # cluster_ids = hierarchy.fcluster(corr_linkage, t=t, criterion='inconsistent')
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
        selected_features_indx = [v[0] for v in cluster_id_to_feature_ids.values()]
        selected_features = []
        for i in selected_features_indx:
            selected_features.append(xvar_list[i])

        # print('selected_features:',selected_features)
        if isplot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
            dendro = hierarchy.dendrogram(
                corr_linkage, labels=xvar_list, ax=ax1, leaf_rotation=90
            )
            dendro_idx = np.arange(0, len(dendro['ivl']))
            ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
            ax2.set_xticks(dendro_idx)
            ax2.set_yticks(dendro_idx)
            ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
            ax2.set_yticklabels(dendro['ivl'])
            fig.tight_layout()
        return selected_features

    def random_forest_train(self, X, Y, variable_list, isplot=False, is_save_png=False, outpngf='', ):
        # from sklearn import XGboost
        # from sklearn.ensemble import GradientBoostingRegressor
        if is_save_png and outpngf == '':
            raise UserWarning
        X = X[variable_list]
        r_X_dic = {}
        # for i in X:
        #     x_val = X[i]
        #     y_val = Y
        #     r,p = stats.pearsonr(x_val,y_val)
        #     r_X_dic[i] = (r,p)
        # X = np.array(X)
        # Y = np.array(Y)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
        # print(X_test[0])
        # exit()
        # Y_train = np.array(Y_train)
        # Y_test = np.array(Y_test)
        clf = RandomForestRegressor(n_estimators=100, n_jobs=7)
        # clf = RandomForestClassifier(n_estimators=100, n_jobs=7)
        # clf = GradientBoostingRegressor(n_estimators=100)
        # print(X_train)
        # print(np.all(np.isfinite(X)))
        # print(np.isnan(X.any()))
        # print(np.isfinite(X.all()))
        # exit()
        print('fitting')
        clf.fit(X_train, Y_train)
        print('fitted')

        # importances = clf.feature_importances_
        result = permutation_importance(clf, X_train, Y_train, scoring=None,
                                        n_repeats=10, random_state=42,
                                        n_jobs=7)
        importances = result.importances_mean
        importances_dic = dict(zip(variable_list, importances))
        labels = []
        importance = []
        for key in variable_list:
            labels.append(key)
            importance.append(importances_dic[key])
        # print(result)
        # exit()
        y_pred = clf.predict(X_test)
        # y_pred = clf.predict(X_train)
        r_model = stats.pearsonr(Y_test, y_pred)[0]
        mse = sklearn.metrics.mean_squared_error(Y_test, y_pred)
        # KDE_plot().plot_scatter(Y_test,y_pred,s=100)
        # plt.show()
        score = clf.score(X_test, Y_test)

        #### plot ####
        if isplot:
            print('score', score)
            print(importances)
            print('mse:%s\nr:%s' % (mse, r_model))
            # out_png_dir = self.this_class_png + '/RF_importances/'
            # Tools().mk_dir(out_png_dir)
            # 1 plot spatial
            # plt.figure()
            # plt.imshow(selected_pix_spatial,cmap='gray')

            # 2 plot importance
            plt.figure(figsize=(20, 8))
            plt.subplot(211)
            title_new = 'data_length:{} test_length:{} RMSE:{:0.2f} score:{:0.2f}'.format(len(X), len(X_test), mse,
                                                                                          score)
            plt.title(title_new)
            y_min = min(importances)
            y_max = max(importances)
            offset = (y_max - y_min)
            y_min = y_min - offset * 0.3
            y_max = y_max + offset * 0.3

            plt.ylim(y_min, y_max)
            plt.bar(range(len(importances)), importances, width=0.3)
            # print(variable_list)
            plt.xticks(range(len(importances)), labels, rotation=90)

            # ax = plt.subplot(212)
            # KDE_plot().plot_scatter(Y_test, y_pred, ax=ax, linewidth=0)
            # plt.axis('equal')

            # ax = plt.subplot(313)
            # DIC_and_TIF(Global_vars().tif_template_7200_3600).plot_back_ground_arr()
            # selected_pix_spatial_dic_arr = DIC_and_TIF(Global_vars().tif_template_7200_3600).pix_dic_to_spatial_arr(selected_pix_spatial_dic)
            # plt.imshow(selected_pix_spatial_dic_arr,cmap='gray')
            if is_save_png == True:
                plt.savefig(outpngf, dpi=300)
                plt.close()
            elif is_save_png == False:
                plt.show()
        #### plot ####
        return clf, importances_dic, r_X_dic, mse, r_model, score, Y_test, y_pred

    def plot_result(self):

        pass


class Koppen_Linear_Regression:
    # 全球花点图
    def __init__(self):
        self.this_class_arr = result_root_this_script + 'arr/Koppen_Linear_Regression/'
        # self.this_class_tif = results_root + 'tif/Koppen_Linear_Regression/'
        self.this_class_png = result_root_this_script + 'png/Koppen_Linear_Regression/'

        Tools().mk_dir(self.this_class_arr, force=True)
        # Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        pass

    def run(self):
        result_dir = join(self.this_class_arr, 'result')
        Tools().mk_dir(result_dir)

        png_dir = self.this_class_png + 'Koppen_Linear_Regression/'
        T.mk_dir(png_dir)

        # for xlabel in ['correlation','water_balance','lag','NDVI_pre_6_mean',
        #                'current_sos_anomaly','PRE_pre_6_mean','VPD_pre_6_mean','VPD_lag_mean',
        #                'PRE_lag_mean','post_VPD','post_PRE','TMP_pre_6_mean', 'TMP_lag_mean',
        #                'post_TMP']:
        # for xlabel in ['MA_PRE','MA_TMP',]:
        for xlabel in [
            'current_sos',
            'current_sos_std_anomaly',
            'current_sos_anomaly',
            'drought_year_sos_anomaly',
            'drought_year_sos_std_anomaly',
            'NDVI_pre_6_mean',
            'lag',
            'MA_TMP',
            'correlation',
            'dormant_length',
            'PRE_lag_mean',
            'VPD_lag_mean',
            'dormant_TMP',
            'dormant_SWE_terra',
            'post_PRE',
            'post_VPD',
        ]:
            # for xlabel in ['current_sos_std_anomaly', 'lag', 'post_VPD_3', 'post_PRE_3',
            #                'dormant_SWE', 'dormant_length', 'post_withoutwinter_VPD_3', 'post_withoutwinter_PRE_3'
            #                ]:
            print(xlabel)
            plt.figure(figsize=(12, 3.5))
            plt.suptitle(xlabel)
            # self.cross_koppen_landuse_linear_reg_all(xlabel)
            # try:
            self.cross_koppen_landuse_linear_reg_no_winter(xlabel)
            df_out_no_winter = self.cross_koppen_landuse_linear_reg_no_winter_to_df(xlabel)
            df_out_no_winter_f = join(result_dir, xlabel + '_no_winter.df')
            T.save_df(df_out_no_winter, df_out_no_winter_f)
            # except:
            #     pass
            # try:
            self.cross_koppen_landuse_linear_reg_with_winter(xlabel)
            df_out_with_winter = self.cross_koppen_landuse_linear_reg_with_winter_to_df(xlabel)
            df_out_with_winter_f = join(result_dir, xlabel + '_with_winter.df')
            T.save_df(df_out_with_winter, df_out_with_winter_f)
            # except:
            #     pass
            # plt.savefig(png_dir + xlabel + '.pdf',dpi=300)
            # plt.close()
            # plt.show()
            # pause()

    def cross_koppen_landuse_linear_reg_all(self, var):
        dest_Y = 'recovery_time'
        plt.subplot(133)
        df = self.load_df()
        # df = df[df['winter_mark_new'] == 1]
        kp_list = []
        for zone in Global_vars().koppen_landuse():
            lc, kp = zone.split('.')
            kp_list.append(kp)
        kp_list = list(set(kp_list))
        kp_list.sort()
        # cmap = sns.diverging_palette(236, 0, s=99, l=50, n=len(kp_list), center="light")[::-1]
        # T.plot_colors_palette(cmap)
        # plt.figure()
        color_dic = {}
        cm = 0
        print(kp_list)
        color_dic['B'] = '#EB6100'
        color_dic['Cf'] = '#F6B580'
        color_dic['Csw'] = '#FCE4CF'
        color_dic['Df'] = '#C7B2D6'
        color_dic['Dsw'] = '#8E60A5'
        color_dic['E'] = '#601986'
        # print(color_dic)
        # exit()
        for kp in kp_list:
            # print(kp)
            # color_dic[kp] = cmap[cm]
            cm += 1
        marker_dic = Global_vars().marker_dic()

        line_color_dic = {
            'early': 'g',
            'mid': 'r',
            'late': 'b'
        }

        # for eln in ['early', 'mid', 'late']:
        # plt.figure()
        xx = []
        yy = []
        for zone in Global_vars().koppen_landuse():
            df_kl = df[df['climate_zone'] == zone]
            # df_kl = df_kl[df_kl['timing'] == eln]
            Y = df_kl[dest_Y]
            HI = df_kl[var]
            lc, kp = zone.split('.')
            if '.A' in zone:
                continue
            y_mean = Y.mean()
            HI_mean = HI.mean()
            xx.append(HI_mean)
            yy.append(y_mean)
            xerr = HI.std()
            yerr = Y.std()
            c = color_dic[kp]
            s = marker_dic[lc]
            # print(c)
            plt.scatter(HI_mean, y_mean, color=c, s=80, marker=s, edgecolors='black', linewidths=1, zorder=99)
            plt.errorbar(HI_mean, y_mean, xerr=xerr / 8., yerr=yerr / 8., color='gray', zorder=0, alpha=0.5)

        sns.regplot(xx, yy, scatter=False, color='gray')
        a, b, r = KDE_plot().linefit(xx, yy)
        # plt.title('a:{} b:{} r:{}'.format(a,b,r))

        # plt.ylim(5,13)
        plt.title('all')
        plt.text(min(xx), max(yy), 'y={:0.2f}x+{:0.2f}, r2:{:0.2f}'.format(a, b, r * r), zorder=100)
        # plt.show()
        pass

    def load_df(self):

        df = Global_vars().load_df_all()
        df = df[df['lat'] > 30]
        df = df[df['CV_PRE'] < 100000]
        df = df[df['current_sos_std_anomaly'] < 20]
        df = df[df['current_sos_std_anomaly'] > -20]
        df = df[df['current_sos_anomaly'] < 20]
        df = df[df['current_sos_anomaly'] > -20]
        return df

    def cross_koppen_landuse_linear_reg_with_winter(self, var):
        dest_Y = 'recovery_time'
        plt.subplot(132)
        df = self.load_df()
        df = df[df['winter_mark'] == 1]
        color_dic = {}
        lc_list = T.get_df_unique_val_list(df, 'lc')
        lc_list = list(lc_list)
        lc_list.remove('Forest')
        kp_list = T.get_df_unique_val_list(df, 'kp')
        kp_list = list(kp_list)
        color_dic['B'] = '#EB6100'
        color_dic['Cf'] = '#F6B580'
        color_dic['Csw'] = '#FCE4CF'
        color_dic['Df'] = '#C7B2D6'
        color_dic['Dsw'] = '#8E60A5'
        color_dic['E'] = '#601986'
        marker_dic = Global_vars().marker_dic()

        xx = []
        yy = []
        for lc in lc_list:
            df_lc = df[df['lc'] == lc]
            for kp in kp_list:
                if kp == 'A':
                    continue
                df_kp = df_lc[df_lc['kp'] == kp]
                Y = df_kp[dest_Y]
                HI = df_kp[var]
                y_mean = Y.mean()
                HI_mean = HI.mean()
                xx.append(HI_mean)
                yy.append(y_mean)
                xerr = HI.std()
                yerr = Y.std()
                c = color_dic[kp]
                s = marker_dic[lc]
                plt.scatter(HI_mean, y_mean, color=c, s=80, marker=s, edgecolors='black', linewidths=1, zorder=99)
                plt.errorbar(HI_mean, y_mean, xerr=xerr / 4., yerr=yerr / 4., color='gray', zorder=0, alpha=0.5)

        sns.regplot(xx, yy, scatter=False, color='gray')
        a, b, r, _ = T.nan_line_fit(xx, yy)
        _, p = T.nan_correlation(xx, yy)
        # plt.title('a:{} b:{} r:{}'.format(a,b,r))

        # plt.ylim(5,13)
        plt.title('with winter')
        plt.text(min(xx), max(yy), 'y={:0.2f}x+{:0.2f}\n r2:{:0.2f}\np:{:0.2f}'.format(a, b, r * r, p), zorder=100)

        # plt.show()

    def cross_koppen_landuse_linear_reg_with_winter_to_df(self, var):
        dest_Y = 'recovery_time'
        plt.subplot(132)
        df = self.load_df()
        df = df[df['winter_mark'] == 1]
        color_dic = {}
        lc_list = T.get_df_unique_val_list(df, 'lc')
        lc_list = list(lc_list)
        lc_list.remove('Forest')
        kp_list = T.get_df_unique_val_list(df, 'kp')
        kp_list = list(kp_list)
        color_dic['B'] = '#EB6100'
        color_dic['Cf'] = '#F6B580'
        color_dic['Csw'] = '#FCE4CF'
        color_dic['Df'] = '#C7B2D6'
        color_dic['Dsw'] = '#8E60A5'
        color_dic['E'] = '#601986'
        marker_dic = Global_vars().marker_dic()

        xx = []
        yy = []
        x_err_list = []
        y_err_list = []
        lc_list_out = []
        kp_list_out = []
        for lc in lc_list:
            df_lc = df[df['lc'] == lc]
            for kp in kp_list:
                if kp == 'A':
                    continue
                df_kp = df_lc[df_lc['kp'] == kp]
                Y = df_kp[dest_Y]
                HI = df_kp[var]
                y_mean = Y.mean()
                HI_mean = HI.mean()
                xx.append(HI_mean)
                yy.append(y_mean)
                xerr = HI.std()
                yerr = Y.std()
                xerr = xerr / 4.
                yerr = yerr / 4.
                x_err_list.append(xerr)
                y_err_list.append(yerr)
                c = color_dic[kp]
                s = marker_dic[lc]
                lc_list_out.append(lc)
                kp_list_out.append(kp)
                plt.scatter(HI_mean, y_mean, color=c, s=80, marker=s, edgecolors='black', linewidths=1, zorder=99)
                plt.errorbar(HI_mean, y_mean, xerr=xerr, yerr=yerr, color='gray', zorder=0, alpha=0.5)

        df_out = pd.DataFrame()
        df_out['x'] = xx
        df_out['y'] = yy
        df_out['x_err'] = x_err_list
        df_out['y_err'] = y_err_list
        df_out['lc'] = lc_list_out
        df_out['kp'] = kp_list_out
        return df_out
        # plt.show()

    def cross_koppen_landuse_linear_reg_no_winter_to_df(self, var):
        dest_Y = 'recovery_time'
        plt.subplot(131)
        df = self.load_df()
        df = df[df['winter_mark'] == 0]
        lc_list = T.get_df_unique_val_list(df, 'lc')
        lc_list = list(lc_list)
        lc_list.remove('Forest')
        kp_list = T.get_df_unique_val_list(df, 'kp')
        kp_list = list(kp_list)
        # df = df[df['timing'] == 'mid']

        color_dic = {}
        cm = 0
        color_dic['B'] = '#EB6100'
        color_dic['Cf'] = '#F6B580'
        color_dic['Csw'] = '#FCE4CF'
        color_dic['Df'] = '#C7B2D6'
        color_dic['Dsw'] = '#8E60A5'
        color_dic['E'] = '#601986'
        marker_dic = Global_vars().marker_dic()

        xx = []
        yy = []
        x_err_list = []
        y_err_list = []
        lc_list_out = []
        kp_list_out = []
        for lc in lc_list:
            df_lc = df[df['lc'] == lc]
            for kp in kp_list:
                if kp == 'A':
                    continue
                df_kp = df_lc[df_lc['kp'] == kp]
                Y = df_kp[dest_Y]
                HI = df_kp[var]
                y_mean = Y.mean()
                HI_mean = HI.mean()
                xx.append(HI_mean)
                yy.append(y_mean)
                xerr = HI.std()
                yerr = Y.std()
                xerr = xerr / 4.
                yerr = yerr / 4.
                c = color_dic[kp]
                s = marker_dic[lc]
                lc_list_out.append(lc)
                kp_list_out.append(kp)
                plt.scatter(HI_mean, y_mean, color=c, s=80, marker=s, edgecolors='black', linewidths=1, zorder=99)
                plt.errorbar(HI_mean, y_mean, xerr=xerr, yerr=yerr, color='gray', zorder=0, alpha=0.5)
                x_err_list.append(xerr)
                y_err_list.append(yerr)
        df_out = pd.DataFrame()
        df_out['x'] = xx
        df_out['y'] = yy
        df_out['x_err'] = x_err_list
        df_out['y_err'] = y_err_list
        df_out['lc'] = lc_list_out
        df_out['kp'] = kp_list_out
        return df_out

    def cross_koppen_landuse_linear_reg_no_winter(self, var):
        dest_Y = 'recovery_time'
        plt.subplot(131)
        df = self.load_df()
        df = df[df['winter_mark'] == 0]
        lc_list = T.get_df_unique_val_list(df, 'lc')
        lc_list = list(lc_list)
        lc_list.remove('Forest')
        kp_list = T.get_df_unique_val_list(df, 'kp')
        kp_list = list(kp_list)
        # df = df[df['timing'] == 'mid']

        color_dic = {}
        cm = 0
        color_dic['B'] = '#EB6100'
        color_dic['Cf'] = '#F6B580'
        color_dic['Csw'] = '#FCE4CF'
        color_dic['Df'] = '#C7B2D6'
        color_dic['Dsw'] = '#8E60A5'
        color_dic['E'] = '#601986'
        marker_dic = Global_vars().marker_dic()

        xx = []
        yy = []
        for lc in lc_list:
            df_lc = df[df['lc'] == lc]
            for kp in kp_list:
                if kp == 'A':
                    continue
                df_kp = df_lc[df_lc['kp'] == kp]
                Y = df_kp[dest_Y]
                HI = df_kp[var]
                y_mean = Y.mean()
                HI_mean = HI.mean()
                xx.append(HI_mean)
                yy.append(y_mean)
                xerr = HI.std()
                yerr = Y.std()
                c = color_dic[kp]
                s = marker_dic[lc]
                plt.scatter(HI_mean, y_mean, color=c, s=80, marker=s, edgecolors='black', linewidths=1, zorder=99)
                plt.errorbar(HI_mean, y_mean, xerr=xerr / 4., yerr=yerr / 4., color='gray', zorder=0, alpha=0.5)

        sns.regplot(xx, yy, scatter=False, color='gray')
        # a, b, r, _ = KDE_plot().linefit(xx, yy)
        a, b, r, _ = T.nan_line_fit(xx, yy)
        # plt.title('a:{} b:{} r:{}'.format(a,b,r))
        _, p = T.nan_correlation(xx, yy)
        # plt.ylim(5,13)
        plt.title('no winter')
        plt.text(min(xx), max(yy), 'y={:0.2f}x+{:0.2f}\n r2:{:0.2f}\np:{:0.2f}'.format(a, b, r * r, p), zorder=100)

        # plt.show()

    def plot_ci_manual(self, t, s_err, n, x, x2, y2, z, color, ax=None):
        """Return an axes of confidence bands using a simple approach.

        Notes
        -----
        .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
        .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}

        References
        ----------
        .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
           http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb

        """
        if ax is None:
            ax = plt.gca()

        ci = t * s_err * np.sqrt(1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
        # ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7", edgecolor="")
        ax.add_collection3d(plt.fill_between(x2, y2 + ci, y2 - ci, color=color, edgecolor=color, alpha=0.3), z,
                            zdir='y')

        return ax

    def plot_ci_bootstrap(self, xs, ys, resid, nboot=500, ax=None):
        """Return an axes of confidence bands using a bootstrap approach.

        Notes
        -----
        The bootstrap approach iteratively resampling residuals.
        It plots `nboot` number of straight lines and outlines the shape of a band.
        The density of overlapping lines indicates improved confidence.

        Returns
        -------
        ax : axes
            - Cluster of lines
            - Upper and Lower bounds (high and low) (optional)  Note: sensitive to outliers

        References
        ----------
        .. [1] J. Stults. "Visualizing Confidence Intervals", Various Consequences.
           http://www.variousconsequences.com/2010/02/visualizing-confidence-intervals.html

        """
        if ax is None:
            ax = plt.gca()

        bootindex = scipy.random.randint

        for _ in range(nboot):
            resamp_resid = resid[bootindex(0, len(resid) - 1, len(resid))]
            # Make coeffs of for polys
            pc = scipy.polyfit(xs, ys + resamp_resid, 1)
            # Plot bootstrap cluster
            ax.plot(xs, scipy.polyval(pc, xs), "b-", linewidth=2, alpha=3.0 / float(nboot))

        return ax

    def koppen_timing_recovery_3d(self, xlabel):
        # Y lag water_balance NDVI_pre_6_mean sand sos_origin
        # xlabel = 'water_balance'
        # xlabel = 'lag'
        # xlabel = 'NDVI_pre_6_mean'
        # xlabel = 'sos_origin'
        # xlabel = 'correlation'
        # xlabel = 'PRE_pre_12_mean'
        # xlabel = 'TMP_pre_12_mean'
        # xlabel = 'TMP_lag_mean'
        # xlabel = 'PRE_lag_mean'
        # xlabel = 'SWE_lag_mean'
        # xlabel = 'sensitivity'
        temp_pdf_dir = results_root + 'AI/temp_pdf/3d/'
        T.mk_dir(temp_pdf_dir, force=1)
        ylabel = 'Y'
        df_f = Main_flow_Prepare().this_class_arr + 'prepare/data_frame_threshold_0.df'
        df = T.load_df(df_f)
        df = df[df['correlation'] > 0]
        # df = df[df['winter_mark'] == 0]
        # df = df[df['winter_mark'] == 1]
        kp_list = []
        for zone in Global_vars().koppen_landuse():
            lc, kp = zone.split('.')
            kp_list.append(kp)
        kp_list = list(set(kp_list))
        kp_list.sort()
        # precmap = sns.diverging_palette(177, 131, s=99, l=50, n=len(kp_list), center="light")
        precmap = sns.color_palette("Greens_r", n_colors=len(kp_list))
        earlycmap = sns.color_palette("Reds_r", n_colors=len(kp_list))
        latecmap = sns.color_palette("Blues_r", n_colors=len(kp_list))
        # earlycmap = sns.diverging_palette(57, 0, s=99, l=50, n=len(kp_list), center="light")
        # latecmap = sns.diverging_palette(275, 240, s=99, l=50, n=len(kp_list), center="light")
        precolor_dic = {}
        earlycolor_dic = {}
        latecolor_dic = {}
        cm = 0
        for kp in kp_list:
            precolor_dic[kp] = precmap[cm]
            earlycolor_dic[kp] = earlycmap[cm]
            latecolor_dic[kp] = latecmap[cm]
            cm += 1
        size_dic = {
            'Shrublands': 20,
            'Forest': 20,
            'Grasslands': 20,
        }
        marker_dic = {
            'Shrublands': 'x',
            'Forest': 'x',
            'Grasslands': 'x',
            # 'Cf':'s',
            # 'Csw':'p',
            # 'Df':'P',
            # 'Dsw':'*',
            # 'E':'D',
        }
        size_dic_new = {}
        for zone in Global_vars().koppen_landuse():
            for lc in size_dic:
                if lc in zone:
                    size_dic_new[zone] = size_dic[lc]

        marker_dic_new = {}
        for zone in Global_vars().koppen_landuse():
            for kp in marker_dic:
                if kp in zone:
                    marker_dic_new[zone] = marker_dic[kp]

        # line_color_dic = {
        #     'pre':'g',
        #     'early':'r',
        #     'late':'b'
        # }
        # colordic_eln = {
        #     'pre':precolor_dic,
        #     'early':earlycolor_dic,
        #     'late':latecolor_dic,
        # }

        bcmap = sns.color_palette("Blues_r", n_colors=4)
        rcmap = sns.color_palette("Reds_r", n_colors=4)
        gcmap = sns.color_palette("Greens_r", n_colors=3)
        line_color_dic = {
            'p1': gcmap[1],
            'p2': gcmap[0],
            'e1': rcmap[2],
            'e2': rcmap[1],
            'e3': rcmap[0],
            'l1': bcmap[2],
            'l2': bcmap[1],
            'l3': bcmap[0],
        }
        # print line_color_dic
        # exit()

        # for eln in ['pre','early','late']:
        k = []
        rr = []
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # ax.set_xticks()
        zz = 0
        for eln in Global_vars().timing_list():
            color = line_color_dic[eln]
            # plt.figure()
            xx = []
            yy = []
            for zone in Global_vars().koppen_landuse():

                df_kl = df[df['climate_zone'] == zone]
                # df_kl = df_kl[df_kl['new_eln'] == eln]
                df_kl = df_kl[df_kl['timing'] == eln]
                Y = df_kl[ylabel]
                X = df_kl[xlabel]
                lc, kp = zone.split('.')
                if '.A' in zone:
                    continue
                # X = X[X>0.2]
                # print len(X)
                # X = X[X<0]
                y_mean = np.nanmean(Y)
                X_mean = np.nanmean(X)
                if not np.isnan(y_mean) and not np.isnan(X_mean):
                    xx.append(X_mean)
                    yy.append(y_mean)
                    xerr = X.std()
                    yerr = Y.std()
                    # c = colordic_eln[eln][kp]
                    # s = marker_dic[lc]
                # plt.scatter(HI_mean,y_mean,color=c,s=80,marker=s,edgecolors='black',linewidths=1,zorder=99)
                # plt.scatter(x_mean,y_mean,s=80,marker=markers_dic_new[zone],edgecolors='black',linewidths=1,zorder=99)
                ax.scatter(X_mean, zz, y_mean, s=size_dic_new[zone], color=color, edgecolors='black',
                           marker=marker_dic_new[zone], alpha=0.7)
                # plt.errorbar(HI_mean, y_mean, xerr=xerr / 8., yerr=yerr / 8., color='gray', zorder=0, alpha=0.5)

            # sns.regplot(xx, yy, scatter=True, color=line_color_dic[eln])
            # ax.scatter(xx,zz,yy,marker='x',color=color,edgecolors=None)
            # print len(xx)
            # print xx
            # df_temp = pd.DataFrame()
            # df_temp['X'] = xx
            # df_temp['Y'] = yy
            # df_temp = df_temp.dropna()
            # xx = df_temp['X']
            # yy = df_temp['Y']
            # xx = list(xx)
            # yy = list(yy)
            # print xx
            # print yy
            # sns.regplot(xx, yy, scatter=True)
            a, b, r = KDE_plot().linefit(xx, yy)
            xfit = np.linspace(min(xx), max(xx), 10)
            yfit = [zz] * 10
            zfit = a * xfit + b
            ax.plot(xfit, yfit, zfit, color=color)

            # p, cov = np.polyfit(xx, yy, 1, cov=True)  # parameters and covariance from of the fit of 1-D polynom.
            # y_model = self.equation(p, xx)
            # xx = np.array(xx)
            # yy = np.array(yy)
            # n = xx.size  # number of observations
            # m = p.size
            # t = stats.t.ppf(0.99, n - m)
            # dof = n - m
            # resid = yy - y_model
            # chi2 = np.sum((resid / y_model) ** 2)  # chi-squared; estimates error in data
            # chi2_red = chi2 / dof  # reduced chi-squared; measures goodness of fit
            # s_err = np.sqrt(np.sum(resid ** 2) / dof)

            # x2 = np.linspace(np.min(xx), np.max(xx), 100)
            # y2 = self.equation(p, x2)
            # self.plot_ci_manual(t, s_err, n, xx, x2, y2,color=color,z=zz,ax=ax)
            zz += 1

            k.append(a)
            rr.append(r * r)
            # print '{} a:{} b:{} r:{}'.format(eln,a,b,r)
            # print 'y={:0.2f}x+{:0.2f}'.format(a,b),'r:{:0.2f}'.format(r)
            # plt.title('{:0.2f} a:{} b:{} r:{}'.format(eln,a,b,r))
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Timing')
        ax.set_zlabel(ylabel)
        # self.axisEqual3D(ax)
        ax.view_init(elev=36., azim=-45)
        plt.savefig(temp_pdf_dir + '3d_Y_{}_timing.pdf'.format(xlabel))
        plt.close()
        ##############################
        ##############################
        fig, host = plt.subplots()
        print('r min:{} max:{}'.format(min(rr), max(rr)))
        par1 = host.twinx()
        host.plot(Global_vars().months_list(), k, '-o', color='b', linewidth=4, alpha=0.7, markersize=14)
        host.set_xlabel('Timing')
        host.set_ylabel('{}-{} Slope'.format(xlabel, ylabel))
        host.yaxis.label.set_color('b')
        host.tick_params(axis='y', colors='b')
        host.tick_params(axis="y", direction="in", length=10)
        host.tick_params(axis="x", direction="in", length=0)

        par1.plot(Global_vars().months_list(), rr, '-o', color='r', linewidth=4, alpha=0.7, markersize=14)
        par1.set_ylabel('{}-{} r2'.format(xlabel, ylabel))
        par1.yaxis.label.set_color('r')
        par1.tick_params(axis='y', colors='r')
        par1.tick_params(axis="y", direction="in", length=10)
        par1.spines["right"].set_edgecolor('r')
        par1.spines["left"].set_edgecolor('b')
        plt.savefig(temp_pdf_dir + '3d_Y_{}_timing_k.pdf'.format(xlabel))
        plt.close()
        ##############################
        ##############################

        fig3, ax3 = plt.subplots()
        xx = []
        yy = []
        color_dic = {}
        cm = 0

        kp_list = kp_list[1:]
        kp_list = kp_list[::-1]
        cmap = sns.diverging_palette(236, 0, s=99, l=50, n=len(kp_list), center="light")

        for kp in kp_list:
            color_dic[kp] = cmap[cm]
            cm += 1
        marker_dic = Global_vars().marker_dic()
        for zone in Global_vars().koppen_landuse():
            if 'A' in zone:
                continue
            df_kl = df[df['climate_zone'] == zone]
            x = df_kl[xlabel]
            y = df_kl[ylabel]
            x = x.dropna()
            y = y.dropna()
            xmean = np.mean(x)
            ymean = np.mean(y)
            xerr = np.std(x)
            yerr = np.std(y)
            xx.append(xmean)
            yy.append(ymean)
            lc, kp = zone.split('.')
            c = color_dic[kp]
            s = marker_dic[lc]
            ax3.scatter(xmean, ymean, color=c, s=300, marker=s, edgecolors='black', linewidths=1, zorder=99, alpha=1)
            ax3.errorbar(xmean, ymean, xerr=xerr / 8., yerr=yerr / 8., color='gray', zorder=0, alpha=0.5)

        sns.regplot(xx, yy, scatter=False)
        a, b, r = KDE_plot().linefit(xx, yy)
        plt.title('y={:0.2f}x+{:0.2f} r={:0.2f}'.format(a, b, r))
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel(ylabel)

        ax3.tick_params(axis="y", direction="in", length=10)
        ax3.tick_params(axis="x", direction="in", length=10)
        plt.savefig(temp_pdf_dir + '3d_Y_{}_scatter.pdf'.format(xlabel))
        plt.close()
        # plt.show()
        ##############################
        ##############################

        #####################
        #####################
        #####################
        #####################

    def box_plot(self):
        # Y lag water_balance NDVI_pre_6_mean sand sos_origin
        # xlabel = 'water_balance'
        # xlabel = 'lag'
        xlabel = 'Y'
        # xlabel = 'NDVI_pre_12_mean'
        # xlabel = 'sos_origin'
        # xlabel = 'correlation'
        # xlabel = 'PRE_pre_12_mean'
        # xlabel = 'TMP_pre_12_mean'
        # xlabel = 'TMP_lag_mean'
        # xlabel = 'PRE_lag_mean'
        # xlabel = 'SWE_lag_mean'
        # xlabel = 'sensitivity'
        df_f = Prepare_CWD_X_pgs_egs_lgs4().this_class_arr + 'prepare/data_frame_threshold_0.df'
        df = T.load_df(df_f)

        bcmap = sns.color_palette("Blues_r", n_colors=4)
        rcmap = sns.color_palette("Reds_r", n_colors=4)
        gcmap = sns.color_palette("Greens_r", n_colors=3)
        line_color_dic = {
            'p1': gcmap[1],
            'p2': gcmap[0],
            'e1': rcmap[2],
            'e2': rcmap[1],
            'e3': rcmap[0],
            'l1': bcmap[2],
            'l2': bcmap[1],
            'l3': bcmap[0],
        }

        colors = []
        for timing in Global_vars().timing_list():
            colors.append(line_color_dic[timing])

        recovery = []
        # err = []
        for timing in Global_vars().timing_list():
            timing_df = df[df['timing'] == timing]
            r = timing_df[xlabel]
            r = r.dropna()
            std = np.std(r) / 8.
            recovery.append(r)
            # err.append(std)
        # plt.bar(Global_vars().months_list(),recovery,
        #         yerr=err,capsize=10,align='center',
        #         alpha=0.7,ecolor='black',color=colors
        #         )
        # patch = plt.boxplot(recovery,
        #             showfliers=False,
        #             showmeans=True,
        #             labels=Global_vars().months_list(),
        #             color=colors
        #             )
        # element = 'boxes'
        flag = 0
        for box in recovery:
            # print box
            patch = plt.boxplot(box,
                                showfliers=False,
                                showmeans=True,
                                # labels=[Global_vars().months_list()[flag]],
                                positions=[flag],
                                patch_artist=True,
                                widths=0.5
                                )
            for element in ['boxes', 'whiskers', 'caps']:
                plt.setp(patch[element], color=colors[flag], alpha=0.7)
            flag += 1

        plt.xlabel('Timing')
        plt.ylabel('{}'.format(xlabel))
        plt.show()

    def equation(self, a, b):
        """Return a 1D polynomial."""
        return np.polyval(a, b)

    def sensitivity(self):
        df_f = Prepare_CWD_X_pgs_egs_lgs4().this_class_arr + 'prepare/data_frame_threshold_0.df'
        df = T.load_df(df_f)

        kp_list = []
        for zone in Global_vars().koppen_landuse():
            lc, kp = zone.split('.')
            kp_list.append(kp)
        kp_list = list(set(kp_list))
        kp_list.sort()
        kp_list = kp_list[2:]
        print(kp_list)
        cmap = sns.diverging_palette(236, 0, s=99, l=50, n=len(kp_list), center="light")
        color_dic = {}
        cm = 0
        for kp in kp_list:
            color_dic[kp] = cmap[cm]
            cm += 1
        marker_dic = Global_vars().marker_dic()

        line_color_dic = {
            'pre': 'g',
            'early': 'r',
            'late': 'b'
        }

        for eln in ['pre', 'early', 'late']:
            # plt.figure()
            xx = []
            yy = []
            for zone in Global_vars().koppen_landuse():
                df_kl = df[df['climate_zone'] == zone]
                df_kl = df_kl[df_kl['new_eln'] == eln]
                # print df_kl.columns
                # exit()
                # Y = df_kl.sensitivity
                Y = df_kl['Y']
                HI = df_kl.water_balance
                lc, kp = zone.split('.')
                if '.A' in zone or '.B' in zone:
                    continue
                y_mean = Y.mean()
                HI_mean = HI.mean()
                xx.append(HI_mean)
                yy.append(y_mean)
                xerr = HI.std()
                yerr = Y.std()
                c = color_dic[kp]
                s = marker_dic[lc]
                plt.scatter(HI_mean, y_mean, color=c, s=80, marker=s, edgecolors='black', linewidths=1, zorder=99)
                plt.errorbar(HI_mean, y_mean, xerr=xerr / 8., yerr=yerr / 8., color='gray', zorder=0, alpha=0.5)

            sns.regplot(xx, yy, scatter=False, color=line_color_dic[eln])
            a, b, r = KDE_plot().linefit(xx, yy)
            # plt.title('{} a:{} b:{} r:{}'.format(eln,a,b,r))
            # plt.ylim(5,13)
            # plt.title(eln)
        plt.show()
        pass

    def effect(self):
        effect_df_f = Sensitivity_and_effect().this_class_arr + 'add_ndvi_decrease_to_df/df.df'
        effect_df = T.load_df(effect_df_f)

        # effect_df = Sensitivity_and_effect().add_ndvi_decrease_to_df()

        kp_list = []
        for zone in Global_vars().koppen_landuse():
            lc, kp = zone.split('.')
            kp_list.append(kp)
        kp_list = list(set(kp_list))
        kp_list.sort()
        cmap = sns.diverging_palette(236, 0, s=99, l=50, n=len(kp_list), center="light")
        color_dic = {}
        cm = 0
        for kp in kp_list:
            color_dic[kp] = cmap[cm]
            cm += 1
        marker_dic = Global_vars().marker_dic()

        line_color_dic = {
            'pre': 'g',
            'early': 'r',
            'late': 'b'
        }

        for eln in ['pre', 'early', 'late']:
            print(eln)
            plt.figure()
            xx = []
            yy = []
            for zone in Global_vars().koppen_landuse():
                df_kl = effect_df[effect_df['climate_zone'] == zone]
                df_kl = df_kl[df_kl['eln'] == eln]
                Y = df_kl['Y']
                effect = df_kl.ndvi_decrease
                lc, kp = zone.split('.')
                if '.A' in zone:
                    continue
                print(zone, len(Y))
                y_mean = Y.mean()
                effect_mean = effect.mean()
                xx.append(effect_mean)
                yy.append(y_mean)
                xerr = effect.std()
                yerr = Y.std()
                c = color_dic[kp]
                s = marker_dic[lc]
                plt.scatter(effect_mean, y_mean, color=c, s=80, marker=s, edgecolors='black', linewidths=1, zorder=99)
                plt.errorbar(effect_mean, y_mean, xerr=xerr / 8., yerr=yerr / 8., color='gray', zorder=0, alpha=0.5)

            sns.regplot(xx, yy, scatter=False, color=line_color_dic[eln])
            # a, b, r = KDE_plot().linefit(xx, yy)
            # plt.title('{} a:{} b:{} r:{}'.format(eln,a,b,r))
            # plt.ylim(5, 13)
        plt.show()


class Partial_Dependence_Plots:
    '''
    Ref:
    https://towardsdatascience.com/looking-beyond-feature-importance-37d2807aaaa7
    '''

    def __init__(self):
        self.this_class_arr = result_root_this_script + 'arr/Partial_Dependence_Plots/'
        # self.this_class_tif = result_root_this_script + 'tif/Partial_Dependence_Plots/'
        self.this_class_png = result_root_this_script + 'png/Partial_Dependence_Plots/'
        Tools().mk_dir(self.this_class_arr, force=True)
        # Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        pass

    def run(self):

        # self.count_sample_size_in_different_models()
        # self.save_pdp_result()
        self.plot_result()
        # self.save_pdp_result_humid_arid()
        # self.plot_result_humid_arid()
        # self.check_sos_hist()

    def save_pdp_result(self):
        # out_dir = self.this_class_arr + 'save_pdp_result_all_new_lc/'
        out_dir = self.this_class_arr + 'save_pdp_result_all_new_lc_add_xy/'
        T.mk_dir(out_dir)
        y_var = 'recovery_time'
        df = Global_vars().load_df()
        # T.print_head_n(df,n=100)
        # exit()

        # exit()
        # df = df[df['post_3_month_greater_than_threshold']==1]
        winter_list = [0, 1]
        winter_str_list = ['no_winter', 'with_winter']
        x_var_list = [Global_vars().vars_RF_no_winter(), Global_vars().vars_RF_with_winter()]
        lc_list = Global_vars().lc_list()
        # lat_list = np.linspace(30, 90, 5)
        lat_list = [30, 90]

        # print(lat_list)
        # exit()
        all_result_dic = {}
        for l in range(len(lat_list)):
            if l + 1 >= len(lat_list):
                continue
            df_lat = df[df['lat'] > lat_list[l]]
            df_lat = df_lat[df_lat['lat'] < lat_list[l + 1]]
            for lc in lc_list:
                df_lc = df_lat[df_lat['lc'] == lc]
                print(lc, len(df_lc))
                for i in range(len(winter_list)):
                    df_winter = df_lc[df_lc['winter_mark'] == winter_list[i]]
                    df_winter = Global_vars().clean_df(df_winter, winter_str_list[i])

                    key = f'{lat_list[l]}>>{lat_list[l + 1]}-{lc}-{winter_str_list[i]}'
                    print(key)
                    print(len(df_winter))
                    # try:
                    df_new = pd.DataFrame()
                    df_new[x_var_list[i]] = df_winter[x_var_list[i]]
                    df_new[y_var] = df_winter[y_var]
                    # df_new = Global_vars().clean_df(df_new)
                    df_new = df_new.dropna()
                    if len(df_new) < 10:
                        continue
                    pdp_result_dic = self.partial_dependent_plot_arr(df_new, x_var_list[i], y_var)

                    X = df_new[x_var_list[i]]
                    Y = df_new[y_var]
                    clf, importances_dic, r_X_dic, mse, r_model, score, Y_test, y_pred = RF().random_forest_train(X, Y,
                                                                                                                  x_var_list[
                                                                                                                      i])
                    all_result_dic[key] = (pdp_result_dic, importances_dic)
                    # except:
                    #     pass
        T.save_npy(all_result_dic, out_dir + 'pdp_result')

    def count_sample_size_in_different_models(self):
        # dff = Global_vars().dff
        # df = T.load_df(dff)
        # dff = Global_vars().dff
        # df = T.load_df(dff)
        df = Global_vars().load_df()
        # df = df[df['post_3_month_greater_than_threshold']==1]
        winter_list = [0, 1]
        winter_str_list = ['no_winter', 'with_winter']
        lc_list = Global_vars().lc_list()
        # lat_list = np.linspace(30, 90, 5)
        lat_list = [30, 90]
        all_sample = 0
        all_result_dic = {}
        for l in range(len(lat_list)):
            if l + 1 >= len(lat_list):
                continue
            df_lat = df[df['lat'] > lat_list[l]]
            df_lat = df_lat[df_lat['lat'] < lat_list[l + 1]]
            for lc in lc_list:
                df_lc = df_lat[df_lat['lc'] == lc]
                # print(len(df_lc))
                for i in range(len(winter_list)):
                    df_winter = df_lc[df_lc['winter_mark'] == winter_list[i]]
                    # df_winter = df_lc[df_lc['winter_mark_new']==winter_list[i]]
                    df_winter = Global_vars().clean_df(df_winter, winter_str_list[i])
                    all_sample += len(df_winter)
                    key = f'{lat_list[l]}>>{lat_list[l + 1]}-{lc}-{winter_str_list[i]}'
                    print(key)
                    print(len(df_winter))
                    print('----' * 8)
        print('all sample:', all_sample)

    def partial_dependent_plot_arr(self, df, x_vars, y_var):

        vars_list = x_vars
        selected_features = vars_list
        vars_list1 = copy.copy(selected_features)
        vars_list1.append(y_var)
        XX = df[vars_list1]
        XX = XX.dropna()
        # XX = Global_vars().clean_df(XX)
        vars_list1.remove(y_var)
        X = XX[vars_list1]
        Y = XX[y_var]
        model, r2 = self.train_model(X, Y)
        # print(r2)
        flag = 0
        # plt.figure(figsize=(12, 8))
        # exit()
        result_dic = {}
        for var in tqdm(x_vars):
            flag += 1
            # ax = plt.subplot(5, 5, flag)
            df_partial_plot = self.__get_PDPvalues(var, X, model)
            # print(df_partial_plot)
            # exit()
            ppx = df_partial_plot[var]
            ppy = df_partial_plot['PDs']
            ppy_std = df_partial_plot['PDs_std']
            ppx_smooth = SMOOTH().smooth_convolve(ppx, window_len=11)
            ppy_smooth = SMOOTH().smooth_convolve(ppy, window_len=11)
            ppy_std_smooth = SMOOTH().smooth_convolve(ppy_std, window_len=11)
            result_dic[var] = {
                'x': ppx,
                'y': ppy,
                'y_std': ppy_std,
                'x_smooth': ppx_smooth,
                'y_smooth': ppy_smooth,
                'y_std_smooth': ppy_std_smooth,
                'r2': r2,
            }
            # plt.plot(ppx_smooth,ppy_smooth)
            # plt.show()
        return result_dic

    def save_pdp_result_humid_arid(self):
        out_dir = self.this_class_arr + 'save_pdp_result_humid_arid/'
        T.mk_dir(out_dir)
        y_var = 'recovery_time'
        df = Global_vars().load_df()
        # df = df[df['post_3_month_greater_than_threshold']==1]
        winter_list = [0, 1]
        winter_str_list = ['no_winter', 'with_winter']
        humid_arid_list = ['humid', 'arid', ]
        x_var_list = [Global_vars().vars_RF_no_winter(), Global_vars().vars_RF_with_winter()]
        lc_list = Global_vars().lc_list()
        # lat_list = np.linspace(30, 90, 5)

        # print(lat_list)
        # exit()
        all_result_dic = {}
        for humid_arid in humid_arid_list:
            df_ha = df[df['humid_arid'] == humid_arid]
            for lc in lc_list:
                df_lc = df_ha[df_ha['lc'] == lc]
                for i in range(len(winter_list)):
                    df_winter = df_lc[df_lc['winter_mark'] == winter_list[i]]
                    key = f'{humid_arid}-{lc}-{winter_str_list[i]}'
                    print(key)
                    # try:
                    df_new = pd.DataFrame()
                    df_new[x_var_list[i]] = df_winter[x_var_list[i]]
                    df_new[y_var] = df_winter[y_var]
                    df_new = Global_vars().clean_df(df_new)
                    df_new = df_new.dropna()
                    if len(df_new) < 10:
                        continue
                    pdp_result_dic = self.partial_dependent_plot_arr(df_new, x_var_list[i], y_var)

                    X = df_new[x_var_list[i]]
                    Y = df_new[y_var]
                    clf, importances_dic, r_X_dic, mse, r_model, score, Y_test, y_pred = RF().random_forest_train(X, Y,
                                                                                                                  x_var_list[
                                                                                                                      i])
                    all_result_dic[key] = (pdp_result_dic, importances_dic)
        T.save_npy(all_result_dic, out_dir + 'pdp_result')

    def train_model(self, X, y):
        print(len(X))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42, test_size=0.2)
        # rf = RandomForestClassifier(n_estimators=300, random_state=42)
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=4)
        # rf = LinearRegression()
        rf.fit(X_train, y_train)
        r2 = rf.score(X_test, y_test)
        y_pred = rf.predict(X_test)
        # y_pred = rf.predict(X_train)
        # plt.scatter(y_pred,y_test)
        # print(r2)
        # plt.scatter(y_pred,y_train)
        # plt.show()

        return rf, r2

    def __get_PDPvalues(self, col_name, data, model, grid_resolution=50):
        Xnew = data.copy()
        sequence = np.linspace(np.min(data[col_name]), np.max(data[col_name]), grid_resolution)
        Y_pdp = []
        Y_pdp_std = []
        for each in sequence:
            Xnew[col_name] = each
            Y_temp = model.predict(Xnew)
            Y_pdp.append(np.mean(Y_temp))
            Y_pdp_std.append(np.std(Y_temp))
        return pd.DataFrame({col_name: sequence, 'PDs': Y_pdp, 'PDs_std': Y_pdp_std})

    def __plot_PDP(self, col_name, data, model):
        df = self.__get_PDPvalues(col_name, data, model)
        plt.rcParams.update({'font.size': 16})
        plt.rcParams["figure.figsize"] = (6, 5)
        fig, ax = plt.subplots()
        # ax.plot(data[col_name], np.zeros(data[col_name].shape)+min(df['PDs'])-1, 'k|', ms=15)  # rug plot
        ax.plot(df[col_name], df['PDs'], lw=2)
        ax.set_ylabel('Recovery time')
        ax.set_xlabel(col_name)
        plt.tight_layout()
        return ax

    def __x_var_list_rank(self, x_var_list, imp_dic):
        imp_list = []
        # print(x_var_list)
        # exit()
        for x in x_var_list:
            imp = imp_dic[x]
            imp_list.append(imp)
        imp_list_sort = np.argsort(imp_list)[::-1]
        # print(imp_list_sort)
        # exit()
        sorted_vals = []
        sorted_var_name = []
        for i in range(len(imp_list)):
            sorted_vals.append(imp_list[imp_list_sort[i]])
            sorted_var_name.append(x_var_list[imp_list_sort[i]])
        return sorted_var_name


    def plot_result(self):
        outpngdir = self.this_class_png + 'save_pdp_result_all_new_lc_add_xy/'
        T.mk_dir(outpngdir)
        # dic_f = self.this_class_arr + '/save_pdp_result/pdp_result.npy'
        dic_f = self.this_class_arr + '/save_pdp_result_all_new_lc_add_xy/pdp_result.npy'
        # print(dic_f)
        # exit()
        dic = T.load_npy(dic_f)
        # for key in dic:
        #     # print(key)
        #     dic_i = dic[key]
        #     print(dic_i)
        #     print('----')
        # exit()

        # for key_i in dic_i:
        #     print(key_i)
        #     print('----')
        # dic_ii = dic_i[key_i]
        # exit()
        winter_list = [0, 1]
        winter_str_list = ['no_winter', 'with_winter']
        x_var_list = [Global_vars().vars_RF_no_winter(), Global_vars().vars_RF_with_winter()]
        lc_list = Global_vars().lc_list()
        # lat_list = np.linspace(30, 90, 5)
        # lat_list = [30,45,90]
        lat_list = [30, 90]
        for l in range(len(lat_list)):
            if l + 1 >= len(lat_list):
                continue
            for i in range(len(winter_list)):
                title = f'{lat_list[l]}>>{lat_list[l + 1]}-{winter_str_list[i]}'
                x_var_list_i = x_var_list[i]
                # print(x_var_list_i)
                # exit()
                plt.figure(figsize=(12, 8))
                x_var_list_i_sort = 'None'
                for lc in lc_list:
                    key = f'{lat_list[l]}>>{lat_list[l + 1]}-{lc}-{winter_str_list[i]}'
                    print(key)

                    if not key in dic:
                        continue
                    # print(lc)
                    # print(key)
                    # exit()
                    vals = dic[key][0]
                    imps = dic[key][1]
                    # print(imps)
                    # exit()
                    # print(vals)
                    # exit()
                    flag = 0
                    c = Global_vars().lc_colors(lc)
                    if x_var_list_i_sort == 'None':
                        x_var_list_i_sort = self.__x_var_list_rank(x_var_list_i, imps)
                    for var in x_var_list_i_sort:
                        # print(var)
                        # exit()
                        flag += 1
                        plt.subplot(5, 5, flag)
                        # if not var == 'current_sos_anomaly':
                        #     continue
                        y = vals[var]['y']
                        x = vals[var]['x']
                        # print(y)
                        # print(x)
                        # exit()
                        y_std = vals[var]['y_std']
                        window_len = 11
                        y = SMOOTH().smooth_convolve(y, window_len=window_len)
                        y_std = SMOOTH().smooth_convolve(y_std, window_len=window_len)
                        r2 = vals[var]['r2']

                        if var in ['current_sos_anomaly']:
                            y = SMOOTH().smooth_convolve(y, window_len=21)[1:]
                            y_std = y_std / 9.
                        elif var in ['post_VPD', 'post_TMP', 'post_PRE', ]:
                            y = SMOOTH().smooth_convolve(y, window_len=21)[1:]
                            y_std = y_std / 3.
                        else:
                            y_std = y_std / 8.
                        ## unify the length of y and y_std
                        if len(y) > len(y_std):
                            y = y[:-1]
                        elif len(y) < len(y_std):
                            y_std = y_std[:-1]
                        ## unify the length of x and y
                        if len(x) > len(y):
                            x = x[:-1]
                        elif len(x) < len(y):
                            y = y[:-1]
                        # # print(y)
                        # print(len(y))
                        # print(len(y_std))

                        up = y + y_std
                        down = y - y_std

                        # just see sos -15 --> 15 days ################
                        # select_index = []
                        # for iii,xi in enumerate(x):
                        #     if xi > 15:
                        #         continue
                        #     if xi < -15:
                        #         continue
                        #     else:
                        #         select_index.append(iii)
                        # x = T.pick_vals_from_1darray(x,select_index)
                        # y = T.pick_vals_from_1darray(y,select_index)
                        # up = T.pick_vals_from_1darray(up,select_index)
                        # down = T.pick_vals_from_1darray(down,select_index)
                        # just see sos -15 --> 15 days ################

                        plt.plot(x, y, color=c)
                        plt.fill_between(x, up, down, color=c, alpha=0.2)
                        plt.xlabel(var)
                        plt.ylabel('recovery_time')
                        # if i == 0:
                        #     plt.ylim(0,2)
                        # if i == 1:
                        #     plt.ylim(2,7)
                plt.suptitle(title)
                plt.tight_layout()
                # plt.show()
                outpng_path = outpngdir + title + 'pdp.pdf'
                plt.savefig(outpng_path)
                # exit()
                plt.figure()
                for lc in lc_list:
                    key = f'{lat_list[l]}>>{lat_list[l + 1]}-{lc}-{winter_str_list[i]}'
                    if not key in dic:
                        continue
                    pdp = dic[key][0]
                    importance_dic = dic[key][1]
                    importance_list = []
                    r2_list = []
                    if x_var_list_i_sort == 'None':
                        x_var_list_i_sort = self.__x_var_list_rank(x_var_list_i, importance_dic)
                    for var in x_var_list_i_sort:
                        r2 = pdp[var]['r2']
                        r2_list.append(r2)
                        importance = importance_dic[var]
                        importance_list.append(importance)
                    c = Global_vars().lc_colors(lc)
                    plt.scatter(range(len(importance_list)), importance_list, color=c)
                    # print(r2_list)
                    plt.plot(range(len(importance_list)), r2_list, color=c)
                plt.xticks(range(len(x_var_list_i_sort)), x_var_list_i_sort, rotation=90)
                plt.title(title)
                plt.tight_layout()
                # exit()
                # plt.show()
                outpng_path = outpngdir + title + 'importance.pdf'
                plt.savefig(outpng_path)
        T.open_path_and_file(outpngdir)

    def plot_result_humid_arid(self):

        outpngdir = self.this_class_png + 'save_pdp_result_humid_arid/'
        T.mk_dir(outpngdir)
        dic_f = self.this_class_arr + '/save_pdp_result_humid_arid/pdp_result.npy'
        dic = T.load_npy(dic_f)
        winter_list = [0, 1]
        winter_str_list = ['no_winter', 'with_winter']
        x_var_list = [Global_vars().vars_RF_no_winter(), Global_vars().vars_RF_with_winter()]
        lc_list = Global_vars().lc_list()
        # lat_list = np.linspace(30, 90, 5)
        # lat_list = [30,45,90]
        humid_arid_list = ['humid', 'arid']
        for humid_arid in humid_arid_list:
            for i in range(len(winter_list)):
                title = f'{humid_arid}-{winter_str_list[i]}'
                x_var_list_i = x_var_list[i]
                plt.figure(figsize=(12, 8))
                x_var_list_i_sort = 'None'
                for lc in lc_list:
                    key = f'{humid_arid}-{lc}-{winter_str_list[i]}'

                    if not key in dic:
                        continue

                    vals = dic[key][0]
                    imps = dic[key][1]
                    # print(imps)
                    # exit()
                    # print(vals)
                    # exit()
                    flag = 0
                    c = Global_vars().lc_colors(lc)
                    if x_var_list_i_sort == 'None':
                        x_var_list_i_sort = self.__x_var_list_rank(x_var_list_i, imps)
                    for var in x_var_list_i_sort:
                        flag += 1
                        plt.subplot(5, 5, flag)
                        x = vals[var]['x']
                        y = vals[var]['y']
                        y_std = vals[var]['y_std']
                        window_len = 5
                        y = SMOOTH().smooth_convolve(y, window_len=window_len)[1:]
                        y_std = SMOOTH().smooth_convolve(y_std, window_len=window_len)[1:]
                        r2 = vals[var]['r2']
                        plt.plot(x, y, color=c)
                        y_std = y_std / 12.
                        up = y + y_std
                        down = y - y_std
                        plt.fill_between(x, up, down, color=c, alpha=0.2)
                        plt.xlabel(var)
                        plt.ylabel('recovery_time')
                        # if i == 0:
                        #     plt.ylim(0,2)
                        # if i == 1:
                        #     plt.ylim(2,7)
                plt.suptitle(title)
                plt.tight_layout()
                # plt.show()
                outpng_path = outpngdir + title + 'pdp.pdf'
                plt.savefig(outpng_path)
                plt.close()

                # exit()
                plt.figure()
                for lc in lc_list:
                    key = f'{humid_arid}-{lc}-{winter_str_list[i]}'
                    if not key in dic:
                        continue
                    pdp = dic[key][0]
                    importance_dic = dic[key][1]
                    importance_list = []
                    r2_list = []
                    if x_var_list_i_sort == 'None':
                        x_var_list_i_sort = self.__x_var_list_rank(x_var_list_i, importance_dic)
                    for var in x_var_list_i_sort:
                        r2 = pdp[var]['r2']
                        r2_list.append(r2)
                        importance = importance_dic[var]
                        importance_list.append(importance)
                    c = Global_vars().lc_colors(lc)
                    plt.scatter(range(len(importance_list)), importance_list, color=c)
                    plt.plot(range(len(importance_list)), r2_list, color=c)
                plt.xticks(range(len(x_var_list_i_sort)), x_var_list_i_sort, rotation=90)
                plt.title(title)
                plt.tight_layout()
                # plt.show()
                outpng_path = outpngdir + title + 'importance.pdf'
                plt.savefig(outpng_path)
                plt.close()

    def check_sos_hist(self):
        df = Global_vars().load_df()
        df = df[df['winter_mark'] == 0]
        df = df[df['current_sos_anomaly'] < 50]
        df = df[df['current_sos_anomaly'] > -50]
        current_sos_anomaly = df['current_sos_anomaly'].tolist()
        print(len(current_sos_anomaly))
        plt.hist(current_sos_anomaly, bins=80)
        plt.show()


class Partial_Dependence_Plots_timing:
    '''
    Ref:
    https://towardsdatascience.com/looking-beyond-feature-importance-37d2807aaaa7
    '''

    def __init__(self):
        self.this_class_arr = result_root_this_script + 'arr/Partial_Dependence_Plots_timing/'
        # self.this_class_tif = result_root_this_script + 'tif/Partial_Dependence_Plots_timing/'
        self.this_class_png = result_root_this_script + 'png/Partial_Dependence_Plots_timing/'
        Tools().mk_dir(self.this_class_arr, force=True)
        # Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        pass

    def run(self):

        self.save_pdp_result()
        # self.plot_result()
        # self.save_pdp_result_humid_arid()
        # self.plot_result_humid_arid()
        # self.check_sos_hist()

    def save_pdp_result(self):
        timing_list = Global_vars().timing_list()
        all_result_dic = {}
        for timing in timing_list:
            out_dir = self.this_class_arr + 'save_pdp_result/{}/'.format(timing)
            T.mk_dir(out_dir, force=True)
            y_var = 'recovery_time'
            df = Global_vars().load_df()
            winter_list = [0, 1]
            winter_str_list = ['no_winter', 'with_winter']
            x_var_list = [Global_vars().vars_RF_no_winter(), Global_vars().vars_RF_with_winter()]
            lc_list = Global_vars().lc_list()
            lat_list = [30, 90]
            df_timing = df[df['timing'] == timing]
            for l in range(len(lat_list)):
                if l + 1 >= len(lat_list):
                    continue
                df_lat = df_timing[df_timing['lat'] > lat_list[l]]
                df_lat = df_lat[df_lat['lat'] < lat_list[l + 1]]
                for lc in lc_list:
                    df_lc = df_lat[df_lat['lc'] == lc]
                    print(len(df_lc))
                    for i in range(len(winter_list)):
                        df_winter = df_lc[df_lc['winter_mark'] == winter_list[i]]
                        df_winter = Global_vars().clean_df(df_winter, winter_str_list[i])

                        key = f'{lat_list[l]}>>{lat_list[l + 1]}-{lc}-{winter_str_list[i]}'
                        print(key)
                        print(len(df_winter))
                        # try:
                        df_new = pd.DataFrame()
                        df_new[x_var_list[i]] = df_winter[x_var_list[i]]
                        df_new[y_var] = df_winter[y_var]
                        # df_new = Global_vars().clean_df(df_new)
                        df_new = df_new.dropna()
                        if len(df_new) < 10:
                            continue
                        pdp_result_dic = self.partial_dependent_plot_arr(df_new, x_var_list[i], y_var)

                        X = df_new[x_var_list[i]]
                        Y = df_new[y_var]
                        clf, importances_dic, r_X_dic, mse, r_model, score, Y_test, y_pred = RF().random_forest_train(X,
                                                                                                                      Y,
                                                                                                                      x_var_list[
                                                                                                                          i])
                        all_result_dic[key] = (pdp_result_dic, importances_dic)
                        # except:
                        #     pass
            T.save_npy(all_result_dic, out_dir + 'pdp_result')

    def partial_dependent_plot_arr(self, df, x_vars, y_var):

        vars_list = x_vars
        selected_features = vars_list
        vars_list1 = copy.copy(selected_features)
        vars_list1.append(y_var)
        XX = df[vars_list1]
        XX = XX.dropna()
        # XX = Global_vars().clean_df(XX)
        vars_list1.remove(y_var)
        X = XX[vars_list1]
        Y = XX[y_var]
        model, r2 = self.train_model(X, Y)
        # print(r2)
        flag = 0
        # plt.figure(figsize=(12, 8))
        # exit()
        result_dic = {}
        for var in tqdm(x_vars):
            flag += 1
            # ax = plt.subplot(5, 5, flag)
            df_partial_plot = self.__get_PDPvalues(var, X, model)
            # print(df_partial_plot)
            # exit()
            ppx = df_partial_plot[var]
            ppy = df_partial_plot['PDs']
            ppy_std = df_partial_plot['PDs_std']
            ppx_smooth = SMOOTH().smooth_convolve(ppx, window_len=11)
            ppy_smooth = SMOOTH().smooth_convolve(ppy, window_len=11)
            ppy_std_smooth = SMOOTH().smooth_convolve(ppy_std, window_len=11)
            result_dic[var] = {
                'x': ppx,
                'y': ppy,
                'y_std': ppy_std,
                'x_smooth': ppx_smooth,
                'y_smooth': ppy_smooth,
                'y_std_smooth': ppy_std_smooth,
                'r2': r2,
            }
            # plt.plot(ppx_smooth,ppy_smooth)
            # plt.show()
        return result_dic

    def save_pdp_result_humid_arid(self):
        out_dir = self.this_class_arr + 'save_pdp_result_humid_arid/'
        T.mk_dir(out_dir)
        y_var = 'recovery_time'
        df = Global_vars().load_df()
        # df = df[df['post_3_month_greater_than_threshold']==1]
        winter_list = [0, 1]
        winter_str_list = ['no_winter', 'with_winter']
        humid_arid_list = ['humid', 'arid', ]
        x_var_list = [Global_vars().vars_RF_no_winter(), Global_vars().vars_RF_with_winter()]
        lc_list = Global_vars().lc_list()
        # lat_list = np.linspace(30, 90, 5)

        # print(lat_list)
        # exit()
        all_result_dic = {}
        for humid_arid in humid_arid_list:
            df_ha = df[df['humid_arid'] == humid_arid]
            for lc in lc_list:
                df_lc = df_ha[df_ha['lc'] == lc]
                for i in range(len(winter_list)):
                    df_winter = df_lc[df_lc['winter_mark'] == winter_list[i]]
                    key = f'{humid_arid}-{lc}-{winter_str_list[i]}'
                    print(key)
                    # try:
                    df_new = pd.DataFrame()
                    df_new[x_var_list[i]] = df_winter[x_var_list[i]]
                    df_new[y_var] = df_winter[y_var]
                    df_new = Global_vars().clean_df(df_new)
                    df_new = df_new.dropna()
                    if len(df_new) < 10:
                        continue
                    pdp_result_dic = self.partial_dependent_plot_arr(df_new, x_var_list[i], y_var)

                    X = df_new[x_var_list[i]]
                    Y = df_new[y_var]
                    clf, importances_dic, r_X_dic, mse, r_model, score, Y_test, y_pred = RF().random_forest_train(X, Y,
                                                                                                                  x_var_list[
                                                                                                                      i])
                    all_result_dic[key] = (pdp_result_dic, importances_dic)
        T.save_npy(all_result_dic, out_dir + 'pdp_result')

    def train_model(self, X, y):
        print(len(X))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42, test_size=0.2)
        # rf = RandomForestClassifier(n_estimators=300, random_state=42)
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=4)
        # rf = LinearRegression()
        rf.fit(X_train, y_train)
        r2 = rf.score(X_test, y_test)
        y_pred = rf.predict(X_test)
        # y_pred = rf.predict(X_train)
        # plt.scatter(y_pred,y_test)
        # print(r2)
        # plt.scatter(y_pred,y_train)
        # plt.show()

        return rf, r2

    def __get_PDPvalues(self, col_name, data, model, grid_resolution=50):
        Xnew = data.copy()
        sequence = np.linspace(np.min(data[col_name]), np.max(data[col_name]), grid_resolution)
        Y_pdp = []
        Y_pdp_std = []
        print('---')
        print(len(data))
        for each in sequence:
            Xnew[col_name] = each
            Y_temp = model.predict(Xnew)
            plt.plot(Y_temp)
            plt.show()
            print(len(Y_temp))
            exit()
            Y_pdp.append(np.mean(Y_temp))
            Y_pdp_std.append(np.std(Y_temp))
        return pd.DataFrame({col_name: sequence, 'PDs': Y_pdp, 'PDs_std': Y_pdp_std})

    def __plot_PDP(self, col_name, data, model):
        df = self.__get_PDPvalues(col_name, data, model)
        plt.rcParams.update({'font.size': 16})
        plt.rcParams["figure.figsize"] = (6, 5)
        fig, ax = plt.subplots()
        # ax.plot(data[col_name], np.zeros(data[col_name].shape)+min(df['PDs'])-1, 'k|', ms=15)  # rug plot
        ax.plot(df[col_name], df['PDs'], lw=2)
        ax.set_ylabel('Recovery time')
        ax.set_xlabel(col_name)
        plt.tight_layout()
        return ax

    def __x_var_list_rank(self, x_var_list, imp_dic):
        imp_list = []
        # print(x_var_list)
        # exit()
        for x in x_var_list:
            imp = imp_dic[x]
            imp_list.append(imp)
        imp_list_sort = np.argsort(imp_list)[::-1]
        # print(imp_list_sort)
        # exit()
        sorted_vals = []
        sorted_var_name = []
        for i in range(len(imp_list)):
            sorted_vals.append(imp_list[imp_list_sort[i]])
            sorted_var_name.append(x_var_list[imp_list_sort[i]])
        return sorted_var_name


    def plot_result(self):
        for timing in Global_vars().timing_list():
            outpngdir = self.this_class_png + 'save_pdp_result/{}/'.format(timing)
            T.mk_dir(outpngdir, force=True)
            # dic_f = self.this_class_arr + '/save_pdp_result/pdp_result.npy'
            dic_f = self.this_class_arr + '/save_pdp_result/{}/pdp_result.npy'.format(timing)
            dic = T.load_npy(dic_f)
            # for key in dic:
            #     print(key)
            # print('----')
            # exit()
            winter_list = [0, 1]
            winter_str_list = ['no_winter', 'with_winter']
            x_var_list = [Global_vars().vars_RF_no_winter(), Global_vars().vars_RF_with_winter()]
            lc_list = Global_vars().lc_list()
            # lat_list = np.linspace(30, 90, 5)
            # lat_list = [30,45,90]
            lat_list = [30, 90]
            for l in range(len(lat_list)):
                if l + 1 >= len(lat_list):
                    continue
                for i in range(len(winter_list)):
                    title = f'{lat_list[l]}>>{lat_list[l + 1]}-{winter_str_list[i]}'
                    x_var_list_i = x_var_list[i]
                    plt.figure(figsize=(12, 8))
                    x_var_list_i_sort = 'None'
                    for lc in lc_list:
                        key = f'{lat_list[l]}>>{lat_list[l + 1]}-{lc}-{winter_str_list[i]}'
                        print(key)

                        if not key in dic:
                            continue
                        # print(lc)
                        # print(key)

                        vals = dic[key][0]
                        imps = dic[key][1]
                        # print(imps)
                        # exit()
                        # print(vals)
                        # exit()
                        flag = 0
                        c = Global_vars().lc_colors(lc)
                        if x_var_list_i_sort == 'None':
                            x_var_list_i_sort = self.__x_var_list_rank(x_var_list_i, imps)
                        for var in x_var_list_i_sort:
                            # print(var)
                            # exit()
                            flag += 1
                            plt.subplot(5, 5, flag)
                            y = vals[var]['y']
                            x = vals[var]['x']
                            y_std = vals[var]['y_std']
                            window_len = 11
                            y = SMOOTH().smooth_convolve(y, window_len=window_len)[1:]
                            y_std = SMOOTH().smooth_convolve(y_std, window_len=window_len)[1:]
                            r2 = vals[var]['r2']

                            if var in ['current_sos_std_anomaly']:
                                y = SMOOTH().smooth_convolve(y, window_len=21)[1:]
                                y_std = y_std / 4.
                            elif var in ['post_VPD', 'post_TMP', 'post_PRE', ]:
                                y = SMOOTH().smooth_convolve(y, window_len=21)[1:]
                                y_std = y_std / 3.
                            else:
                                y_std = y_std / 8.

                            up = y + y_std
                            down = y - y_std
                            plt.plot(x, y, color=c)
                            plt.fill_between(x, up, down, color=c, alpha=0.2)
                            plt.xlabel(var)
                            plt.ylabel('recovery_time')
                            # if i == 0:
                            #     plt.ylim(0,2)
                            # if i == 1:
                            #     plt.ylim(2,7)
                    plt.suptitle(title)
                    plt.tight_layout()
                    # plt.show()
                    outpng_path = outpngdir + title + 'pdp.pdf'
                    plt.savefig(outpng_path)
                    # exit()
                    plt.figure()
                    for lc in lc_list:
                        key = f'{lat_list[l]}>>{lat_list[l + 1]}-{lc}-{winter_str_list[i]}'
                        if not key in dic:
                            continue
                        pdp = dic[key][0]
                        importance_dic = dic[key][1]
                        importance_list = []
                        r2_list = []
                        if x_var_list_i_sort == 'None':
                            x_var_list_i_sort = self.__x_var_list_rank(x_var_list_i, importance_dic)
                        for var in x_var_list_i_sort:
                            r2 = pdp[var]['r2']
                            r2_list.append(r2)
                            importance = importance_dic[var]
                            importance_list.append(importance)
                        c = Global_vars().lc_colors(lc)
                        plt.scatter(range(len(importance_list)), importance_list, color=c)
                        plt.plot(range(len(importance_list)), r2_list, color=c)
                    plt.xticks(range(len(x_var_list_i_sort)), x_var_list_i_sort, rotation=90)
                    plt.title(title)
                    plt.tight_layout()
                    # plt.show()
                    outpng_path = outpngdir + title + 'importance.pdf'
                    plt.savefig(outpng_path)
            T.open_path_and_file(outpngdir)

    def plot_result_humid_arid(self):
        outpngdir = self.this_class_png + 'save_pdp_result_humid_arid/'
        T.mk_dir(outpngdir)
        dic_f = self.this_class_arr + '/save_pdp_result_humid_arid/pdp_result.npy'
        dic = T.load_npy(dic_f)
        winter_list = [0, 1]
        winter_str_list = ['no_winter', 'with_winter']
        x_var_list = [Global_vars().vars_RF_no_winter(), Global_vars().vars_RF_with_winter()]
        lc_list = Global_vars().lc_list()
        # lat_list = np.linspace(30, 90, 5)
        # lat_list = [30,45,90]
        humid_arid_list = ['humid', 'arid']
        for humid_arid in humid_arid_list:
            for i in range(len(winter_list)):
                title = f'{humid_arid}-{winter_str_list[i]}'
                x_var_list_i = x_var_list[i]
                plt.figure(figsize=(12, 8))
                x_var_list_i_sort = 'None'
                for lc in lc_list:
                    key = f'{humid_arid}-{lc}-{winter_str_list[i]}'

                    if not key in dic:
                        continue

                    vals = dic[key][0]
                    imps = dic[key][1]
                    # print(imps)
                    # exit()
                    # print(vals)
                    # exit()
                    flag = 0
                    c = Global_vars().lc_colors(lc)
                    if x_var_list_i_sort == 'None':
                        x_var_list_i_sort = self.__x_var_list_rank(x_var_list_i, imps)
                    for var in x_var_list_i_sort:
                        flag += 1
                        plt.subplot(5, 5, flag)
                        x = vals[var]['x']
                        y = vals[var]['y']
                        y_std = vals[var]['y_std']
                        window_len = 5
                        y = SMOOTH().smooth_convolve(y, window_len=window_len)[1:]
                        y_std = SMOOTH().smooth_convolve(y_std, window_len=window_len)[1:]
                        r2 = vals[var]['r2']
                        plt.plot(x, y, color=c)
                        y_std = y_std / 12.
                        up = y + y_std
                        down = y - y_std
                        plt.fill_between(x, up, down, color=c, alpha=0.2)
                        plt.xlabel(var)
                        plt.ylabel('recovery_time')
                        # if i == 0:
                        #     plt.ylim(0,2)
                        # if i == 1:
                        #     plt.ylim(2,7)
                plt.suptitle(title)
                plt.tight_layout()
                # plt.show()
                outpng_path = outpngdir + title + 'pdp.pdf'
                plt.savefig(outpng_path)
                plt.close()

                # exit()
                plt.figure()
                for lc in lc_list:
                    key = f'{humid_arid}-{lc}-{winter_str_list[i]}'
                    if not key in dic:
                        continue
                    pdp = dic[key][0]
                    importance_dic = dic[key][1]
                    importance_list = []
                    r2_list = []
                    if x_var_list_i_sort == 'None':
                        x_var_list_i_sort = self.__x_var_list_rank(x_var_list_i, importance_dic)
                    for var in x_var_list_i_sort:
                        r2 = pdp[var]['r2']
                        r2_list.append(r2)
                        importance = importance_dic[var]
                        importance_list.append(importance)
                    c = Global_vars().lc_colors(lc)
                    plt.scatter(range(len(importance_list)), importance_list, color=c)
                    plt.plot(range(len(importance_list)), r2_list, color=c)
                plt.xticks(range(len(x_var_list_i_sort)), x_var_list_i_sort, rotation=90)
                plt.title(title)
                plt.tight_layout()
                # plt.show()
                outpng_path = outpngdir + title + 'importance.pdf'
                plt.savefig(outpng_path)
                plt.close()

    def check_sos_hist(self):
        df = Global_vars().load_df()
        df = df[df['winter_mark'] == 0]
        df = df[df['current_sos_anomaly'] < 30]
        df = df[df['current_sos_anomaly'] > -30]
        current_sos_anomaly = df['current_sos_anomaly'].tolist()
        print(len(current_sos_anomaly))
        plt.hist(current_sos_anomaly, bins=80)
        plt.show()


class Correlation_matrix:
    '''
    permutation importance
    '''

    def __init__(self):
        # self.this_class_arr = result_root_this_script + 'arr/Correlation_matrix/'
        # self.this_class_tif = result_root_this_script + 'tif/Correlation_matrix/'
        self.this_class_png = result_root_this_script + 'png/Correlation_matrix/'

        # Tools().mk_dir(self.this_class_arr, force=True)
        # Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        self.df = Global_vars().load_df()

    def run(self):
        self.plot_scatter_lc_timing()
        pass

    def cal_correlation(self, xlist, winter_mark):
        df = self.df
        df = df[df['winter_mark'] == winter_mark]
        lc_list = Global_vars().lc_list()
        timing_list = Global_vars().timing_list()
        plot_result_dic_all = {}

        for timing in timing_list:
            df_timing = df[df['timing'] == timing]
            pdp_result_f = T.path_join(Partial_Dependence_Plots_timing().this_class_arr,
                                       'save_pdp_result', timing, 'pdp_result.npy'
                                       )
            result_dic = T.load_npy(pdp_result_f)

            for lc in lc_list:
                df_lc = df_timing[df_timing['lc'] == lc]
                key = f'30>>90-{lc}-no_winter'
                imp_result = result_dic[key][1]
                recovery_time = df_lc['recovery_time']
                recovery_time = recovery_time.tolist()
                plot_result_dic = {}
                for var_i in xlist:
                    val = df_lc[var_i]
                    # imp = imp_result[var_i]
                    val = val.tolist()
                    r, c = T.nan_correlation(val, recovery_time)
                    plot_result_dic[var_i] = [r]
                key_all = f'{timing}-{lc}'
                plot_result_dic_all[key_all] = plot_result_dic
        return plot_result_dic_all

    def square_colormap(self, n, m):

        # n=30
        r_arr = []
        for i in range(n):
            arr = np.linspace(0, n - 1, m)
            arr = arr / (n - 1.)
            r_arr.append(arr)

        g_arr = []
        for i in range(n):
            arr = np.linspace(i, n - 1 - i, m)
            arr = arr / (n - 1.)
            g_arr.append(arr)

        b_arr = []
        for i in range(n):
            arr = np.linspace(n - 1 - i, 0, m)
            arr = arr / (n - 1.)
            b_arr.append(arr)
        r_arr = np.array(r_arr)
        g_arr = np.array(g_arr)
        b_arr = np.array(b_arr)
        c_dic = {}
        for i in range(len(r_arr)):
            for j in range(len(r_arr[0])):
                c_val = (r_arr[i][j], g_arr[i][j], b_arr[i][j], 1.)
                c_dic[(i + 1, j + 1)] = c_val
            # pass

        # print c_dic
        # exit()

        # plt.figure()
        # plt.imshow(r_arr)
        # plt.figure()
        # plt.imshow(g_arr)
        # plt.figure()
        # plt.imshow(b_arr)
        # plt.show()

        for i in tqdm(list(range(n))):
            for j in range(m):
                c_pos = (i, j)
                r = r_arr[c_pos]
                g = g_arr[c_pos]
                b = b_arr[c_pos]
                c = np.array([r, g, b, 1.])
                # print c
                plt.scatter(i, m - j, color=c, marker='s', s=100)
        plt.axis('off')
        plt.axis('equal')
        # plt.show()

        return c_dic

        pass

    def sort_dic(self, indic):

        val_list = []
        dic_label = []
        for key in indic:
            dic_label.append(key)
            val_list.append(indic[key])
        sort_indx = np.argsort(val_list)
        val_list = np.array(val_list)
        dic_label = np.array(dic_label)
        sort_dic = dict(list(zip(dic_label[sort_indx], list(range(1, len(dic_label) + 1)))))
        return sort_dic
        pass

    def plot_scatter_square_lc_timing(self, dff):

        rf_result_f = self.permutation_result_arr + 'lc_timing/dest_var_Y/result_dic.npy'
        # var_list = Global_vars().valid_vars_post()
        # rf_result_f = self.permutation_result_arr + 'lc_timing/dest_var_lag/result_dic.npy'
        # var_list = Global_vars().valid_vars_pre()
        suqare_color_map = self.square_colormap(n=8, m=6)
        variables = \
            '''
            sand
            lag
            water_balance
            sos
            PRE_pre_6_mean
            NDVI_pre_6_mean
            '''
        dest_var = 'Y'
        var_list = variables.split()

        rf_result_dic = T.load_npy(rf_result_f)

        for lc in Global_vars().landuse_list():
            ## 1 rebuild dic ##########################
            timing_key_dic = {}
            for t in Global_vars().timing_list():
                key = '{}_{}'.format(t, lc)
                result = rf_result_dic[key]
                timing_key_dic[t] = result

            var_key_dic = {}
            for var in var_list:
                temp_dic = {}
                for t in Global_vars().timing_list():
                    key = '{}_{}'.format(t, lc)
                    # print var
                    result = rf_result_dic[key][0][var]
                    temp_dic[t] = result
                var_key_dic[var] = temp_dic
            ### 2 sort dic #######################

            timing_sort_dic = {}
            for t in timing_key_dic:
                dic_i = timing_key_dic[t][0]
                sort_dic = self.sort_dic(dic_i)
                timing_sort_dic[t] = sort_dic

            var_sort_dic = {}
            for var in var_key_dic:
                dic_i = var_key_dic[var]
                sort_dic = self.sort_dic(dic_i)
                var_sort_dic[var] = sort_dic

            # print timing_sort_dic
            # print var_sort_dic

            grid_dic = {}
            for var in var_list:
                for t in Global_vars().timing_list():
                    key = '{}_{}'.format(var, t)
                    var_val = var_sort_dic[var][t]
                    timing_val = timing_sort_dic[t][var]
                    # print var_val
                    # print timing_val
                    grid_dic[key] = (var_val, timing_val)

            fig1, ax1 = plt.subplots()
            fig2, ax2 = plt.subplots()

            y = 0
            for t in Global_vars().timing_list():
                x = 0
                for var in var_list:
                    key = '{}_{}'.format(var, t)
                    rank = grid_dic[key]
                    # suqare_color_map
                    c = suqare_color_map[rank]
                    ax1.scatter(x, y, color=c, marker='s', s=800)
                    ax2.scatter(x, y, color='black', marker='o', s=4800 * rf_result_dic['{}_{}'.format(t, lc)][0][var])
                    x += 1
                y += 1

            # plt.axis('equal')
            # plt.xticks(range(len(var_list)),var_list,rotation=90)
            # plt.yticks(range(8),Global_vars().timing_list())
            # plt.tight_layout()
            ax1.set_title(lc)
            ax2.set_title(lc)
            plt.show()

        df = T.load_df(dff)
        print(df.columns)

    def __r_to_color(self, importance, xmin=-0.3, xmax=0.3, color_class=256):
        if np.isnan(importance):
            importance = 0.
        importance_range = np.linspace(xmin, xmax, color_class)
        if importance < xmin:
            pos_indx = 0
        elif importance > xmax:
            pos_indx = len(importance_range) - 1
        else:
            pos_indx = int(round(((importance - xmin) / (xmax - xmin) * len(importance_range)), 0)) - 1

        cmap = sns.diverging_palette(236, 0, s=99, l=50, n=color_class, center="light")
        c = cmap[pos_indx]
        return c
        pass

    def plot_scatter_lc_timing(self):
        outpdfdir = T.path_join(self.this_class_png, 'plot_scatter_lc_timing')
        T.mk_dir(outpdfdir)
        T.open_path_and_file(outpdfdir)
        # exit()
        # var_list = Global_vars().valid_vars_post()
        # rf_result_f = self.permutation_result_arr + 'lc_timing/dest_var_lag/result_dic.npy'
        # var_list = Global_vars().valid_vars_pre()
        # print partial_corr_result_dic
        # exit()
        var_list_0 = Global_vars().vars_RF_no_winter()
        var_list_1 = Global_vars().vars_RF_with_winter()
        var_list_all = [var_list_0, var_list_1]
        # print(var_list_0)
        # print(var_list_1)
        # exit()
        for winter in [0, 1]:
            var_list = var_list_all[winter]
            var_list = var_list[::-1]
            result_all_dic = self.cal_correlation(var_list, winter)
            df = T.dic_to_df(result_all_dic)
            T.print_head_n(df, 10)
            T.save_df(df, join(outpdfdir, f'winter_{winter}.df'))
            continue
            exit()
            lc_list = Global_vars().lc_list()
            timing_list = Global_vars().timing_list()
            for timing in timing_list:
                plt.figure(figsize=(10, 10))
                y = 0
                for lc in lc_list:
                    key = f'{timing}-{lc}'
                    result_dic = result_all_dic[key]
                    x = 0
                    for var in var_list:
                        # print(var)
                        if not var in result_dic:
                            imp = np.nan
                        else:
                            r = result_dic[var][0]
                            # imp = result_dic[var][1]
                            imp = -999
                            color = self.__r_to_color(r)
                            # print color
                            # exit()
                            # plt.scatter(y,x,s=imp*100,marker='o',c='black')
                            # plt.scatter(y, x, marker='o', alpha=1, c='None', zorder=99, edgecolors='black', s=imp*100)
                            if imp == 1:
                                size = 2
                            elif imp == 2:
                                size = 20
                            elif imp == 3:
                                size = 60
                            elif imp == 4:
                                size = 100
                            elif imp == 5:
                                size = 140
                            elif imp == 6:
                                size = 200
                            else:
                                size = imp * 300
                            # if p < 0.05:
                            # plt.scatter(y, x, marker='o', alpha=1, c='None', zorder=99, edgecolors='black', s=size, linewidths=2)
                            plt.scatter(y, x, marker='s', alpha=1, color=color, zorder=0, s=200)
                            # else:
                            # plt.scatter(y, x, marker='s', alpha=1, color=color, zorder=0, s=200)
                            # plt.scatter(y, x, marker='x', alpha=1, zorder=99, color='black', s=50, linewidth=2)
                        x += 1
                    y += 1
                    plt.yticks(list(range(len(var_list))), var_list)
                    plt.xticks(list(range(len(lc_list))), lc_list, rotation=90)
                    plt.axis('equal')
                    plt.title(timing + f' {winter}')
                    plt.tight_layout()
                outpath = T.path_join(outpdfdir, timing + f' {winter}.pdf')
                plt.savefig(outpath, dpi=300)
                plt.close()
                # plt.show()


class Sensitivity:

    def __init__(self):
        # self.datadir = '/Volumes/NVME2T/project05_sensitivity/data'
        self.datadir = data_root
        pass

    def run(self):
        # self.resample()
        # self.per_pix_CSIF_005()
        # self.anomaly_CSIF_005()
        # self.per_pix_SPEI03_2003_2015()
        # self.per_pix_CWD_2003_2015()
        # self.SPEI3_CSIF_correlation()
        # self.CWD_CSIF_correlation()
        # self.VOD_CWD_correlation()
        # self.VOD_SPEI_correlation()
        # self.SPEI36912_NDVI_correlation()
        # self.SPEI36912_VOD_correlation()
        self.SPEI36912_CSIF_correlation()
        pass

    def resample_CSIF_005(self):
        fdir = T.path_join(self.datadir, 'CSIF_005', 'tif')
        outdir = T.path_join(self.datadir, 'CSIF_005', 'tif_05')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            fpath = T.path_join(fdir, f)
            outpath = T.path_join(outdir, f)
            ToRaster().resample_reproj(fpath, outpath, res=0.5)

    def per_pix_CSIF_005(self):
        fdir = T.path_join(self.datadir, 'CSIF_005', 'tif_05')
        outdir = T.path_join(self.datadir, 'CSIF_005', 'per_pix')
        T.mk_dir(outdir)
        Pre_Process().data_transform(fdir, outdir)
        pass

    def per_pix_SPEI03_2003_2015(self):
        '/Volumes/NVME2T/project05_sensitivity/data/SPEI/spei0315'
        fdir = T.path_join(self.datadir, 'SPEI', 'spei0315', 'tif')
        outdir = T.path_join(self.datadir, 'SPEI', 'spei0315', 'per_pix')
        T.mk_dir(outdir)
        Pre_Process().data_transform(fdir, outdir)
        pass

    def per_pix_CWD_2003_2015(self):
        cwd_f = T.path_join(self.datadir, 'CWD', 'per_pix', 'CWD.npy')
        outdir = T.path_join(self.datadir, 'CWD', 'per_pix_2003_2015')
        outf = T.path_join(outdir, 'CWD')
        T.mk_dir(outdir)
        cwd_dic = T.load_npy(cwd_f)

        total_month = []
        for y in list(range(1982, 2016)):
            for m in range(1, 13):
                date = (y, m)
                total_month.append(date)
        selected_month = []
        for y in list(range(2003, 2016)):
            for m in range(1, 13):
                date = (y, m)
                selected_month.append(date)
        pick_index = []
        for i in selected_month:
            indx = total_month.index(i)
            pick_index.append(indx)

        cwd_dic_2003_2015 = {}
        for pix in tqdm(cwd_dic):
            vals = cwd_dic[pix]
            picked_vals = T.pick_vals_from_1darray(vals, pick_index)
            cwd_dic_2003_2015[pix] = picked_vals
        T.save_npy(cwd_dic_2003_2015, outf)

    def anomaly_CSIF_005(self):
        fdir = T.path_join(self.datadir, 'CSIF_005', 'per_pix')
        outdir = T.path_join(self.datadir, 'CSIF_005', 'per_pix_anomaly')
        T.mk_dir(outdir)
        Pre_Process().cal_anomaly(fdir, outdir)

        pass

    def SPEI3_CSIF_correlation(self):
        outdir = T.path_join(result_root_this_script, 'tif', 'Sensitivity', 'SPEI3_CSIF_correlation')
        outf = T.path_join(outdir, 'CSIF_SPEI3_corr.tif')
        T.mk_dir(outdir, force=True)
        df = Global_vars().load_df()
        pix_list = T.get_df_unique_val_list(df, 'pix')
        pix_list = set(pix_list)
        spei_dir = T.path_join(self.datadir, 'SPEI', 'spei0315', 'per_pix')
        csif_dir = T.path_join(self.datadir, 'CSIF_005', 'per_pix_anomaly')
        spei_dic = T.load_npy_dir(spei_dir)
        csif_dic = T.load_npy_dir(csif_dir)
        spatial_dic = {}
        for pix in tqdm(csif_dic):
            if not pix in pix_list:
                continue
            csif = csif_dic[pix]
            spei = spei_dic[pix]
            try:
                r, p = stats.pearsonr(csif, spei)
            except:
                print(len(csif))
                print(len(spei))
                r = np.nan
            spatial_dic[pix] = r

        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        DIC_and_TIF().arr_to_tif(arr, outf)

        # plt.imshow(arr,vmin=-0.5,vmax=0.5)
        # plt.colorbar()
        # plt.show()

        pass

    def CWD_CSIF_correlation(self):
        outdir = T.path_join(result_root_this_script, 'tif', 'Sensitivity', 'CWD_CSIF_correlation')
        outf = T.path_join(outdir, 'CSIF_CWD_corr.tif')
        T.mk_dir(outdir, force=True)
        df = Global_vars().load_df()
        pix_list = T.get_df_unique_val_list(df, 'pix')
        pix_list = set(pix_list)
        spei_dir = T.path_join(self.datadir, 'CWD', 'per_pix_2003_2015')
        csif_dir = T.path_join(self.datadir, 'CSIF_005', 'per_pix_anomaly')
        spei_dic = T.load_npy_dir(spei_dir)
        csif_dic = T.load_npy_dir(csif_dir)
        spatial_dic = {}
        for pix in tqdm(csif_dic):
            if not pix in pix_list:
                continue
            csif = csif_dic[pix]
            spei = spei_dic[pix]
            try:
                r, p = stats.pearsonr(csif, spei)
            except:
                print(len(csif))
                print(len(spei))
                r = np.nan
            spatial_dic[pix] = r

        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        DIC_and_TIF().arr_to_tif(arr, outf)

        # plt.imshow(arr,vmin=-0.5,vmax=0.5)
        # plt.colorbar()
        # plt.show()
        pass

    def VOD_CWD_correlation(self):
        outdir = T.path_join(result_root_this_script, 'tif', 'Sensitivity', 'VOD_CWD_correlation')
        outf = T.path_join(outdir, 'VOD_CWD_corr.tif')
        T.mk_dir(outdir, force=True)
        df = Global_vars().load_df()
        pix_list = T.get_df_unique_val_list(df, 'pix')
        pix_list = set(pix_list)
        spei_dir = T.path_join(self.datadir, 'CWD', 'per_pix_2003_2015')
        csif_dir = T.path_join(data_root, 'VOD', 'per_pix_2003_2015')
        spei_dic = T.load_npy_dir(spei_dir)
        csif_dic = T.load_npy_dir(csif_dir)
        spatial_dic = {}
        for pix in tqdm(csif_dic):
            if not pix in pix_list:
                continue
            csif = csif_dic[pix]
            spei = spei_dic[pix]
            try:
                r, p = T.nan_correlation(csif, spei)
            except:
                print(len(csif))
                print(len(spei))
                r = np.nan
            spatial_dic[pix] = r

        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        DIC_and_TIF().arr_to_tif(arr, outf)

        # plt.imshow(arr,vmin=-0.5,vmax=0.5)
        # plt.colorbar()
        # plt.show()
        pass

    def VOD_SPEI_correlation(self):
        outdir = T.path_join(result_root_this_script, 'tif', 'Sensitivity', 'VOD_SPEI_correlation')
        outf = T.path_join(outdir, 'VOD_SPEI_corr.tif')
        T.mk_dir(outdir, force=True)
        df = Global_vars().load_df()
        pix_list = T.get_df_unique_val_list(df, 'pix')
        pix_list = set(pix_list)
        spei_dir = T.path_join(self.datadir, 'SPEI', 'spei0315', 'per_pix')
        csif_dir = T.path_join(data_root, 'VOD', 'per_pix_2003_2015')
        spei_dic = T.load_npy_dir(spei_dir)
        csif_dic = T.load_npy_dir(csif_dir)
        spatial_dic = {}
        for pix in tqdm(csif_dic):
            if not pix in pix_list:
                continue
            csif = csif_dic[pix]
            spei = spei_dic[pix]
            try:
                r, p = T.nan_correlation(csif, spei)
            except:
                print(len(csif))
                print(len(spei))
                r = np.nan
            spatial_dic[pix] = r

        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        DIC_and_TIF().arr_to_tif(arr, outf)

        # plt.imshow(arr,vmin=-0.5,vmax=0.5)
        # plt.colorbar()
        # plt.show()

        pass

    def SPEI36912_CSIF_correlation(self):
        outdir = T.path_join(result_root_this_script, 'tif', 'Sensitivity', 'SPEI36912_CSIF_correlation')
        for scale in [3, 6, 9, 12]:
            print(scale)
            outf = T.path_join(outdir, f'CSIF_SPEI{scale:02d}_corr.tif')
            # print(outf)
            # exit()
            T.mk_dir(outdir, force=True)
            df = Global_vars().load_df_all()
            pix_list = T.get_df_unique_val_list(df, 'pix')
            pix_list = set(pix_list)
            # spei_f = T.path_join(self.datadir, 'SPEI', 'compose_spei_n_to_one_file_2003_2015',f'spei{scale:02d}.npy')
            # spei_dir = '/Volumes/NVME2T/project05_sensitivity/data/SPEI/spei0315/spei{:02d}/per_pix/'.format(scale)
            spei_dir = '/Volumes/Ugreen_4T_25/project05_redo/data/SPEI/per_pix_2003_2015/spei{:02d}/'.format(scale)
            # csif_dir = T.path_join(self.datadir, 'VOD', 'per_pix_2001_2015_anomaly')
            # csif_dir = T.path_join(self.datadir, 'VOD', 'per_pix_2001_2015')
            csif_dir = '/Volumes/NVME2T/project05_sensitivity/data/CSIF_005/per_pix'
            spei_dic = T.load_npy_dir(spei_dir)
            # spei_dic = T.load_npy(spei_f)
            csif_dic = T.load_npy_dir(csif_dir)
            spatial_dic = {}
            for pix in tqdm(csif_dic):
                if not pix in pix_list:
                    continue
                if not pix in spei_dic:
                    continue
                csif = csif_dic[pix]
                spei = spei_dic[pix]
                try:
                    # r,p = stats.pearsonr(csif,spei)
                    r, p = T.nan_correlation(csif, spei)
                except:
                    print(len(csif))
                    print(len(spei))
                    print(csif)
                    print(spei)
                    # exit()
                    r = np.nan
                spatial_dic[pix] = r

            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            DIC_and_TIF().arr_to_tif(arr, outf)

        # plt.imshow(arr,vmin=-0.5,vmax=0.5)
        # plt.colorbar()
        # plt.show()
        pass

    def SPEI36912_NDVI_correlation(self):
        outdir = T.path_join(result_root_this_script, 'tif', 'Sensitivity', 'SPEI36912_NDVI_correlation')
        for scale in [3, 6, 9, 12]:
            print(scale)
            outf = T.path_join(outdir, f'NDVI_SPEI{scale:02d}_corr.tif')
            # print(outf)
            # exit()
            T.mk_dir(outdir, force=True)
            df = Global_vars().load_df_all()
            pix_list = T.get_df_unique_val_list(df, 'pix')
            pix_list = set(pix_list)
            spei_f = T.path_join(self.datadir, 'SPEI', 'compose_spei_n_to_one_file', f'spei{scale:02d}.npy')
            csif_dir = T.path_join(self.datadir, 'NDVI', 'per_pix_clean_anomaly_smooth')
            spei_dic = T.load_npy(spei_f)
            csif_dic = T.load_npy_dir(csif_dir)
            spatial_dic = {}
            for pix in tqdm(csif_dic):
                if not pix in pix_list:
                    continue
                if not pix in spei_dic:
                    continue
                csif = csif_dic[pix]
                spei = spei_dic[pix]
                try:
                    r, p = stats.pearsonr(csif, spei)
                except:
                    print(len(csif))
                    print(len(spei))
                    r = np.nan
                spatial_dic[pix] = r

            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            DIC_and_TIF().arr_to_tif(arr, outf)

        # plt.imshow(arr,vmin=-0.5,vmax=0.5)
        # plt.colorbar()
        # plt.show()
        pass

    def SPEI36912_VOD_correlation(self):
        outdir = T.path_join(result_root_this_script, 'tif', 'Sensitivity', 'SPEI36912_VOD_correlation')
        for scale in [3, 6, 9, 12]:
            print(scale)
            outf = T.path_join(outdir, f'VOD_SPEI{scale:02d}_corr.tif')
            # print(outf)
            # exit()

            T.mk_dir(outdir, force=True)
            df = Global_vars().load_df_all()
            pix_list = T.get_df_unique_val_list(df, 'pix')
            pix_list = set(pix_list)
            # spei_f = T.path_join(self.datadir, 'SPEI', 'compose_spei_n_to_one_file_2003_2015',f'spei{scale:02d}.npy')
            # spei_dir = '/Volumes/NVME2T/project05_sensitivity/data/SPEI/spei0315/spei{:02d}/per_pix/'.format(scale)
            spei_dir = '/Volumes/Ugreen_4T_25/project05_redo/data/SPEI/per_pix_2003_2015/spei{:02d}/'.format(scale)
            # csif_dir = T.path_join(self.datadir, 'VOD', 'per_pix_2001_2015_anomaly')
            # csif_dir = T.path_join(self.datadir, 'VOD', 'per_pix_2001_2015')
            csif_dir = T.path_join(data_root, 'VOD_AMSR', 'per_pix_2003_2015_anomaly', 'D')
            spei_dic = T.load_npy_dir(spei_dir)
            # spei_dic = T.load_npy(spei_f)
            csif_dic = T.load_npy_dir(csif_dir)
            spatial_dic = {}
            for pix in tqdm(csif_dic):
                if not pix in pix_list:
                    continue
                if not pix in spei_dic:
                    continue
                csif = csif_dic[pix]
                spei = spei_dic[pix]
                try:
                    # r,p = stats.pearsonr(csif,spei)
                    r, p = T.nan_correlation(csif, spei)
                except:
                    print(len(csif))
                    print(len(spei))
                    print(csif)
                    print(spei)
                    # exit()
                    r = np.nan
                spatial_dic[pix] = r

            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            DIC_and_TIF().arr_to_tif(arr, outf)

        # plt.imshow(arr,vmin=-0.5,vmax=0.5)
        # plt.colorbar()
        # plt.show()
        pass


class Statistics:
    def __init__(self):

        pass

    def run(self):
        # self.unify_glc2000()
        self.count_rf_sample_size()
        # self.count_NH_vegetated_pix_ratio()
        # self.correlation_ratio_of_latitude_50NH()
        # self.phenology_change_drought_non_drought()
        pass

    def unify_glc2000(self):
        tif = T.path_join(data_root, 'landcover', 'Tiff', 'glc2000_0_5.tif')
        outtif = T.path_join(data_root, 'landcover', 'Tiff', 'glc2000_0_5_unify.tif')
        DIC_and_TIF().unify_raster(tif, outtif)
        pass

    def count_rf_sample_size(self):
        Partial_Dependence_Plots().count_sample_size_in_different_models()
        pass

    def count_NH_valid_pix_ratio(self):
        dff = Global_vars().dff
        df = T.load_df(dff)
        lat = 0
        df = df[df['lat'] > lat]
        land_tif = T.path_join(this_root, 'conf', 'land.tif')
        land_arr = ToRaster().raster2array(land_tif)[0]
        NH_land_arr = land_arr[:(90 - lat) * 2]
        NH_total_pix_num = 0.
        for i in NH_land_arr:
            for j in i:
                if j < -9999:
                    continue
                NH_total_pix_num += 1.
        valid_pix_list = T.get_df_unique_val_list(df, 'pix')
        valid_pix_list_num = len(valid_pix_list)
        ratio = valid_pix_list_num / NH_total_pix_num
        print(ratio)

    def count_NH_vegetated_pix_ratio(self):
        dff = Global_vars().dff
        df = T.load_df(dff)
        lat = 30
        df = df[df['lat'] > lat]
        land_tif = T.path_join(data_root, 'landcover', 'Tiff', 'glc2000_0_5_unify.tif')
        land_arr = ToRaster().raster2array(land_tif)[0]
        vegetated_class = list(range(1, 19))
        NH_land_arr = land_arr[:(90 - lat) * 2]
        NH_total_pix_num = 0.
        for i in NH_land_arr:
            for j in i:
                if j in vegetated_class:
                    NH_total_pix_num += 1.
        # exit()
        valid_pix_list = T.get_df_unique_val_list(df, 'pix')
        valid_pix_list_num = len(valid_pix_list)
        ratio = valid_pix_list_num / NH_total_pix_num
        print(ratio)

    def correlation_ratio_of_latitude_50NH(self):
        df = Global_vars().load_df_all()
        df = df[df['lat'] > 50]
        # df = df[df['winter_mark'] == 0]
        # df = df[df['winter_mark'] == 1]
        total_events = 0
        shorter_event = 0
        for i, row in tqdm(df.iterrows(), total=len(df)):
            recovery_time = row['recovery_time']
            total_events += 1
            if recovery_time < 3:
                shorter_event += 1
        ratio = shorter_event / total_events
        print(ratio)

        pass

    def __kernel_phenology_change_drought_non_drought(self, vals_reshape):
        vals_reshape_T = vals_reshape.T
        mean_list = []
        for i in vals_reshape_T:
            mean_list.append(np.mean(i))
        return mean_list
        pass

    def phenology_change_drought_non_drought(self):
        events_f = T.path_join(Main_flow_Pick().this_class_arr, 'compose_cwd_spei_events/events.npy')
        events_dic = T.load_npy(events_f)
        # df = Global_vars().load_df_all()
        df = Global_vars().load_df()
        # df = df[df['lc']=='Grasslands']
        # df = df[df['lc']=='deciduous']
        # hants_smooth_fdir = T.path_join(result_root_this_script,'arr/Main_flow_Early_Peak_Late_Dormant/hants_smooth_annual')
        ndvi_bi_weekly_dir = T.path_join(data_root, 'NDVI/tif_05deg_bi_weekly_per_pix')
        pix_list = T.get_df_unique_val_list(df, 'pix')
        dic = T.load_npy_dir(ndvi_bi_weekly_dir, condition='')
        normal_all_mean_list = []
        drought_all_mean_list = []
        for pix in tqdm(pix_list):
            if not pix in dic:
                continue
            events = events_dic[pix]
            drought_year = []
            for timing, event in events:
                for m in event:
                    year = m // 12
                    drought_year.append(year)
            drought_year = list(set(drought_year))
            drought_year.sort()

            vals = dic[pix]
            vals = np.array(vals)
            vals[vals < 0] = 0
            vals_reshape = np.reshape(vals, (-1, 24))
            normal_year = []
            for i in range(len(vals_reshape)):
                if i in drought_year:
                    continue
                else:
                    normal_year.append(i)
            vals_reshape_drought = T.pick_vals_from_1darray(vals_reshape, drought_year)
            vals_reshape_normal = T.pick_vals_from_1darray(vals_reshape, normal_year)
            # plt.imshow(vals_reshape)
            drought_year_mean = self.__kernel_phenology_change_drought_non_drought(vals_reshape_drought)
            normal_year_mean = self.__kernel_phenology_change_drought_non_drought(vals_reshape_normal)
            normal_all_mean_list.append(normal_year_mean)
            drought_all_mean_list.append(drought_year_mean)
            # plt.plot(mean_list,color='gray',alpha=0.005)
        normal_all_mean_list = np.array(normal_all_mean_list)
        drought_all_mean_list = np.array(drought_all_mean_list)
        normal_all_mean_list_T = normal_all_mean_list.T
        drought_all_mean_list_T = drought_all_mean_list.T
        normal_all_mean_list_T_mean = []
        drought_all_mean_list_T_mean = []
        normal_all_mean_list_T_std = []
        drought_all_mean_list_T_std = []
        for i in normal_all_mean_list_T:
            normal_all_mean_list_T_mean.append(np.mean(i))
            normal_all_mean_list_T_std.append(np.std(i))

        for i in drought_all_mean_list_T:
            drought_all_mean_list_T_mean.append(np.mean(i))
            drought_all_mean_list_T_std.append(np.std(i))
        normal_all_mean_list_T_mean = np.array(normal_all_mean_list_T_mean)
        drought_all_mean_list_T_mean = np.array(drought_all_mean_list_T_mean)
        plt.plot(normal_all_mean_list_T_mean, color='b')
        plt.plot(drought_all_mean_list_T_mean, color='r')
        # plt.plot(drought_all_mean_list_T_mean - normal_all_mean_list_T_mean,color='r')

        # Plot_line().plot_line_with_gradient_error_band(range(len(normal_all_mean_list_T_mean)),normal_all_mean_list_T_mean,normal_all_mean_list_T_std,c='b')
        # Plot_line().plot_line_with_gradient_error_band(range(len(drought_all_mean_list_T_mean)),drought_all_mean_list_T_mean,drought_all_mean_list_T_std,c='r')
        plt.show()

        pass


class Main_flow_shui_re:

    def __init__(self):
        self.this_class_png = result_root_this_script + 'png/Main_flow_shui_re/'
        Tools().mk_dir(self.this_class_png, force=True)
        pass

    def run(self):
        # self.plot_MAP()
        out_png_dir = self.this_class_png + '3*3_shuire/'
        T.mk_dir(out_png_dir)
        # func_list = [self.sensitivity,self.lag,self.recovery]
        # fig_name_var = ['sensitivity','lag','recovery']
        func_list = [self.recovery]
        fig_name_var = ['recovery']
        for lc in Global_vars().lc_list():
            for i, func in enumerate(func_list):
                print(lc, i)
                func(lc)
                fig_path = out_png_dir + '{}__{}.pdf'.format(lc, fig_name_var[i])
                plt.show()
                # plt.savefig(fig_path,dpi=300)
                # plt.close()
            # self.sensitivity(lc)
            # self.lag(lc)
        #     self.recovery(lc)
        # plt.show()

        pass

    def __divide_MA(self, arr, min_v=None, max_v=None, step=None, n=None):
        if min_v == None:
            min_v = np.min(arr)
        if max_v == None:
            max_v = np.max(arr)
        if n == None:
            d = np.arange(start=min_v, step=step, stop=max_v)
        if step == None:
            d = np.linspace(min_v, max_v, num=n)

        # print d
        # exit()
        # if step >= 10:
        #     d_str = []
        #     for i in d:
        #         d_str.append('{}'.format(int(round(i*12.,0))))
        # else:
        d_str = []
        for i in d:
            d_str.append('{}'.format(int(round(i, 0))))
        # print d_str
        # exit()
        return d, d_str
        pass

    def sensitivity(self, lc):
        dff = Global_vars().dff
        df = T.load_df(dff)
        df = df[df['lc'] == lc]
        # df = df.drop_duplicates(subset=['pix'])
        vals_dic = DIC_and_TIF().void_spatial_dic()
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            # sens = row.lag
            sens = row.correlation
            vals_dic[pix].append(sens)
        MAT_series = df.MA_TMP
        MAP_series = df.MA_PRE * 12.
        df['MA_PRE'] = MAP_series
        d_mat, mat_str = self.__divide_MA(MAT_series, step=1)
        d_map, map_str = self.__divide_MA(MAP_series, min_v=0, max_v=2001, step=100)
        # print map_str
        # print d_map
        # exit()

        shuire_matrix = []
        x = []
        y = []
        z = []
        for t in tqdm(list(range(len(d_mat)))):
            if t + 1 >= len(d_mat):
                continue
            df_t = df[df['MA_TMP'] > d_mat[t]]
            df_t = df_t[df_t['MA_TMP'] < d_mat[t + 1]]
            temp = []
            for p in range(len(d_map)):
                if p + 1 >= len(d_map):
                    continue
                df_p = df_t[df_t['MA_PRE'] > d_map[p]]
                df_p = df_p[df_p['MA_PRE'] < d_map[p + 1]]
                pixs = df_p.pix

                if len(pixs) != 0:
                    vals = []
                    for pix in pixs:
                        val = vals_dic[pix]
                        val = np.nanmean(val)
                        vals.append(val)
                    val_mean = np.nanmean(vals)
                else:
                    val_mean = np.nan
                temp.append(val_mean)
                x.append(d_map[p])
                y.append(d_mat[t])
                z.append(val_mean)
            shuire_matrix.append(temp)
        # plt.imshow(shuire_matrix,vmin=-0.3,vmax=0.3)
        # plt.imshow(shuire_matrix)
        # plt.xticks(range(len(shuire_matrix[0])),map_str,rotation=90)
        # plt.yticks(range(len(shuire_matrix)),mat_str,rotation=0)

        plt.figure(figsize=(4, 6))
        cmap = 'RdBu_r'
        plt.scatter(x, y, c=z, marker='s', cmap=cmap, norm=None, vmin=-0.3, vmax=0.3)
        plt.gca().invert_yaxis()
        plt.subplots_adjust(
            top=0.88,
            bottom=0.11,
            left=0.12,
            right=0.90,
            hspace=0.2,
            wspace=0.2
        )
        plt.title('Lag (months)')
        plt.colorbar()
        plt.xlabel('MAP (mm)')
        plt.ylabel('MAT (°C)')
        plt.title('{} Sensitivity'.format(lc))
        # plt.show()

    def sensitivity_scatter(self):
        dff = Global_vars().dff
        df = T.load_df(dff)
        # df = df.drop_duplicates(subset=['pix'])
        vals_dic = DIC_and_TIF().void_spatial_dic()
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            # sens = row.lag
            sens = row.correlation
            vals_dic[pix].append(sens)
        MAT_series = df.MA_TMP
        MAP_series = df.MA_PRE
        d_mat, mat_str = self.__divide_MA(MAT_series, step=5)
        d_map, map_str = self.__divide_MA(MAP_series, step=30)
        # print map_str
        # print d_map
        # exit()

        shuire_matrix = []
        for t in tqdm(list(range(len(d_mat)))):
            if t + 1 >= len(d_mat):
                continue
            df_t = df[df['MA_TMP'] > d_mat[t]]
            df_t = df_t[df_t['MA_TMP'] < d_mat[t + 1]]
            temp = []
            for p in range(len(d_map)):
                if p + 1 >= len(d_map):
                    continue
                df_p = df_t[df_t['MA_PRE'] > d_map[p]]
                df_p = df_p[df_p['MA_PRE'] < d_map[p + 1]]
                pixs = df_p.pix

                if len(pixs) != 0:
                    vals = []
                    for pix in pixs:
                        val = vals_dic[pix]
                        val = np.nanmean(val)
                        vals.append(val)
                    val_mean = np.nanmean(vals)
                else:
                    val_mean = np.nan
                temp.append(val_mean)
            shuire_matrix.append(temp)
        # plt.imshow(shuire_matrix,vmin=2,vmax=5)
        plt.imshow(shuire_matrix)
        plt.xticks(list(range(len(shuire_matrix[0]))), map_str, rotation=90)
        plt.yticks(list(range(len(shuire_matrix))), mat_str, rotation=0)
        plt.xlabel('MAP (mm)')
        plt.ylabel('MAT (°C)')
        plt.title('Sensitivity')
        plt.colorbar()
        plt.show()

    def lag(self, lc):

        dff = Global_vars().dff
        df = T.load_df(dff)
        df = df[df['lc'] == lc]
        # df = df.drop_duplicates(subset=['pix'])
        vals_dic = DIC_and_TIF().void_spatial_dic()
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            sens = row.lag
            # sens = row.correlation
            vals_dic[pix].append(sens)
        MAT_series = df.MA_TMP
        MAP_series = df.MA_PRE * 12.
        df['MA_PRE'] = MAP_series
        d_mat, mat_str = self.__divide_MA(MAT_series, step=1)
        d_map, map_str = self.__divide_MA(MAP_series, min_v=0, max_v=2001, step=100)
        # print map_str
        # print d_map
        # exit()

        shuire_matrix = []
        x = []
        y = []
        z = []
        for t in tqdm(list(range(len(d_mat)))):
            if t + 1 >= len(d_mat):
                continue
            df_t = df[df['MA_TMP'] > d_mat[t]]
            df_t = df_t[df_t['MA_TMP'] < d_mat[t + 1]]
            temp = []
            for p in range(len(d_map)):
                if p + 1 >= len(d_map):
                    continue
                df_p = df_t[df_t['MA_PRE'] > d_map[p]]
                df_p = df_p[df_p['MA_PRE'] < d_map[p + 1]]
                pixs = df_p.pix

                if len(pixs) != 0:
                    vals = []
                    for pix in pixs:
                        val = vals_dic[pix]
                        val = np.nanmean(val)
                        vals.append(val)
                    val_mean = np.nanmean(vals)
                else:
                    val_mean = np.nan
                x.append(d_map[p])
                y.append(d_mat[t])
                z.append(val_mean)
                temp.append(val_mean)
            shuire_matrix.append(temp)
        plt.figure(figsize=(4, 6))
        cmap = 'RdBu_r'
        plt.scatter(x, y, c=z, marker='s', cmap=cmap, norm=None, vmin=1, vmax=6)
        plt.gca().invert_yaxis()
        plt.subplots_adjust(
            top=0.88,
            bottom=0.11,
            left=0.12,
            right=0.90,
            hspace=0.2,
            wspace=0.2
        )
        plt.title('{} Lag (months)'.format(lc))
        plt.colorbar()
        plt.xlabel('MAP (mm)')
        plt.ylabel('MAT (°C)')
        # plt.title('{} Sensitivity'.format(lc))
        # plt.show()

    def recovery(self, lc):
        dff = Global_vars().dff
        df = T.load_df(dff)
        df = df[df['lc'] == lc]
        # df = df.drop_duplicates(subset=['pix'])
        vals_dic = DIC_and_TIF().void_spatial_dic()
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            sens = row.recovery_time
            # sens = row.correlation
            vals_dic[pix].append(sens)
        MAT_series = df.MA_TMP
        # MAP_series = df.MA_PRE
        MAP_series = df.MA_PRE * 12.
        df['MA_PRE'] = MAP_series
        d_mat, mat_str = self.__divide_MA(MAT_series, step=1)
        d_map, map_str = self.__divide_MA(MAP_series, min_v=0, max_v=2001, step=100)
        # print map_str
        # print d_map
        # exit()

        shuire_matrix = []
        x = []
        y = []
        z = []
        for t in tqdm(list(range(len(d_mat)))):
            if t + 1 >= len(d_mat):
                continue
            df_t = df[df['MA_TMP'] > d_mat[t]]
            df_t = df_t[df_t['MA_TMP'] < d_mat[t + 1]]
            temp = []
            for p in range(len(d_map)):
                if p + 1 >= len(d_map):
                    continue
                df_p = df_t[df_t['MA_PRE'] > d_map[p]]
                df_p = df_p[df_p['MA_PRE'] < d_map[p + 1]]
                pixs = df_p.pix

                if len(pixs) != 0:
                    vals = []
                    for pix in pixs:
                        val = vals_dic[pix]
                        val = np.nanmean(val)
                        vals.append(val)
                    val_mean = np.nanmean(vals)
                else:
                    val_mean = np.nan
                x.append(d_map[p])
                y.append(d_mat[t])
                z.append(val_mean)
                temp.append(val_mean)
            shuire_matrix.append(temp)
        # plt.imshow(shuire_matrix,vmin=1,vmax=6)
        # plt.xticks(range(len(shuire_matrix[0])), map_str, rotation=90)
        # plt.yticks(range(len(shuire_matrix)), mat_str, rotation=0)
        plt.figure(figsize=(4, 6))
        cmap = 'RdBu_r'
        plt.scatter(x, y, c=z, marker='s', cmap=cmap, norm=None, vmin=1, vmax=6)
        plt.gca().invert_yaxis()
        # plt.subplots_adjust(
        #     top=0.88,
        #     bottom=0.11,
        #     left=0.12,
        #     right=0.90,
        #     hspace=0.2,
        #     wspace=0.2
        # )
        plt.xlabel('MAP (mm)')
        plt.ylabel('MAT (°C)')
        plt.title('{} Recovery time (months)'.format(lc))

        plt.colorbar()
        # plt.show()

    # def plot_MAP(self):
    #     dff = Main_flow_Prepare().dff
    #     df = T.load_df(dff)
    #     df = Global_vars().clean_df(df)
    #
    #     map_dic = {}
    #     for i,row in tqdm(df.iterrows(),total=len(df)):
    #         pix = row.pix
    #         map = row.MA_PRE
    #         map_dic[pix] = map
    #     arr = DIC_and_TIF().pix_dic_to_spatial_arr(map_dic)
    #     arr = arr * 12.
    #     plt.imshow(arr)
    #     plt.colorbar()
    #     plt.show()
    #     pass


class SEM:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('SEM', result_root_this_script)
        self.Var_names()
        pass

    def run(self):
        # self.build_model()
        self.export_excel()
        pass

    def Var_names(self):
        observed_var = '''
        recovery_time
        current_sos_anomaly
        dormant_TMP
        dormant_SWE_terra

        PRE_drought_year
        TMP_drought_year
        VPD_drought_year

        PRE_second_year_spring
        TMP_second_year_spring
        VPD_second_year_spring
        '''

        latent_var = '''
        Drought_year_condition
        winter_condition
        spring_condition # maybe drop
        '''

        print('observed_var:', observed_var)
        print('latent_var:', latent_var)

        pass

    def model_description1(self):
        # PRE_drought_year
        # TMP_drought_year
        # VPD_drought_year
        desc = '''
        # regressions
        recovery_time ~ current_sos_std_anomaly + PRE_second_year_spring  + TMP_second_year_spring + dormant_SWE_terra + dormant_TMP + PRE_drought_year + TMP_drought_year + VPD_drought_year
        '''

        # '''
        # # residual correlations
        # current_sos_std_anomaly~~current_sos_std_anomaly
        # current_sos_std_anomaly~~PRE_second_year_spring
        # current_sos_std_anomaly~~TMP_second_year_spring
        # current_sos_std_anomaly~~dormant_SWE_terra
        # current_sos_std_anomaly~~dormant_TMP
        # current_sos_std_anomaly~~PRE_drought_year
        # current_sos_std_anomaly~~TMP_drought_year
        # current_sos_std_anomaly~~VPD_drought_year
        #
        # recovery_time~~recovery_time
        # recovery_time~~PRE_second_year_spring
        # recovery_time~~TMP_second_year_spring
        # recovery_time~~dormant_SWE_terra
        # recovery_time~~dormant_TMP
        # recovery_time~~PRE_drought_year
        # recovery_time~~TMP_drought_year
        # recovery_time~~VPD_drought_year
        # recovery_time~~current_sos_std_anomaly'''

        return desc
        pass

        # VPD_lag_mean~~VPD_lag_mean

    def model_description(self):
        # PRE_drought_year
        # TMP_drought_year
        # VPD_drought_year
        desc = '''
        # regressions
        current_sos_std_anomaly ~ dormant_SWE_terra + dormant_TMP + PRE_MR_drought_start_to_eos + TMP_MR_drought_start_to_eos + VPD_MR_drought_start_to_eos
        recovery_time ~ PRE_second_year_spring  + TMP_second_year_spring + dormant_SWE_terra + dormant_TMP + PRE_MR_drought_start_to_eos + TMP_MR_drought_start_to_eos + VPD_MR_drought_start_to_eos + current_sos_std_anomaly
        # residual correlations
        current_sos_std_anomaly~~current_sos_std_anomaly
        current_sos_std_anomaly~~dormant_SWE_terra
        current_sos_std_anomaly~~dormant_TMP
        current_sos_std_anomaly~~PRE_MR_drought_start_to_eos
        current_sos_std_anomaly~~TMP_MR_drought_start_to_eos
        current_sos_std_anomaly~~VPD_MR_drought_start_to_eos

        recovery_time~~recovery_time
        recovery_time~~PRE_second_year_spring
        recovery_time~~TMP_second_year_spring
        recovery_time~~dormant_SWE_terra
        recovery_time~~dormant_TMP
        recovery_time~~PRE_MR_drought_start_to_eos
        recovery_time~~TMP_MR_drought_start_to_eos
        recovery_time~~VPD_MR_drought_start_to_eos
        recovery_time~~current_sos_std_anomaly
        '''
        # '''

        return desc
        pass

        # VPD_lag_mean~~VPD_lag_mean

    def build_model(self):
        desc = self.model_description()
        print(desc)
        exit()

        # desc = self.model_description1()
        # desc = self.model_description_with_latent_variables()
        df = Global_vars().load_df()
        df = df[df['winter_mark'] == 1]
        df = df[df['winter_mark_new'] == 1]
        # df = df[df['recovery_start_gs']=='first']
        # df = df[df['recovery_start_gs']=='second']
        # df = df[df['timing_int']!='early']
        df = df.dropna(subset=['lc'])
        df = df[df['product'] == 'spei03']
        df = df.dropna(subset=['timing_int'])
        T.print_head_n(df, 10)
        # exit()
        # recovery_time = df['recovery_time'].tolist()
        # plt.hist(recovery_time,bins=30)
        # plt.show()
        mod = semopy.Model(desc)
        res = mod.fit(df)
        # x_var_list = ['PRE_second_year_spring','TMP_second_year_spring','dormant_SWE_terra','dormant_TMP','PRE_drought_year','TMP_drought_year','VPD_drought_year','current_sos_std_anomaly','recovery_time']
        # print(mod.predict_factors())
        # df_X = df[x_var_list]
        # factors = mod.predict_factors(df_X)
        # print(factors.head())
        # exit()
        # print(res)
        # ins = mod.inspect()
        # print(ins)
        # stats = semopy.calc_stats(mod)
        # print(stats)
        semopy.report(mod, "SEM_report1111")
        pass

    def export_excel(self):
        observed_var = '''
        recovery_time
        current_sos_anomaly
        dormant_TMP
        dormant_SWE_terra

        PRE_drought_year
        TMP_drought_year
        VPD_drought_year

        PRE_second_year_spring
        TMP_second_year_spring
        VPD_second_year_spring
        '''
        observed_var = observed_var.split()
        # print(observed_var)
        # exit()
        df = Global_vars().load_df()
        df = df[df['winter_mark'] == 1]
        df = df[df['winter_mark_new'] == 1]
        # df = df[df['recovery_start_gs']=='first']
        # df = df[df['recovery_start_gs']=='second']
        # df = df[df['timing_int']!='early']
        df = df.dropna(subset=['lc'])
        df = df[df['product'] == 'spei03']
        df = df.dropna(subset=['timing_int'])
        df_selected = df[observed_var]
        outf = join(self.this_class_arr, 'SEM.xlsx')
        df_selected.to_excel(outf)


def main():
    # Main_flow_Early_Peak_Late_Dormant().run()
    # Main_flow_Pick().run()
    # Recovery_time().run()
    # Dataframe().run()
    # Analysis().run()
    Tif().run()
    # RF().run()
    # Koppen_Linear_Regression().run()
    # Main_flow_shui_re().run()
    # Partial_Dependence_Plots().run()
    # Partial_Dependence_Plots_timing().run()
    # Correlation_matrix().run()
    # Sensitivity().run()
    # Statistics().run()
    # Main_flow_shui_re().run()
    # SEM().run()
    pass


if __name__ == '__main__':
    main()







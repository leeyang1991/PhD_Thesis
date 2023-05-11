# coding=utf-8
from __init__ import *

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 8
centimeter_factor = 1 / 2.54

this_root = '/Volumes/NVME2T/PhD_Thesis_project/'
data_root = this_root + 'data/'
results_root = this_root + 'results/'
temp_root = this_root + 'temp/'

global_drought_type_list = ['normal-drought', 'hot-drought']
global_drought_type_color_dict = {
    'normal-drought': 'blue',
    'hot-drought': 'red',
}
global_ELI_class = ['Energy-Limited', 'Water-Limited']
global_ELI_class_color_dict = {
    'Energy-Limited': 'blue',
    'Water-Limited': 'red',
}
global_AI_class = ['Humid', 'Arid']
global_land_tif = join(this_root,'conf/land.tif')
global_land_tif_reproj = join(this_root,'conf/land_reproj.tif')
global_year_range = '1982-2015'
global_start_year,global_end_year = global_year_range.split('-')
global_start_year = int(global_start_year)
global_end_year = int(global_end_year)
global_year_range_list = list(range(global_start_year,global_end_year+1))
# data_path_dict = Meta_information().path(global_year_range)
global_gs = list(range(5,11))

global_lc_list = ('deciduous', 'evergreen', 'grass', 'shrubs')
global_lc_marker_dict = {
    'deciduous': 'o',
    'evergreen': 's',
    'grass': 'v',
    'shrubs': 'D',
}
global_koppen_list = ('arid', 'cold arid', 'cold humid', 'hot arid', 'hot humid')
global_koppen_color_dict = {
    'arid': '#EB6100',
    'cold arid': '#601986',
    'cold humid': 'b',
    'hot arid': 'r',
    'hot humid': 'g',
}
global_ELI_class_list = ('Energy-Limited', 'Water-Limited')
global_AI_class_list = ('Humid', 'Arid')
global_threshold = 0.05

global_VIs_list = ['NDVI','VOD-anomaly','CSIF-anomaly']
global_VIs_origin_list = ['NDVI-origin','VOD-origin','CSIF-origin']
global_VIs_year_range_dict = {
    'NDVI': '1982-2015',
    'NDVI-origin': '1982-2015',
    'VOD-origin': '2003-2015',
    'VOD-anomaly': '2003-2015',
    'CSIF-origin': '2001-2015',
    'CSIF-anomaly': '2001-2015',
    'VOD-k-band-origin': '1988-2015',
    'VOD-k-band-anomaly': '1988-2015',
}
global_color_list = [
    '#844000',
    '#fc9831',
    '#fffbd4',
    '#86b9d2',
    '#064c6c',
]
global_cmap = T.cmap_blend(global_color_list,)
global_cmap_r = T.cmap_blend(global_color_list[::-1])
global_spei_list = ['spei03', 'spei06', 'spei09', 'spei12',
                     'spei15','spei18', 'spei21', 'spei24',
                     ]
global_all_spei_list = [
    'spei01', 'spei02', 'spei03', 'spei04', 'spei05', 'spei06', 'spei07', 'spei08', 'spei09', 'spei10', 'spei11', 'spei12',
    'spei13', 'spei14', 'spei15', 'spei16', 'spei17', 'spei18', 'spei19', 'spei20', 'spei21', 'spei22', 'spei23', 'spei24',
                     ]
global_all_spi_list = [i.replace('spei','spi') for i in global_all_spei_list]
global_ELI_bins = np.arange(-0.8, 0.75, .05)
global_lag_list = list(range(0, 7))
global_drought_timing_list = ['early','mid','late']


class Meta_information:

    def __init__(self):
        pass

    def path(self,year_range=global_year_range):
        vars_info_dic = {
            'SPEI': {
                'path':join(data_root, 'SPEI/per_pix_clean',year_range),
                'path_type':'multi-files',
            },
            'CCI-SM': {
                'path': join(data_root, f'CCI-SM/detrend/{year_range}/CCI-SM.npy'),
                'path_type': 'file',
            },
            'ERA-SM': {
                'path': join(data_root, f'ERA-SM/detrend/{year_range}/ERA-SM.npy'),
                'path_type': 'file',
            },
            'SPI': {
                'path': join(data_root, 'SPI/per_pix',year_range),
                'path_type': 'multi-files',
            },
            'NDVI': {
                'path': join(data_root, 'GIMMS_NDVI/per_pix_anomaly_detrend',year_range),
                'path_type': 'dir',
            },
            'NDVI-origin': {
                'path': join(data_root, 'GIMMS_NDVI/per_pix', year_range),
                'path_type': 'dir',
            },
            'CSIF-origin': {
                'path': join(data_root, 'CSIF/per_pix', year_range),
                'path_type': 'dir',
            },
            'CSIF-anomaly': {
                'path': join(data_root, 'CSIF/per_pix_anomaly', year_range),
                'path_type': 'dir',
            },
            'VOD-origin': {
                'path': join(data_root, 'VOD_Xband/per_pix', year_range),
                'path_type': 'dir',
            },
            'VOD-anomaly': {
                'path': join(data_root, 'VOD_Xband/per_pix_anomaly_detrend', year_range),
                'path_type': 'dir',
            },
            'VOD-k-band-origin': {
                'path': join(data_root, 'VOD_Kband/per_pix', year_range),
                'path_type': 'dir',
            },
            'VOD-k-band-anomaly': {
                'path': join(data_root, 'VOD_Kband/detrend', year_range),
                'path_type': 'dir',
            },
            'Temperature-detrend': {
                'path': join(data_root, f'CRU_tmp/detrend/{year_range}/temp.npy'),
                'path_type': 'file',
            },
            'Temperature-origin': {
                'path': join(data_root, f'CRU_tmp/per_pix/{year_range}'),
                'path_type': 'dir',
            },
            'Temperature-anomaly': {
                'path': join(data_root, f'CRU_tmp/anomaly/{year_range}'),
                'path_type': 'dir',
            },
            'Precipitation-detrend': {
                'path': join(data_root, f'CRU_precip/detrend/{year_range}/precip.npy'),
                'path_type': 'file',
            },
            'Precipitation-origin': {
                'path': join(data_root, f'CRU_precip/per_pix/{year_range}'),
                'path_type': 'dir',
            },
            'Precipitation-anomaly': {
                'path': join(data_root, f'CRU_precip/anomaly/{year_range}'),
                'path_type': 'dir',
            },
            'Radiation': {
                'path': join(data_root, f'Terraclimate/srad/detrend/{year_range}/srad.npy'),
                'path_type': 'file',
            },
            'Terra-ET': {
                'path': join(data_root, f'Terraclimate/aet/detrend/{year_range}/aet.npy'),
                'path_type': 'file',
            },
            'GLEAM-ET': {
                'path': join(data_root, f'GLEAM_ET/detrend/{year_range}/GLEAM_ET.npy'),
                'path_type': 'file',
            },
            'VPD': {
                'path': join(data_root, f'VPD/anomaly/{year_range}/VPD.npy'),
                'path_type': 'file',
            },
            'VPD-origin': {
                'path': join(data_root, f'VPD/per_pix/{year_range}/VPD.npy'),
                'path_type': 'file',
            },
            'GOME2_SIF-origin': {
                'path': join(data_root, f'GOME2_SIF/per_pix/{year_range}'),
                'path_type': 'dir',
            },
            'GOME2_SIF-anomaly': {
                'path': join(data_root, f'GOME2_SIF/detrend/{year_range}'),
                'path_type': 'dir',
            },
            'ERA_Ta2m-origin': {
                'path': join(data_root, 'ERA_2m_T/perpix', year_range),
                'path_type': 'dir',
            },
            'ERA_Ta2m-anomaly': {
                'path': join(data_root, 'ERA_2m_T/anomaly', year_range),
                'path_type': 'dir',
            },
            'ERA_precip-origin': {
                'path': join(data_root, 'ERA_Precip/perpix', year_range),
                'path_type': 'dir',
            },
            'ERA_precip-anomaly': {
                'path': join(data_root, 'ERA_Precip/anomaly', year_range),
                'path_type': 'dir',
            },
        }
        return vars_info_dic

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


class Load_dataframe:

    def __init__(self):

        pass

    def load_chapter3(self):
        from Chapter3 import statistic
        dff = statistic.Dataframe().dff
        df = T.load_df(dff)
        df = df[df['lat'] > 30]
        df = df[df['aridity_index'] < 3]
        df = df[df['MAP'] < 1500]
        df = df[df['ISO_Hydricity'] < 1.5]
        df = df[df['ISO_Hydricity'] > 0]
        return df

    def load_chapter4(self):

        raise NotImplementedError

    def load_chapter5(self):
        from Chapter5 import statistic
        dff = statistic.Dataframe().dff
        df = T.load_df(dff)
        cols = get_rs_rt_cols()
        for col in cols:
            df[df[col] > 1.4] = np.nan
            df[df[col] < 0.6] = np.nan
        df = df[df['lat'] > 30]
        df = df.drop_duplicates()
        return df

def get_rs_rt_cols():
    post_n_list = [1, 2, 3, 4]
    cols = ['rt']
    for n in post_n_list:
        cols.append('rs_{}'.format(n))
    return cols

def year_range_str_to_list(year_range_str):
    year_range = year_range_str.split('-')
    year_range = list(range(int(year_range[0]), int(year_range[1]) + 1))
    return year_range
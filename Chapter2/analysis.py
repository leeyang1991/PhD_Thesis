# coding=utf-8

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


def main():
    Plot_SPEI().run()

if __name__ == '__main__':
    main()
import os
import shutil
import numpy as np
import torch
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from aux import tvector_2_se3

class DBMetrics(object):
    '''
    Metrics implemented:
    MDM aka motion diversity metric'''
    def __init__(self,datasetname = None, dataset_sequences = None ,plot = True, logging_dir = None,
                 brightness = False, blur = False, delentropy = False, pca = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pca = PCA(n_components=3)

        self.data_loader = # your data loader
        
        trans_data, rot_data, sigma_trans, sigma_rots, sigma, metrics_df, max_delentropy_img, min_delentropy_img = self.unified_analysis(brightness = brightness, blur = blur, 
                                                                                                 delentropy = delentropy, pca = pca)
        # Create folder for log
        survey_log_dir = os.path.join(logging_dir,datasetname)
        if os.path.exists(survey_log_dir):
            shutil.rmtree(survey_log_dir)
        os.makedirs(survey_log_dir)
        
        # Set latex fonts for matplotlib
        matplotlib.rcParams['mathtext.fontset'] = 'cm'
        matplotlib.rcParams['font.family'] = 'STIXGeneral'

        if pca:
            fig_motion = plt.gcf()
            # ax_motion = fig_motion.add_subplot(121)
            metrics_df.plot(y = ['t1','t2','t3'],use_index=True)
            plt.title("Principal axis for motion - translations", loc="left")
            plt.savefig(os.path.join(survey_log_dir,'motion_pattern_t.svg'), format='svg')
            # ax_motion = fig_motion.add_subplot(122)
            metrics_df.plot(y = ['r1','r2','r3'],use_index=True)
            plt.title("Principal axis for motion - rotations", loc="left")
            plt.savefig(os.path.join(survey_log_dir,'motion_pattern_r.svg'), format='svg')

        if brightness or delentropy:
            fig_violin = plt.figure()           
            ax_violin = fig_violin.add_subplot(111)  
            # Filter columns based on conditions
            columns_to_plot = []   
            if brightness:    
                columns_to_plot.append('brightness distance')        

            if delentropy:
                columns_to_plot.append('delentropy')


            # Melt the dataframe for seaborn plotting
            melted_df = metrics_df.melt(id_vars=['sequence'], value_vars=columns_to_plot, var_name='metric')

            # Plotting all metrics in one violin plot grouped by sequence
            sns.violinplot(x='sequence', y='value', data=melted_df)
            # plt.title("Combined Metrics Analysis", loc="left")
            
            if plot:
                plt.show()

        if logging_dir is not None:
            if delentropy:
                # Save csv files
                metrics_df.to_csv(os.path.join(survey_log_dir,'image_metrics.csv'))
                # Save svg images
                fig_violin.savefig(os.path.join(survey_log_dir,'image_metrics.svg'), format='svg')
                # Get min entropy image
                cv2.imwrite(os.path.join(survey_log_dir,"min_entropy_img.png"), (min_delentropy_img+.5)*255)
                # Get max entropy image
                max_index = int(metrics_df[['delentropy']].idxmax())
                cv2.imwrite(os.path.join(survey_log_dir,"max_entropy_img.png"), (max_delentropy_img+.5)*255) 
                
            if pca and delentropy:
                txt1 = []
                txt2 = []
                txt3 = []
                txt4 = []
                txt5 = []
                if pca:
                    txt1 = '\n rotation motion diversity metric for '+ datasetname +' :'+str(sigma_rots)
                    txt2 = '\n traslation motion diversity metric for '+ datasetname +' :'+str(sigma_trans)
                    txt3 = '\n overall motion diversity metric for '+ datasetname +' :'+str(sigma)
                if delentropy:
                    txt4 = '\n Entropy value for min_entropy_img.png'+' :'+str(metrics_df['delentropy'].min())
                    txt5 = '\n Entropy value for max_entropy_img.png'+' :'+str(metrics_df['delentropy'].max())
                
                txt = [txt1, txt2, txt3, txt4,txt5]
                logfile = open(os.path.join(survey_log_dir,'datalogger.txt'), 'w+')
                logfile.writelines(txt) 
                logfile.close()

    
    def unified_analysis(self, brightness=False, blur=False, delentropy=False,pca = False):
        # Initialize necessary variables and data structures
        # for motion analysis
        T_target_prev = []
        # for image analysis
        sequence_id = []
        dataset_id = []
        all_blur_fft = []
        brightness_distance = []
        all_delentropy = []
        # Variables to track the max and min delentropy and corresponding images
        max_delentropy = float('-inf')
        min_delentropy = float('inf')
        max_delentropy_img = None
        min_delentropy_img = None

        for idx, seq in tqdm(enumerate(self.data_loader), total=len(self.data_loader)):
            # Image Metrics
            if brightness or blur or delentropy:
                # Retrieve image or relevant data from 'seq'
                current_imgdata = seq["keyframe"].squeeze().permute(1, 2, 0).numpy()

                # TODO: if brightness metric
                # TODO: if blur metric

                if delentropy:
                    # Compute delentropy metric
                    delentropy_result = self.compute_delentropy(current_imgdata)
                    all_delentropy.append(delentropy_result)
                    # Update max and min delentropy and corresponding images
                    if delentropy_result > max_delentropy:
                        max_delentropy = delentropy_result
                        max_delentropy_img = current_imgdata.copy()
                    if delentropy_result < min_delentropy:
                        min_delentropy = delentropy_result
                        min_delentropy_img = current_imgdata.copy()

            # Motion Pattern Analysis
            if seq["poses"] is not None:
                H_kf0_kf1 = seq["poses"][0]               
                T_target_prev.append(H_kf0_kf1)
                
            dataset_id.append(seq['dataset_name'][0])
            sequence_id.append(seq['sequence_name'][0])
        
        trans_data, rot_data, sigma_trans, sigma_rots, sigma = self.motion_metric(T_target_prev)

        metrics_df = pd.DataFrame({ 'dataset' : pd.Series(dataset_id),#'imgpath' : pd.Series(image_paths) , 
                                    'sequence' : pd.Series(sequence_id),#'imgpath' : pd.Series(image_paths) , 
                                    'delentropy': pd.Series(all_delentropy),
                                    't1': trans_data[:,0], 't2': trans_data[:,1], 't3': trans_data[:,2], 
                                    'r1': rot_data[:,0], 'r2': rot_data[:,1], 'r3': rot_data[:,2] })

        return trans_data, rot_data, sigma_trans, sigma_rots, sigma, metrics_df, max_delentropy_img, min_delentropy_img

    def motion_metric(self, T_target_prev):
        seq_transforms = [tensor.reshape(1,-1).numpy().flatten() for tensor in T_target_prev]
        seq_transforms = tvector_2_se3(seq_transforms)
        
        trans_data = self.pca.fit_transform(seq_transforms[:, 0:3])
        trans_ratio = self.pca.explained_variance_ratio_
        
        rot_data = self.pca.fit_transform(seq_transforms[:, 3:6])
        rot_ratio = self.pca.explained_variance_ratio_
        
        sigma_trans = np.sqrt((trans_ratio[1]/trans_ratio[0])*(trans_ratio[2]/trans_ratio[0]))
        sigma_rots = np.sqrt((rot_ratio[1]/rot_ratio[0])*(rot_ratio[2]/rot_ratio[0]))
        sigma = 0.5*sigma_trans+0.5*sigma_rots

        return trans_data, rot_data, sigma_trans, sigma_rots, sigma

    def compute_delentropy(self, current_imgdata):
        # see paper eq. (4)
        fx = cv2.Sobel(current_imgdata, ddepth=-1, dx=1, dy=0, ksize=3)
        fx = fx.astype(int)
        fy = cv2.Sobel(current_imgdata, ddepth=-1, dx=0, dy=1, ksize=3)
        fy = fy.astype(int)

        nBins = min( 1024, 2*255+1 )
        if current_imgdata.dtype == float:
            nBins = 1024
        # Centering the bins is necessary because else all value will lie on
        # the bin edges thereby leading to assymetric artifacts
        dbin = 0 if current_imgdata.dtype == float else 0.5
        diffRange = 255
        r = diffRange + dbin
        delDensity, xedges, yedges = np.histogram2d( fx.flatten(), fy.flatten(), bins = nBins, range = [ [-r,r], [-r,r] ] )
        if nBins == 2*diffRange+1:
            assert( xedges[1] - xedges[0] == 1.0 )
            assert( yedges[1] - yedges[0] == 1.0 )

        # Normalization for entropy calculation. np.sum( H ) should be ( imageWidth-1 )*( imageHeight-1 )
        # The -1 stems from the lost pixels when calculating the gradients with non-periodic boundary conditions
        #assert( np.product( np.array( image.shape ) - 1 ) == np.sum( delDensity ) )
        delDensity = delDensity / np.sum( delDensity ) # see paper eq. (17)
        delDensity = delDensity.T
        # "The entropy is a sum of terms of the form p log(p). 
        # When p=0 you instead use the limiting value (as p approaches 0 from above), which is 0."
        # The 0.5 factor is discussed in the paper chapter "4.3 Papoulis generalized sampling halves the delentropy"
        H = - 0.5 * np.sum( delDensity[ delDensity.nonzero() ] * np.log2( delDensity[ delDensity.nonzero() ] ) ) # see paper eq. (16)
        return H

           
if __name__ == "__main__":
    dataset_name = "SubPipe"
    dataset_sequences = ["Chunk0/Cam0_images", 
                         "Chunk1/Cam0_images", 
                         "Chunk2/Cam0_images", 
                         "Chunk3/Cam0_images", 
                         "Chunk4/Cam0_images"]
    # dataset_name = "EuRoC"
    # dataset_sequences = ["MH_01_easy","MH_04_difficult",  
    #                      "V1_02_medium","V2_02_medium",
    #                      "MH_02_easy","MH_05_difficult","V1_03_difficult",
    #                      "V2_03_difficult", "MH_03_medium","V1_01_easy","V2_01_easy"]
    # dataset_name = "KITTI"
    # dataset_sequences = ["00","01",  
    #                      "02","03",
    #                      "04","05","06",
    #                      "07", "08","09","10"]
    # dataset_name = "MIMIR"
    # dataset_sequences = ["SeaFloor/track0","SeaFloor/track1", "SeaFloor/track2",
    #                      "SeaFloor_Algae/track0","SeaFloor_Algae/track1", "SeaFloor_Algae/track2",
    #                      "OceanFloor/track0_dark","OceanFloor/track0_light","OceanFloor/track1_light",
    #                      "SandPipe/track0_dark", "SandPipe/track0_light"]
    # dataset_name = "Aqualoc/Archaeological_site_sequences"
    # dataset_sequences = ["1", "2", "3"]
    # dataset_name = "TartanAir"
    # dataset_sequences = [
    #             "abandonedfactory/Easy/P001","abandonedfactory/Easy/P002","abandonedfactory/Easy/P004","abandonedfactory/Easy/P005","abandonedfactory/Easy/P006","abandonedfactory/Easy/P008","abandonedfactory/Easy/P009","abandonedfactory/Easy/P010","abandonedfactory/Easy/P011", "abandonedfactory/Hard/P000","abandonedfactory/Hard/P001","abandonedfactory/Hard/P002","abandonedfactory/Hard/P003","abandonedfactory/Hard/P004","abandonedfactory/Hard/P005","abandonedfactory/Hard/P006","abandonedfactory/Hard/P007","abandonedfactory/Hard/P008","abandonedfactory/Hard/P009","abandonedfactory/Hard/P010",
    #             "abandonedfactory_night/Easy/P002","abandonedfactory_night/Easy/P003","abandonedfactory_night/Easy/P004","abandonedfactory_night/Easy/P005","abandonedfactory_night/Easy/P006","abandonedfactory_night/Easy/P007","abandonedfactory_night/Easy/P008","abandonedfactory_night/Easy/P009","abandonedfactory_night/Easy/P010","abandonedfactory_night/Easy/P011","abandonedfactory_night/Easy/P012","abandonedfactory_night/Easy/P013", "abandonedfactory_night/Hard/P000","abandonedfactory_night/Hard/P001","abandonedfactory_night/Hard/P002","abandonedfactory_night/Hard/P003","abandonedfactory_night/Hard/P005","abandonedfactory_night/Hard/P006","abandonedfactory_night/Hard/P007","abandonedfactory_night/Hard/P008","abandonedfactory_night/Hard/P009","abandonedfactory_night/Hard/P010","abandonedfactory_night/Hard/P011","abandonedfactory_night/Hard/P012","abandonedfactory_night/Hard/P013",
    #             "amusement/Easy/P002","amusement/Easy/P003","amusement/Easy/P004","amusement/Easy/P006","amusement/Easy/P007","amusement/Easy/P008", "amusement/Hard/P000","amusement/Hard/P001","amusement/Hard/P002","amusement/Hard/P003","amusement/Hard/P004","amusement/Hard/P005","amusement/Hard/P006",
    #             "carwelding/Easy/P002","carwelding/Easy/P004","carwelding/Easy/P005","carwelding/Easy/P006","carwelding/Easy/P007", "carwelding/Hard/P000","carwelding/Hard/P001","carwelding/Hard/P002",
    #             "endofworld/Easy/P001","endofworld/Easy/P002","endofworld/Easy/P003","endofworld/Easy/P004","endofworld/Easy/P005","endofworld/Easy/P006","endofworld/Easy/P007","endofworld/Easy/P008","endofworld/Easy/P009", "endofworld/Hard/P000","endofworld/Hard/P001","endofworld/Hard/P002","endofworld/Hard/P005",
    #             "gascola/Easy/P003","gascola/Easy/P004","gascola/Easy/P005","gascola/Easy/P006","gascola/Easy/P007","gascola/Easy/P008", "gascola/Hard/P000","gascola/Hard/P001","gascola/Hard/P002","gascola/Hard/P003","gascola/Hard/P004","gascola/Hard/P005","gascola/Hard/P006","gascola/Hard/P007","gascola/Hard/P008",
    #             "hospital/Easy/P001","hospital/Easy/P002","hospital/Easy/P003","hospital/Easy/P004","hospital/Easy/P005","hospital/Easy/P006","hospital/Easy/P007","hospital/Easy/P008","hospital/Easy/P009","hospital/Easy/P010","hospital/Easy/P011","hospital/Easy/P012","hospital/Easy/P013","hospital/Easy/P014","hospital/Easy/P015","hospital/Easy/P016","hospital/Easy/P017","hospital/Easy/P018","hospital/Easy/P019","hospital/Easy/P020","hospital/Easy/P021","hospital/Easy/P022","hospital/Easy/P023","hospital/Easy/P024","hospital/Easy/P025","hospital/Easy/P026","hospital/Easy/P027","hospital/Easy/P028","hospital/Easy/P029","hospital/Easy/P030","hospital/Easy/P031","hospital/Easy/P032","hospital/Easy/P033","hospital/Easy/P034","hospital/Easy/P035","hospital/Easy/P036", 
    #             "hospital/Hard/P038","hospital/Hard/P039","hospital/Hard/P040","hospital/Hard/P041","hospital/Hard/P042","hospital/Hard/P043","hospital/Hard/P044","hospital/Hard/P045","hospital/Hard/P046","hospital/Hard/P047","hospital/Hard/P048","hospital/Hard/P049",
    #             "japanesealley/Easy/P002","japanesealley/Easy/P003","japanesealley/Easy/P004","japanesealley/Easy/P005","japanesealley/Easy/P007", "japanesealley/Hard/P000","japanesealley/Hard/P001","japanesealley/Hard/P002","japanesealley/Hard/P003","japanesealley/Hard/P004",
    #             "neighborhood/Easy/P001","neighborhood/Easy/P002","neighborhood/Easy/P003","neighborhood/Easy/P004","neighborhood/Easy/P005","neighborhood/Easy/P007","neighborhood/Easy/P008","neighborhood/Easy/P009","neighborhood/Easy/P010","neighborhood/Easy/P012","neighborhood/Easy/P013","neighborhood/Easy/P014","neighborhood/Easy/P015","neighborhood/Easy/P016","neighborhood/Easy/P017","neighborhood/Easy/P018","neighborhood/Easy/P019","neighborhood/Easy/P020","neighborhood/Easy/P021", "neighborhood/Hard/P000", "neighborhood/Hard/P001", "neighborhood/Hard/P002", "neighborhood/Hard/P003", "neighborhood/Hard/P004", "neighborhood/Hard/P005", "neighborhood/Hard/P006", "neighborhood/Hard/P007", "neighborhood/Hard/P008", "neighborhood/Hard/P009", "neighborhood/Hard/P010", "neighborhood/Hard/P011", "neighborhood/Hard/P012", "neighborhood/Hard/P013", "neighborhood/Hard/P014", "neighborhood/Hard/P015", "neighborhood/Hard/P016", 
    #             "ocean/Easy/P001","ocean/Easy/P002","ocean/Easy/P004","ocean/Easy/P005","ocean/Easy/P006","ocean/Easy/P008","ocean/Easy/P009","ocean/Easy/P010","ocean/Easy/P011","ocean/Easy/P012","ocean/Easy/P013", "ocean/Hard/P000","ocean/Hard/P001","ocean/Hard/P002","ocean/Hard/P003","ocean/Hard/P004","ocean/Hard/P005","ocean/Hard/P006","ocean/Hard/P007","ocean/Hard/P008",
    #             "office/Easy/P001","office/Easy/P002","office/Easy/P003","office/Easy/P004","office/Easy/P005","office/Easy/P006", "office/Hard/P000","office/Hard/P001","office/Hard/P002","office/Hard/P003","office/Hard/P004","office/Hard/P005","office/Hard/P006",
    #             "office2/Easy/P000","office2/Easy/P003","office2/Easy/P004","office2/Easy/P005","office2/Easy/P006","office2/Easy/P007","office2/Easy/P008","office2/Easy/P009","office2/Easy/P010","office2/Easy/P011", "office2/Hard/P000","office2/Hard/P001","office2/Hard/P002","office2/Hard/P003","office2/Hard/P004","office2/Hard/P005","office2/Hard/P006","office2/Hard/P007","office2/Hard/P008","office2/Hard/P009","office2/Hard/P010",
    #             "oldtown/Easy/P001","oldtown/Easy/P002","oldtown/Easy/P004","oldtown/Easy/P005","oldtown/Easy/P007", "oldtown/Hard/P000","oldtown/Hard/P001","oldtown/Hard/P002","oldtown/Hard/P003","oldtown/Hard/P004","oldtown/Hard/P005","oldtown/Hard/P006","oldtown/Hard/P007",
    #             "seasonsforest/Easy/P002","seasonsforest/Easy/P003","seasonsforest/Easy/P004","seasonsforest/Easy/P005","seasonsforest/Easy/P007","seasonsforest/Easy/P008","seasonsforest/Easy/P009","seasonsforest/Easy/P010","seasonsforest/Easy/P011","seasonsforest/Hard/P001","seasonsforest/Hard/P002","seasonsforest/Hard/P004","seasonsforest/Hard/P005",
    #             "seasonsforest_winter/Easy/P001","seasonsforest_winter/Easy/P002","seasonsforest_winter/Easy/P003","seasonsforest_winter/Easy/P004","seasonsforest_winter/Easy/P005","seasonsforest_winter/Easy/P006","seasonsforest_winter/Easy/P007","seasonsforest_winter/Easy/P008","seasonsforest_winter/Easy/P009", "seasonsforest_winter/Hard/P010", "seasonsforest_winter/Hard/P011", "seasonsforest_winter/Hard/P012", "seasonsforest_winter/Hard/P013", "seasonsforest_winter/Hard/P014", "seasonsforest_winter/Hard/P015", "seasonsforest_winter/Hard/P016", "seasonsforest_winter/Hard/P017", 
    #             "westerndesert/Easy/P002","westerndesert/Easy/P004","westerndesert/Easy/P005","westerndesert/Easy/P006","westerndesert/Easy/P007","westerndesert/Easy/P008","westerndesert/Easy/P009","westerndesert/Easy/P010","westerndesert/Easy/P011","westerndesert/Easy/P012","westerndesert/Easy/P013", "westerndesert/Hard/P000","westerndesert/Hard/P001","westerndesert/Hard/P002","westerndesert/Hard/P003","westerndesert/Hard/P004","westerndesert/Hard/P005","westerndesert/Hard/P006",
    #             "abandonedfactory/Easy/P000","abandonedfactory/Hard/P011",
    #             "abandonedfactory_night/Easy/P001","abandonedfactory_night/Hard/P014",
    #             "amusement/Easy/P001","amusement/Hard/P007",
    #             "carwelding/Easy/P001","carwelding/Hard/P003",
    #             "endofworld/Easy/P000","endofworld/Hard/P006",
    #             "gascola/Easy/P001","gascola/Hard/P009",
    #             "hospital/Easy/P000","hospital/Hard/P037",
    #             "japanesealley/Easy/P001","japanesealley/Hard/P005",
    #             "neighborhood/Easy/P000", "neighborhood/Hard/P017", 
    #             "ocean/Easy/P000","ocean/Hard/P009",
    #             "office/Easy/P000","office/Hard/P007",
    #             "oldtown/Easy/P000","oldtown/Hard/P008",
    #             "seasonsforest/Easy/P001","seasonsforest/Hard/P006",
    #             "seasonsforest_winter/Easy/P000","seasonsforest_winter/Hard/P018", 
    #             "westerndesert/Easy/P001","westerndesert/Hard/P007"
    #         ]
    metricCalculator = DBMetrics(datasetname = dataset_name, dataset_sequences = dataset_sequences, plot =False, 
                                 logging_dir = os.path.join(os.getcwd(),'metrics','results'),
                                 brightness = False, blur = False, delentropy=True, pca = True)
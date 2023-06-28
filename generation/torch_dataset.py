import numpy as np
import pickle
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import constants.constants as constants
import torch

class Set(Dataset):

    def __init__(self, setType = "train"):
        self.index_eye = range(constants.eye_size)
        self.index_pose =range(constants.eye_size, constants.eye_size + constants.pose_r_size)
        self.index_au = range(constants.eye_size + constants.pose_r_size, constants.eye_size + constants.pose_r_size + constants.au_size)

        # Load data
        path = constants.data_path
        datasets = constants.datasets
        self.X = []
        self.Y = []
        self.interval = []

        for set_name in datasets:
            current_X = []
            with open(path +'X_'+setType+'_'+set_name+'.p', 'rb') as f:
                x = pickle.load(f)
            current_X = np.array(x)[:,:,np.r_[constants.selected_os_index_columns]]

            with open(path +'y_'+setType+'_'+set_name+'.p', 'rb') as f:
                current_Y = pickle.load(f)

            with open(path +'intervals_test_'+set_name+'.p', 'rb') as f:
                current_interval = pickle.load(f)

            self.X.extend(current_X)
            self.Y.extend(current_Y)
            self.interval.extend(current_interval)
        
        #one scaler per features 
        self.X_scaled, self.x_scaler_tab = self.scale_x_from_scratch(self.X)
        #one scaler per type of feature
        self.Y_scaled, self.y_scaler_tab = self.scale_y_from_scratch(self.Y)


        if(setType == "test"):
            with open(path +'y_test_final_'+set_name+'.p', 'rb') as f:
                self.Y_final_ori = pickle.load(f)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

    def getInterval(self, i):
        return self.interval[i]
    

    def scale_x_from_scratch(self, x):
        x_array = np.array(x)
        nb_features_x = x_array.shape[2]
        seq_len = x_array.shape[1]

        tab_scaler = []
        if(constants.scale_each_audio):
            tab_scaled = []
            for index_feature in range(nb_features_x):
                features_tab = x_array[:,:,index_feature]
                scaler = MinMaxScaler((0,1)).fit(features_tab) 
                tab_scaler.append(scaler)
                tab_scaled.append(scaler.transform(features_tab))
            x_scaled = np.stack(tab_scaled, axis=2)
        else:
            x_flat = x_array.reshape(-1, nb_features_x)
            scaler = MinMaxScaler((0,1)).fit(x_flat)
            x_scaled = scaler.transform(x_flat).reshape(-1, seq_len, nb_features_x)
            tab_scaler.append(scaler)

        return x_scaled, tab_scaler
    
    def scale_y_from_scratch(self, y):
        y_array = np.array(y)
        nb_features_y = y_array.shape[2]
        seq_len = y_array.shape[1]
        tab_scaler = []

        if(constants.scale_each_pose):
            print("scale each pose")
            tab_scaled = []
            for index_feature in range(nb_features_y):
                features_tab = y_array[:,:,index_feature]
                scaler = MinMaxScaler((0,1)).fit(features_tab) 
                tab_scaler.append(scaler)
                tab_scaled.append(scaler.transform(features_tab))
            y_scaled = np.stack(tab_scaled, axis=2)
        else:
            print("Scale pose by features type")
            y_eye_flat = y_array[:,:,self.index_eye].reshape(-1, len(self.index_eye))
            y_pose_flat = y_array[:,:,self.index_pose].reshape(-1, len(self.index_pose))
            y_au_flat = y_array[:,:,self.index_au].reshape(-1, len(self.index_au))

            scaler_eye = MinMaxScaler((0,1)).fit(y_eye_flat)
            scaler_pose = MinMaxScaler((0,1)).fit(y_pose_flat)       
            scaler_au = MinMaxScaler((0,1)).fit(y_au_flat)  
            tab_scaler = [scaler_eye, scaler_pose, scaler_au]     

            y_eye_scaled = scaler_eye.transform(y_eye_flat).reshape(-1, seq_len, len(self.index_eye))
            y_pose_scaled = scaler_pose.transform(y_pose_flat).reshape(-1, seq_len, len(self.index_pose))
            y_au_scaled = scaler_au.transform(y_au_flat).reshape(-1, seq_len, len(self.index_au))
            y_scaled = np.concatenate((y_eye_scaled, y_pose_scaled, y_au_scaled), axis=2)

        return y_scaled, tab_scaler

    def scale_x(self, x, scaler_tab):
        x_array = np.array(x)
        if(len(x_array.shape) < 3):
            x_array = np.expand_dims(x_array, axis=0) #batch size 1
        nb_features_x = x_array.shape[2]
        seq_len = x_array.shape[1]

        if(constants.scale_each_audio):
            tab_scaled = []
            for index_feature in range(nb_features_x):
                features_tab = x_array[:,:,index_feature]
                tab_scaled.append(scaler_tab[index_feature].transform(features_tab))
            x_scaled = np.stack(tab_scaled, axis=2)
        else:
            x_flat = x_array.reshape(-1, nb_features_x)
            x_scaled = scaler_tab[0].transform(x_flat).reshape(-1, seq_len, nb_features_x)
        return x_scaled

    def scale_y(self, y, scaler_tab):
        y_array = np.array(y)
        if(len(y_array.shape) < 3):
            y_array = np.expand_dims(y_array, axis=0) #batch size 1
        nb_features_y = y_array.shape[2]
        seq_len = y_array.shape[1]

        if(constants.scale_each_pose):
            tab_scaled = []
            for index_feature in range(nb_features_y):
                features_tab = y_array[:,:,index_feature]
                tab_scaled.append(scaler_tab[index_feature].transform(features_tab))
            y_scaled = np.stack(tab_scaled, axis=2)
        else:
            y_eye_flat = y_array[:,:,self.index_eye].reshape(-1, len(self.index_eye))
            y_pose_flat = y_array[:,:,self.index_pose].reshape(-1, len(self.index_pose))
            y_au_flat = y_array[:,:,self.index_au].reshape(-1, len(self.index_au))   

            y_eye_scaled = scaler_tab[0].transform(y_eye_flat).reshape(-1, seq_len, len(self.index_eye))
            y_pose_scaled = scaler_tab[1].transform(y_pose_flat).reshape(-1, seq_len, len(self.index_pose))
            y_au_scaled = scaler_tab[2].transform(y_au_flat).reshape(-1, seq_len, len(self.index_au))

            y_scaled = np.concatenate((y_eye_scaled, y_pose_scaled, y_au_scaled), axis=2)

        return y_scaled


class TrainSet(Set):

    def __init__(self):
        super(TrainSet, self).__init__("train")

    def scaling(self, flag):
        if flag:
            self.X = self.X_scaled
            self.Y = self.Y_scaled

    def scale_pros(self, x):
        return self.scale_x(x, self.x_scaler_tab)


    def scale_pose(self, y):
        return self.scale_y(y, self.y_scaler_tab)

    def rescale_pose(self, y):
        y_array = np.array(y)
        if(len(y_array.shape) < 3):
            y_array = np.expand_dims(y_array, axis=0) #batch size 1
        seq_len = y_array.shape[1]
        nb_features_y = y_array.shape[2]

        if(constants.scale_each_pose):
            tab_rescaled = []
            for index_feature in range(nb_features_y):
                features_tab = y_array[:,:,index_feature]
                tab_rescaled.append(self.y_scaler_tab[index_feature].inverse_transform(features_tab))
            y_rescaled = np.stack(tab_rescaled, axis=2)

        else:
            y_eye_flat = y_array[:,:,self.index_eye].reshape(-1, len(self.index_eye))
            y_pose_flat = y_array[:,:,self.index_pose].reshape(-1, len(self.index_pose))
            y_au_flat = y_array[:,:,self.index_au].reshape(-1, len(self.index_au))   

            y_eye_rescaled = self.y_scaler_tab[0].inverse_transform(y_eye_flat).reshape(-1, seq_len, len(self.index_eye))
            y_pose_rescaled = self.y_scaler_tab[1].inverse_transform(y_pose_flat).reshape(-1, seq_len, len(self.index_pose))
            y_au_rescaled = self.y_scaler_tab[2].inverse_transform(y_au_flat).reshape(-1, seq_len, len(self.index_au))
            y_rescaled = np.concatenate((y_eye_rescaled, y_pose_rescaled, y_au_rescaled), axis=2)
            
        y_rescaled = np.reshape(y_rescaled,(-1, nb_features_y))
        return y_rescaled
    

class TestSet(Set):

    def __init__(self):
        super(TestSet, self).__init__("test")

    def scaling(self, x_scaler, y_scaler):
        self.X_scaled = self.scale_x(self.X, x_scaler)
        self.Y_scaled = self.scale_y(self.Y, y_scaler)

        self.X =  self.X_scaled
        self.Y = self.Y_scaled


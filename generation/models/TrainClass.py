import torch
import constants.constants as constants
from torch_dataset import TestSet, TrainSet

class Train():
    def __init__(self, gan=False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("*"*10, "cuda available ", torch.cuda.is_available(), "*"*10)
        self.gan = gan
        self.batchsize = constants.batch_size
        self.n_epochs = constants.n_epochs

        self.trainset = TrainSet()
        self.trainset.scaling(True)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batchsize,shuffle=True, pin_memory=True, num_workers=2)
        self.n_iteration_per_epoch = len(self.trainloader)

        testset = TestSet()
        testset.scaling(self.trainset.x_scaler_tab, self.trainset.y_scaler_tab)
        self.testloader = torch.utils.data.DataLoader(testset,batch_size=self.batchsize,shuffle=True,pin_memory=True, num_workers=2)

        #create the set for false exemple 
        speak_x = []
        speak_y = []
        for i in range(self.trainset.X_scaled.shape[0]):
            b = self.trainset.X_scaled[i,:,0] == 0
            index = b.nonzero()
            speak_x.append(torch.tensor(self.trainset.X_scaled[i,index,:]))
            speak_y.append(torch.tensor(self.trainset.Y_scaled[i,index,:]))
        self.speak_x = torch.squeeze(torch.cat(speak_x, dim=1))
        self.speak_y = torch.squeeze(torch.cat(speak_y, dim=1))


        no_speak_x = []
        no_speak_y = []
        for i in range(self.trainset.X_scaled.shape[0]):
            b = self.trainset.X_scaled[i,:,0] == 1
            index = b.nonzero()
            no_speak_x.append(torch.tensor(self.trainset.X_scaled[i,index,:]))
            no_speak_y.append(torch.tensor(self.trainset.Y_scaled[i,index,:]))
        self.no_speak_x = torch.squeeze(torch.cat(no_speak_x, dim=1))
        self.no_speak_y = torch.squeeze(torch.cat(no_speak_y, dim=1))

        
        self.reinitialize_loss()
        self.reinitialize_loss_tab()

    def reinitialize_loss_tab(self):
        self.loss_tab_eye = []
        self.loss_tab_pose_r = []
        self.loss_tab_au = []
        self.loss_tab = []
        self.t_loss_tab = []
        self.t_loss_tab_eye = []
        self.t_loss_tab_pose_r = []
        self.t_loss_tab_au = []

        if(self.gan):
            self.d_loss_tab = []
            self.d_real_pred_tab = []
            self.d_fake_pred_tab = []


    def update_loss_tab(self, iteration):
        self.current_loss_eye = self.current_loss_eye/(iteration)
        self.loss_tab_eye.append(self.current_loss_eye)

        self.current_loss_pose_r = self.current_loss_pose_r/(iteration)
        self.loss_tab_pose_r.append(self.current_loss_pose_r)

        self.current_loss_au = self.current_loss_au/(iteration)
        self.loss_tab_au.append(self.current_loss_au)

        self.current_loss = self.current_loss/(iteration)  # loss par epoch
        self.loss_tab.append(self.current_loss)

        self.t_loss_tab.append(self.t_loss)
        self.t_loss_tab_eye.append(self.t_loss_eye)
        self.t_loss_tab_pose_r.append(self.t_loss_pose_r)
        self.t_loss_tab_au.append(self.t_loss_au)

        if(self.gan):
            #d_loss
            self.current_d_loss = self.current_d_loss/(iteration) #loss par epoch
            self.d_loss_tab.append(self.current_d_loss)
            
            #real pred
            self.current_real_pred = self.current_real_pred/(iteration) 
            self.d_real_pred_tab.append(self.current_real_pred)
            
            #fake pred
            self.current_fake_pred = self.current_fake_pred/(iteration)
            self.d_fake_pred_tab.append(self.current_fake_pred)


    def reinitialize_loss(self):
        self.current_loss_eye = 0
        self.current_loss_pose_r = 0
        self.current_loss_au = 0
        self.current_loss = 0
        self.t_loss = 0
        self.t_loss_eye = 0
        self.t_loss_pose_r = 0
        self.t_loss_au = 0

        if(self.gan):
            self.current_d_loss = 0
            self.current_fake_pred = 0
            self.current_real_pred = 0

    def format_data(self, inputs, targets):
        target_eye, target_pose_r, target_au = self.separate_openface_features(targets)

        return inputs.float(), targets.float(), target_eye.float(), target_pose_r.float(), target_au.float()

    def separate_openface_features(self, input):
        input_eye = torch.index_select(input, dim=2, index = torch.tensor(range(constants.eye_size), device=self.device))
        input_pose_r = torch.index_select(input, dim=2, index=torch.tensor(range(constants.eye_size, constants.eye_size + constants.pose_r_size), device=self.device))
        input_au = torch.index_select(input, dim=2, index=torch.tensor(range(constants.eye_size + constants.pose_r_size, constants.eye_size + constants.pose_r_size + constants.au_size), device=self.device))
        return input_eye, input_pose_r, input_au
    
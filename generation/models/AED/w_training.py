from datetime import datetime
import random
from sklearn.utils import shuffle
from utils.noise_generator import NoiseGenerator
from models.TrainClass import Train
from models.AED.model import Generator, Discriminator
from utils.model_utils import saveModel
from utils.params_utils import save_params
from utils.plot_utils import plotHistAllLossEpoch, plotHistLossEpochGAN, plotHistPredEpochGAN
import constants.constants as constants
import torch.nn as nn
import torch

#disable debbuging API for faster training
torch.autograd.profiler.profile(False)
torch.autograd.set_detect_anomaly(False)
# enable cuDNN autotuner
torch.backends.cudnn.benchmark = True


import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set_style('whitegrid')


class TrainModel1_W(Train):

    def __init__(self, gan):
        super(TrainModel1_W, self).__init__(gan)

    def test_loss_w(self, G, D, testloader, criterion_loss, criterion_test_loss):
        torch.cuda.empty_cache()
        noise_g = NoiseGenerator(self.device)
        with torch.no_grad():
            print("Calculate test loss...")
            total_loss = 0
            total_loss_eye = 0
            total_loss_pose_r = 0
            total_loss_au = 0
            for iteration ,data in enumerate(testloader,1):
                inputs, targets = data[0].to(self.device), data[1].to(self.device)
                inputs, targets, target_eye, target_pose_r, target_au = self.format_data(inputs, targets)

                real_batch_size = inputs.shape[0]


                noise = noise_g.getNoise(constants.noise_size, real_batch_size) 
                gen_eye, gen_pose_r, gen_au = G(inputs, noise)
                fake_targets = torch.cat((gen_eye, gen_pose_r, gen_au), 2)
                d_fake_pred = D(fake_targets, inputs)

                g_loss = self.get_gen_loss(d_fake_pred)
                loss_eye = criterion_test_loss(gen_eye, target_eye)
                loss_pose_r = criterion_test_loss(gen_pose_r, target_pose_r)
                loss_au = criterion_test_loss(gen_au, target_au)

                g_total_loss = constants.eye_coeff * loss_eye + constants.pose_coeff * loss_pose_r + constants.au_coeff * loss_au + constants.adversarial_coeff * g_loss

                total_loss += g_total_loss.item()
                total_loss_eye += loss_eye.item()
                total_loss_pose_r += loss_pose_r.item()
                total_loss_au += loss_au.item()

            total_loss = total_loss/(iteration)
            total_loss_eye = total_loss_eye/(iteration)
            total_loss_pose_r = total_loss_pose_r/(iteration)
            total_loss_au = total_loss_au/(iteration)
            return total_loss, total_loss_eye, total_loss_pose_r, total_loss_au
        
    
    def gradient_penalty(self, gradient):
        gradient = gradient.reshape(len(gradient), -1)

        gradient_norm = gradient.norm(2, dim=1)
        
        penalty = torch.mean((gradient_norm - 1)**2)
        return penalty

    def get_gradient(self, D, real, fake, epsilon, inputs):

        mixed_images = real * epsilon + fake * (1 - epsilon)

        mixed_scores = D(mixed_images, inputs)
        
        gradient = torch.autograd.grad(
            inputs=mixed_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores), 
            create_graph=True,
            retain_graph=True,)[0]
        
        return gradient

    def get_gen_loss(self, crit_fake_pred):
        gen_loss = -1. * torch.mean(crit_fake_pred)
        return gen_loss
    
    def get_crit_loss(self, crit_fake_pred, crit_real_pred, gp, c_lambda):
        crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp
        return crit_loss

    def generate_fake_target(self, G, inputs_origin, noise, real_batch_size):
        nb_false_speak = int(real_batch_size/6)
        nb_false_gen = real_batch_size - 2*nb_false_speak

        with torch.no_grad():
            output_eye, output_pose_r, output_au = G(inputs_origin, noise) #the generator generates the false data conditional on the prosody 
            fake_targets = torch.cat((output_eye, output_pose_r, output_au), 2)

        
        tab_fake_inputs_1 = []
        tab_fake_targets_1 = []
        for i in range(nb_false_speak):
            max_idx = len(self.no_speak_x)-100 if len(self.no_speak_x) < len(self.speak_y) else len(self.speak_y)-100
            idx = random.randint(0, max_idx)
            fake_inputs_1 = self.no_speak_x[idx: idx+100].to(self.device).float()
            fake_targets_1 = self.speak_y[idx: idx+100].to(self.device).float()
            tab_fake_inputs_1.append(fake_inputs_1)
            tab_fake_targets_1.append(fake_targets_1)
        result_fake_inputs_1 = torch.stack(tab_fake_inputs_1)
        result_targets_1 = torch.stack(tab_fake_targets_1)

        tab_fake_inputs_2 = []
        tab_fake_targets_2 = []
        for i in range(nb_false_speak):
            max_idx = len(self.speak_x)-100 if len(self.speak_x) < len(self.no_speak_y) else len(self.no_speak_y)-100
            idx = random.randint(0, max_idx)
            fake_inputs_2 = self.speak_x[idx: idx+100].to(self.device).float()
            fake_targets_2 = self.no_speak_y[idx: idx+100].to(self.device).float()
            tab_fake_inputs_2.append(fake_inputs_2)
            tab_fake_targets_2.append(fake_targets_2)
        result_fake_inputs_2 = torch.stack(tab_fake_inputs_2)
        result_targets_2 = torch.stack(tab_fake_targets_2)

        max_idx = len(inputs_origin)-nb_false_gen
        idx = random.randint(0, max_idx)
        
        tab_fake_inputs = torch.cat((result_fake_inputs_1, result_fake_inputs_2, inputs_origin[idx:idx+nb_false_gen]),0)
        tab_fake_targets = torch.cat((result_targets_1, result_targets_2, fake_targets[idx:idx+nb_false_gen]),0)

        return tab_fake_inputs, tab_fake_targets
        
    
    def train_model_w(self):
        G = Generator().to(self.device)
        D = Discriminator().to(self.device)

        self.b1 = 0.5
        self.b2 = 0.999
        critic_iter = 5
        c_lambda = 10

        g_optimizer = torch.optim.Adam(G.parameters(), lr=constants.g_lr, weight_decay=0.00001, betas=(self.b1, self.b2))
        d_optimizer = torch.optim.Adam(D.parameters(), lr=constants.d_lr, weight_decay=0.00001, betas=(self.b1, self.b2))

        save_params(constants.saved_path, G, D)
        
        bce_loss = torch.nn.BCELoss()
        criterion = torch.nn.MSELoss()
        noise_g = NoiseGenerator(self.device)

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        one = one.to(self.device)
        mone = mone.to(self.device)

        print("Starting Training Loop...")
        for epoch in range(1, self.n_epochs + 1):
            print(f"Starting epoch {epoch}/{constants.n_epochs}...")
            start_epoch = datetime.now()
            self.reinitialize_loss()
            for iteration, data in enumerate(self.trainloader, 1):
                # if (iteration % 10 == 0 or iteration == self.n_iteration_per_epoch):
                #     print("*"+f"Starting iteration {iteration}/{self.n_iteration_per_epoch}...")
                torch.cuda.empty_cache()
                # * Configure real data
                inputs, targets = data[0].to(self.device), data[1].to(self.device)
                real_batch_size = inputs.shape[0]
                
                #get target by type of data and transform to float
                inputs, targets, target_eye, target_pose_r, target_au = self.format_data(inputs, targets)
                
                mean_iteration_d_loss = 0
                mean_real_pred = 0
                mean_fake_pred = 0
                for _ in range(critic_iter):
                    ### Update critic ###
                    for param in D.parameters():
                        param.grad = None
                    noise = noise_g.getNoise(constants.noise_size, real_batch_size) 

                    # Train with fake images
                    if(constants.fake_target):
                        fake_inputs_d, fake_targets_d = self.generate_fake_target(G, inputs, noise, real_batch_size)
                    else:
                        fake_inputs_d = inputs
                        gen_eye, gen_pose_r, gen_au = G(fake_inputs_d, noise)
                        fake_targets_d = torch.cat((gen_eye, gen_pose_r, gen_au), 2)

                    d_fake_pred = D(fake_targets_d.detach(), fake_inputs_d)
                    mean_fake_pred += torch.mean(d_fake_pred).item() / critic_iter

                    d_real_pred = D(targets, inputs)
                    mean_real_pred += torch.mean(d_real_pred).item() / critic_iter

                    epsilon = torch.rand(len(targets), 1, 1, device=self.device, requires_grad=True)
                    gradient = self.get_gradient(D, targets, fake_targets_d.detach(), epsilon, inputs)
                    gp = self.gradient_penalty(gradient)
                    d_loss = self.get_crit_loss(d_fake_pred, d_real_pred, gp, c_lambda)

                    # Keep track of the average critic loss in this batch
                    mean_iteration_d_loss += d_loss.item() / critic_iter
                    # Update gradients
                    d_loss.backward(retain_graph=True)
                    # Update optimizer
                    d_optimizer.step()
                self.current_d_loss += mean_iteration_d_loss
                self.current_real_pred += mean_real_pred 
                self.current_fake_pred += mean_fake_pred

                for param in G.parameters():
                    param.grad = None
                # train generator
                # compute loss with fake images
                noise = noise_g.getNoise(constants.noise_size, real_batch_size) 
                gen_eye, gen_pose_r, gen_au = G(inputs, noise)
                fake_targets = torch.cat((gen_eye, gen_pose_r, gen_au), 2)
                d_fake_pred = D(fake_targets, inputs)

                g_loss = self.get_gen_loss(d_fake_pred)
                loss_eye = criterion(gen_eye, target_eye)
                loss_pose_r = criterion(gen_pose_r, target_pose_r)
                loss_au = criterion(gen_au, target_au)

                g_total_loss = constants.eye_coeff * loss_eye + constants.pose_coeff * loss_pose_r + constants.au_coeff * loss_au + constants.adversarial_coeff * g_loss

                g_total_loss.backward()
                g_optimizer.step()
                self.current_loss += g_total_loss.item()


            self.t_loss, self.t_loss_eye, self.t_loss_pose_r, self.t_loss_au = self.test_loss_w(G, D, self.testloader, criterion, criterion)
            self.update_loss_tab(iteration)
            print("[",epoch,"]","last iteration :", "g_loss", g_loss.item(), ",loss_eye", loss_eye.item(), ",loss pose", loss_pose_r.item(), ",loss au", loss_au.item())
            print('[ %d ] g_loss : %.4f, t_loss : %.4f, d_loss : %.4f' % (epoch, self.current_loss, self.t_loss, self.current_d_loss))
            print('[ %d ] pred : %.4f %.4f' % (epoch, self.current_real_pred, self.current_fake_pred))

            print(f'Generator iteration: {epoch}/{constants.n_epochs}, g_loss: {g_loss}')
            diff_in_loss = abs(self.current_real_pred - self.current_fake_pred)
            if (epoch % constants.log_interval == 0) and epoch >= 20:
                print("saving...")
                if(diff_in_loss <= 0.2):
                    print("Close prediction : ", diff_in_loss)
                saveModel(G, epoch, constants.saved_path)
                plotHistLossEpochGAN(epoch, self.d_loss_tab, self.loss_tab, self.t_loss_tab)
                plotHistPredEpochGAN(epoch, self.d_real_pred_tab, self.d_fake_pred_tab)
                plotHistAllLossEpoch(epoch, self.loss_tab_eye, self.t_loss_tab_eye, self.loss_tab_pose_r, self.t_loss_tab_pose_r, self.loss_tab_au, self.t_loss_tab_au)

            end_epoch = datetime.now()   
            diff = end_epoch - start_epoch
            print("Duration of epoch :" + str(diff.total_seconds()))
                
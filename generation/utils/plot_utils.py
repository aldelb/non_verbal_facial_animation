from matplotlib import pyplot as plt
import constants.constants as constants
import numpy as np

def plotHistLossEpoch(num_epoch, loss, t_loss=None):
    plt.figure(dpi=100)
    plt.plot(range(num_epoch), loss, label='loss')
    if(t_loss != None):
        plt.plot(range(num_epoch+1), t_loss, label='test loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(constants.saved_path+f'loss_epoch_{num_epoch}.png')
    plt.close()

def plotHistAllLossEpoch(num_epoch, loss_eye, t_loss_eye, loss_pose_r, t_loss_pose_r, loss_au, t_loss_au):
    plt.figure(dpi=100)
    plt.plot(range(num_epoch), loss_eye, color="darkgreen", label='Loss gaze - Train')
    plt.plot(range(num_epoch), t_loss_eye, color="limegreen", label='Loss gaze - Test')

    plt.plot(range(num_epoch), loss_pose_r, color="darkblue", label='Loss pose r - Train')
    plt.plot(range(num_epoch), t_loss_pose_r, color="cornflowerblue", label='Loss pose r - Test')

    plt.plot(range(num_epoch), loss_au, color="red", label='Loss AU - Train')
    plt.plot(range(num_epoch), t_loss_au, color="lightcoral", label='Loss AU - Test')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(constants.saved_path+f'all_loss_epoch_{num_epoch}.png')
    plt.close()

def plotHistLossEpochGAN(num_epoch, d_loss, g_loss, t_loss=None):
    plt.figure(dpi=100)
    plt.plot(range(num_epoch), d_loss, label='discriminator loss')
    plt.plot(range(num_epoch), g_loss, label='generator loss')
    if(t_loss != None):
        plt.plot(range(num_epoch), t_loss, label='test loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(constants.saved_path+f'loss_epoch_{num_epoch}.png')
    plt.close()


def plotHistPredEpochGAN(num_epoch, d_real_pred, d_fake_pred):
    plt.figure(dpi=100)
    plt.plot(range(num_epoch), d_real_pred, label='discriminator real prediction')
    plt.plot(range(num_epoch), d_fake_pred, label='discriminator fake prediction')
    plt.yticks(np.arange(0, 1, step=0.2)) 
    plt.xlabel("Epoch")
    plt.ylabel("Discriminator prediction")
    plt.legend()
    plt.savefig(constants.saved_path+f'pred_epoch_{num_epoch}.png')
    plt.close()
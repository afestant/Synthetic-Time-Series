import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import datetime
import yaml
from ts_dataset import TSDataset
from tensorboardX import SummaryWriter
from models.recurrent_models import LSTMGenerator, LSTMDiscriminator
from models.convolutional_models import CausalConvGenerator, CausalConvDiscriminator

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from PIL import Image


class TSGANSynthetiser:
    def __init__(self, path_to_yaml='tsgan_configuration.yml', writer=None):

        """
        Args:
            path_to_yaml (string) : path to yml configuration file
        """
        try: 
            with open (path_to_yaml, 'r') as file:
                self.config = yaml.safe_load(file)
        except Exception:
            print('Error reading the config file')

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    

        self.workers = self.config['workers']
        self.batch_size = self.config['batch_size']
        #dimensionality of the latent vector
        self.nz = int(self.config['nz'])
        self.delta_condition = self.config['delta_condition']
        self.alternate = self.config['alternate']
        self.delta_lambda = self.config['delta_lambda']
        self.in_dim = self.nz + 1 if self.delta_condition else self.nz  
        self.dis_type = self.config['dis_type']
        self.gen_type = self.config['gen_type']
        self.lr = self.config['lr']
        self.epochs = self.config['epochs']
        self.outfile = self.config['outfile']
        self.tensorboard_image_every = self.config['tensorboard_image_every']
        self.checkpoint_every = self.config['checkpoint_every']
        self.outf = self.config['outf']
        self.run_tag = self.config['run_tag']
        self.imf = self.config['imf']
        if writer: 
            self.writer=writer

    def fit(self, dataset_path, ts_col, value_col, timew, normalize=True):
        """Fit the CTSGAN Synthesizer models to the training data.
        Args:
            dataset_path (string): path to csv file
            ts_col (string) : datetime column name
            value_col (string): value column name
            timew (int): number of points for building time sequencences
            normalize (bool): whether to normalize the data in [-1,1]
        """

        self.dataset = TSDataset(csv_file=dataset_path, 
                                timestamp_col=ts_col, 
                                value_col=value_col, 
                                time_window=timew, 
                                normalize=True)
        self.seq_len = timew   #same as the first dimension of a sequence in the dataset self.dataset[0].size(0)
        dataloader = torch.utils.data.DataLoader(self.dataset, 
                                                batch_size=self.batch_size, 
                                                shuffle=True,
                                                num_workers=int(self.workers))

        if self.dis_type == "lstm": 
            self.netD = LSTMDiscriminator(in_dim=1, 
                                          hidden_dim=256).to(self.device)
        if self.dis_type == "cnn":
            self.netD = CausalConvDiscriminator(input_size=1, 
                                                n_layers=8, 
                                                n_channel=10, 
                                                kernel_size=8, 
                                                dropout=0).to(self.device)
        if self.gen_type == "lstm":
            self.netG = LSTMGenerator(in_dim=self.in_dim, 
                                      out_dim=1, 
                                      hidden_dim=256).to(self.device)
        if self.gen_type == "cnn":
            self.netG = CausalConvGenerator(noise_size=self.in_dim, 
                                            output_size=1, 
                                            n_layers=8, 
                                            n_channel=10, 
                                            kernel_size=8, 
                                            dropout=0.2).to(self.device)
    
        assert self.netG
        assert self.netD

        print("|Discriminator Architecture|\n", self.netD)
        print("|Generator Architecture|\n", self.netG)

        criterion = nn.BCELoss().to(self.device)
        delta_criterion = nn.MSELoss().to(self.device)     

        #Generate fixed noise to be used for visualization
        fixed_noise = torch.randn(self.batch_size, 
                                  self.seq_len, 
                                  self.nz, 
                                  device=self.device)

        if self.delta_condition :
        #Sample both deltas and noise for visualization
            deltas = self.dataset.sample_deltas(self.batch_size).unsqueeze(2).repeat(1, self.seq_len, 1)
            fixed_noise = torch.cat((fixed_noise, deltas), dim=2)

        real_label = 1
        fake_label = 0

        # setup optimizer
        optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr)
        optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr)  

        print(f'Number of epochs {self.epochs}')
        print(f'Length of the dataloader {len(dataloader)}')
        for epoch in range(self.epochs):
            for i, data in enumerate(dataloader, 0):
                #print(f'batch={i} batch_length={len(data)}, shape_batch={data.shape}')
                niter = epoch * len(dataloader) + i
        
                #Save just first batch of real data for displaying
                if i == 0:
                    real_display = data.cpu()
            
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################

                #Train with real data
                self.netD.zero_grad()
                real = data.to(self.device)
                batch_size, seq_len = real.size(0), real.size(1)
                label = torch.full((batch_size, seq_len, 1), real_label, device=self.device)

                output = self.netD(real)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()
                
                #Train with fake data
                noise = torch.randn(batch_size, seq_len, self.nz, device=self.device)
                if self.delta_condition:
                    #Sample a delta for each batch and concatenate to the noise for each timestep
                    deltas = self.dataset.sample_deltas(batch_size).unsqueeze(2).repeat(1, seq_len, 1)
                    noise = torch.cat((noise, deltas), dim=2)
                fake = self.netG(noise)
                label.fill_(fake_label)
                output = self.netD(fake.detach())
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()
                
                #Visualize discriminator gradients
                if self.writer:
                    for name, param in self.netD.named_parameters():
                        self.writer.add_histogram("DiscriminatorGradients/{}".format(name), param.grad, niter)

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                label.fill_(real_label) 
                output = self.netD(fake)
                errG = criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                

                if self.delta_condition:
                    #If option is passed, alternate between the losses instead of using their sum
                    if self.alternate:
                        optimizerG.step()
                        self.netG.zero_grad()
                    noise = torch.randn(batch_size, seq_len, self.nz, device=self.device)
                    deltas = self.dataset.sample_deltas(batch_size).unsqueeze(2).repeat(1, seq_len, 1)
                    noise = torch.cat((noise, deltas), dim=2)
                    #Generate sequence given noise w/ deltas and deltas
                    out_seqs = self.netG(noise)
                    delta_loss = self.delta_lambda * delta_criterion(out_seqs[:, -1] - out_seqs[:, 0], deltas[:,0])
                    delta_loss.backward()
                
                optimizerG.step()
                
                #Visualize generator gradients
                if self.writer:
                    for name, param in self.netG.named_parameters():
                        self.writer.add_histogram("GeneratorGradients/{}".format(name), param.grad, niter)
                
                ###########################
                # (3) Supervised update of G network: minimize mse of input deltas and actual deltas of generated sequences
                ###########################

                #Report metrics
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' 
                    % (epoch, self.epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2), end='')
                if self.delta_condition and self.writer:
                    self.writer.add_scalar('MSE of deltas of generated sequences', delta_loss.item(), niter)
                    print(' DeltaMSE: %.4f' % (delta_loss.item()/self.delta_lambda), end='')
                print()
                if self.writer:
                    self.writer.add_scalar('DiscriminatorLoss', errD.item(), niter)
                    self.writer.add_scalar('GeneratorLoss', errG.item(), niter)
                    self.writer.add_scalar('D of X', D_x, niter) 
                    self.writer.add_scalar('D of G of z', D_G_z1, niter)
                
            ##### End of the epoch #####
            real_plot = self.time_series_to_plot(self.dataset.denormalize(real_display))
            if self.writer:
                if (epoch % self.tensorboard_image_every == 0) or (epoch == (self.epochs - 1)):
                    self.writer.add_image("Real", real_plot, epoch)
            
            fake = self.netG(fixed_noise)
            fake_plot = self.time_series_to_plot(self.dataset.denormalize(fake))
            #torchvision.utils.save_image(fake_plot, os.path.join(self.imf, self.run_tag+'_epoch'+str(epoch)+'.jpg'))
            fp = os.path.join(self.imf, self.run_tag+'_epoch'+str(epoch)+'.jpg')
            ndarr = fake_plot.mul(255)
            ndarr = ndarr.add(0.5)
            ndarr = ndarr.clamp(0, 255)
            ndarr = ndarr.permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            im.save(fp)
            if self.writer:
                if (epoch % self.tensorboard_image_every == 0) or (epoch == (self.epochs - 1)):
                    self.writer.add_image("Fake", fake_plot, epoch)
                                    
            # Checkpoint
            if (epoch % self.checkpoint_every == 0) or (epoch == (self.epochs - 1)):
                torch.save(self.netG, '%s/%s_netG_epoch_%d.pth' % (self.outf, self.run_tag, epoch))
                torch.save(self.netD, '%s/%s_netD_epoch_%d.pth' % (self.outf, self.run_tag, epoch))


    def sample_data(self, n):
        """Sample data similar to the training data.
        Args:
            n (int):
                Number of rows to sample.
        Returns:
            numpy.ndarray or pandas.DataFrame
        """

        self.delta_list = self.config['deltas']

        #If conditional generation is required, then input for generator must contain deltas
        if self.delta_list:
            noise = torch.randn(len(self.delta_list), self.seq_len, self.nz) 
            deltas = torch.FloatTensor(self.delta_list).view(-1, 1, 1).repeat(1, self.seq_len, 1)
            if self.dataset:
                #Deltas are provided in original range, normalization required
                deltas = self.dataset.normalize_deltas(deltas)
            noise = torch.cat((noise, deltas), dim=2)
        else:
            noise = torch.randn(n, self.seq_len, self.nz) 
        
        out_list = []
        for batch in noise.split(self.batch_size):
            out_list.append(self.netG(batch))
        out_tensor = torch.cat(out_list, dim=0)
         
        #Puts generated sequences in original range
        if self.dataset:
            out_tensor = self.dataset.denormalize(out_tensor)
 
        if self.outfile:
            np.save(self.outfile, out_tensor.detach().numpy())
        
        return out_tensor.squeeze().detach().numpy()

    
    def time_series_to_plot(self, time_series_batch, dpi=35, feature_idx=0, n_images_per_row=4, titles=None):
        """Convert a batch of time series to a tensor with a grid of their plots
        
        Args:
            time_series_batch (Tensor): (batch_size, seq_len, dim) tensor of time series
            dpi (int): dpi of a single image
            feature_idx (int): index of the feature that goes in the plots (the first one by default)
            n_images_per_row (int): number of images per row in the plot
            titles (list of strings): list of titles for the plots

        Output:
            single (channels, width, height)-shaped tensor representing an image
        """

        #Iterates over the time series
        images = []
        for i, series in enumerate(time_series_batch.detach()):
            fig = plt.figure(dpi=dpi)
            ax = fig.add_subplot(1,1,1)
            if titles:
                ax.set_title(titles[i])
            ax.plot(series[:, feature_idx].numpy()) #plots a single feature of the time series
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(data)
            plt.close(fig)

        #Swap channel
        images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
        #Make grid
        grid_image = vutils.make_grid(images.detach(), nrow=n_images_per_row)
        return grid_image

    def tensor_to_string_list(self, tensor):
        """Convert a tensor to a list of strings representing its value"""
        scalar_list = tensor.squeeze().numpy().tolist()
        return ["%.5f" % scalar for scalar in scalar_list]

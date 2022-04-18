import torch
from .base_model import BaseModel
from . import networks


class DeepLIIFModel(BaseModel):
    """ This class implements the DeepLIIF model, for learning a mapping from input images to modalities given paired data."""

    def __init__(self, opt):
        """Initialize the DeepLIIF class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        self.mod_gen_no = self.opt.modalities_no
        self.seg_gen_no = self.opt.modalities_no + 1

        # weights of the modalities in generating segmentation mask
        self.seg_weights = opt.seg_weights
        assert len(self.seg_weights) == self.seg_gen_no, 'The number of the segmentation weights (seg_weights) is not equal to the number of target images (modalities_no)!'
        print(self.seg_weights)
        # loss weights in calculating the final loss
        self.loss_G_weights = [1 / self.seg_gen_no] * self.seg_gen_no
        self.loss_D_weights = [1 / self.seg_gen_no] * self.seg_gen_no

        self.loss_names = []
        self.visual_names = ['real_A']
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        for i in range(1, self.seg_gen_no + 1):
            self.loss_names.extend(['G_GAN_' + str(i), 'G_L1_' + str(i), 'D_real_' + str(i), 'D_fake_' + str(i)])
            self.visual_names.extend(['fake_B_' + str(i), 'real_B_' + str(i)])

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.is_train:
            self.model_names = []
            for i in range(1, self.mod_gen_no + 1):
                self.model_names.extend(['G_' + str(i), 'D_' + str(i)])

            for i in range(1, self.seg_gen_no + 1):
                self.model_names.extend(['GS_' + str(i), 'DS_' + str(i)])

        else:  # during test time, only load G
            self.model_names = []
            for i in range(1, self.seg_gen_no + 1):
                self.model_names.extend(['G_' + str(i)])

            for i in range(1, self.seg_gen_no + 1):
                self.model_names.extend(['GS_' + str(i)])

        # define networks (both generator and discriminator)
        self.netG = [None for _ in range(self.mod_gen_no)]
        self.netGS = [None for _ in range(self.seg_gen_no)]
        for i in range(self.mod_gen_no):
            self.netG[i] = networks.define_G(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, self.opt.net_g, self.opt.norm,
                                             not self.opt.no_dropout, self.opt.init_type, self.opt.init_gain, self.opt.gpu_ids, self.opt.padding)
            print('***************************************')
            print(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, self.opt.net_g, self.opt.norm,
                                             not self.opt.no_dropout, self.opt.init_type, self.opt.init_gain, self.opt.gpu_ids, self.opt.padding)
            print('***************************************')
        for i in range(self.seg_gen_no):
            if self.seg_weights[i] > 0:
                self.netGS[i] = networks.define_G(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, self.opt.net_gs, self.opt.norm,
                                                  not self.opt.no_dropout, self.opt.init_type, self.opt.init_gain, self.opt.gpu_ids)

        if self.is_train:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = [None for _ in range(self.mod_gen_no)]
            self.netDS = [None for _ in range(self.seg_gen_no)]
            for i in range(self.mod_gen_no):
                self.netD[i] = networks.define_D(self.opt.input_nc + self.opt.output_nc, self.opt.ndf, self.opt.net_d,
                                                 self.opt.n_layers_D, self.opt.norm, self.opt.init_type, self.opt.init_gain,
                                                 self.opt.gpu_ids)
            for i in range(self.seg_gen_no):
                if self.seg_weights[i] > 0:
                    self.netDS[i] = networks.define_D(self.opt.input_nc + self.opt.output_nc, self.opt.ndf, self.opt.net_ds,
                                                      self.opt.n_layers_D, self.opt.norm, self.opt.init_type, self.opt.init_gain,
                                                      self.opt.gpu_ids)

        if self.is_train:
            # define loss functions
            self.criterionGAN_mod = networks.GANLoss(self.opt.gan_mode).to(self.device)
            self.criterionGAN_seg = networks.GANLoss(self.opt.gan_mode_s).to(self.device)

            self.criterionSmoothL1 = torch.nn.SmoothL1Loss()

            self.criterionVGG = networks.VGGLoss().to(self.device)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            params = []
            for i in range(len(self.netG)):
                params += list(self.netG[i].parameters())
            for i in range(len(self.netGS)):
                if self.netGS[i]:
                    params += list(self.netGS[i].parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            params = []
            for i in range(len(self.netD)):
                params += list(self.netD[i].parameters())
            for i in range(len(self.netDS)):
                if self.netDS[i]:
                    params += list(self.netDS[i].parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.

        :param input (dict): include the input image and the output modalities
        """
        self.real_A = input['A'].to(self.device)

        self.real_B_array = input['B']
        self.real_B = []
        for i in range(self.mod_gen_no + 1):
            self.real_B.append(self.real_B_array[i].to(self.device))

        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = []
        for i in range(self.mod_gen_no):
            self.fake_B.append(self.netG[i](self.real_A))

        merged_fake_BS_all = []
        for i in range(self.seg_gen_no):
            if self.netGS[i]:
                real_image = self.real_A if i == 0 else self.fake_B[i - 1]
                fake_BS = self.netGS[i](real_image)
                merged_fake_BS_all.append(torch.mul(fake_BS, self.seg_weights[i]))

        self.fake_B.append(torch.stack(merged_fake_BS_all).sum(dim=0))

    def backward_D(self):
        """Calculate GAN loss for the discriminators"""
        pred_fake = []
        for i in range(self.mod_gen_no):
            pred_fake.append(self.netD[i](torch.cat((self.real_A, self.fake_B[i]), 1).detach()))

        pred_fake_s_merged = []
        for i in range(self.seg_gen_no):
            if self.netDS[i]:
                real_image = self.real_A if i == 0 else self.real_B[i - 1]
                pred_fake_s = self.netDS[i](torch.cat((real_image, self.fake_B[-1]), 1).detach())
                pred_fake_s_merged.append(torch.mul(pred_fake_s, self.seg_weights[i]))

        pred_fake.append(torch.stack(pred_fake_s_merged).sum(dim=0))

        self.loss_D_fake = []
        for i in range(self.mod_gen_no):
            self.loss_D_fake.append(self.criterionGAN_mod(pred_fake[i], False))
        self.loss_D_fake.append(self.criterionGAN_seg(pred_fake[-1], False))

        pred_real = []
        for i in range(self.mod_gen_no):
            pred_real.append(self.netD[i](torch.cat((self.real_A, self.real_B[i]), 1)))

        pred_real_s_merged = []
        for i in range(self.seg_gen_no):
            if self.netDS[i]:
                real_image = self.real_A if i == 0 else self.real_B[i - 1]
                pred_real_s = self.netDS[i](torch.cat((real_image, self.real_B[-1]), 1))
                pred_real_s_merged.append(torch.mul(pred_real_s, self.seg_weights[i]))

        pred_real.append(torch.stack(pred_real_s_merged).sum(dim=0))

        self.loss_D_real = []
        for i in range(self.mod_gen_no):
            self.loss_D_real.append(self.criterionGAN_mod(pred_real[i], True))
        self.loss_D_real.append(self.criterionGAN_seg(pred_real[-1], True))

        # combine losses and calculate gradients
        # self.loss_D = (self.loss_D_fake[0] + self.loss_D_real[0]) * 0.5 * self.loss_D_weights[0]
        self.loss_D = torch.tensor(0., device=self.device)
        for i in range(0, self.seg_gen_no):
            self.loss_D += (self.loss_D_fake[i] + self.loss_D_real[i]) * 0.5 * self.loss_D_weights[i]

        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        pred_fake = []
        for i in range(self.mod_gen_no):
            pred_fake.append(self.netD[i](torch.cat((self.real_A, self.fake_B[i]), 1)))

        pred_fake_s_merged = []
        for i in range(self.seg_gen_no):
            if self.netDS[i]:
                real_image = self.real_A if i == 0 else self.real_B[i - 1]
                pred_fake_s = self.netDS[i](torch.cat((real_image, self.fake_B[-1]), 1))
                pred_fake_s_merged.append(torch.mul(pred_fake_s, self.seg_weights[i]))

        pred_fake.append(torch.stack(pred_fake_s_merged).sum(dim=0))

        self.loss_G_GAN = []
        for i in range(self.mod_gen_no):
            self.loss_G_GAN.append(self.criterionGAN_mod(pred_fake[i], True))
        self.loss_G_GAN.append(self.criterionGAN_seg(pred_fake[-1], True))

        # Second, G(A) = B
        self.loss_G_L1 = []
        for i in range(self.seg_gen_no):
            self.loss_G_L1.append(self.criterionSmoothL1(self.fake_B[i], self.real_B[i]) * self.opt.lambda_L1)

        self.loss_G_VGG = []
        for i in range(self.mod_gen_no):
            self.loss_G_VGG.append(self.criterionVGG(self.fake_B[i], self.real_B[i]) * self.opt.lambda_feat)

        # self.loss_G = (self.loss_G_GAN[0] + self.loss_G_L1[0]) * self.loss_G_weights[0]
        self.loss_G = torch.tensor(0., device=self.device)
        for i in range(0, self.seg_gen_no):
            loss_VGG = self.loss_G_VGG[i - 1] if i > 0 else 0
            self.loss_G += (self.loss_G_GAN[i] + self.loss_G_L1[i] + loss_VGG) * self.loss_G_weights[i]

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        for i in range(self.mod_gen_no):
            self.set_requires_grad(self.netD[i], True)  # enable backprop for D1
        for i in range(self.seg_gen_no):
            if self.netDS[i]:
                self.set_requires_grad(self.netDS[i], True)

        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights

        # update G
        for i in range(self.mod_gen_no):
            self.set_requires_grad(self.netD[i], False)
        for i in range(self.seg_gen_no):
            if self.netDS[i]:
                self.set_requires_grad(self.netDS[i], False)

        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
import torch
from .base_model import BaseModel
from . import networks
from .networks import get_optimizer
from . import init_nets, run_dask, get_opt
from torch import nn

class DeepLIIFKDModel(BaseModel):
    """ This class implements the DeepLIIF model, for learning a mapping from input images to modalities given paired data."""

    def __init__(self, opt):
        """Initialize the DeepLIIF class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        if not hasattr(opt,'net_gs'):
            opt.net_gs = 'unet_512'
        
        self.seg_weights = opt.seg_weights
        self.loss_G_weights = opt.loss_G_weights
        self.loss_D_weights = opt.loss_D_weights
        
        if not opt.is_train:
            self.gpu_ids = [] # avoid the models being loaded as DP
        else:
            self.gpu_ids = opt.gpu_ids

        self.loss_names = []
        self.visual_names = ['real_A']
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        for i in range(1, self.opt.modalities_no + 1 + 1):
            self.loss_names.extend([f'G_GAN_{i}', f'G_L1_{i}', f'D_real_{i}', f'D_fake_{i}', f'G_KLDiv_{i}', f'G_KLDiv_5_{i}'])
            self.visual_names.extend([f'fake_B_{i}', f'fake_B_5_{i}', f'fake_B_{i}_teacher', f'fake_B_5_{i}_teacher', f'real_B_{i}'])

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.is_train:
            self.model_names = []
            for i in range(1, self.opt.modalities_no + 1):
                self.model_names.extend(['G' + str(i), 'D' + str(i)])

            for i in range(1, self.opt.modalities_no + 1 + 1):
                self.model_names.extend(['G5' + str(i), 'D5' + str(i)])
        else:  # during test time, only load G
            self.model_names = []
            for i in range(1, self.opt.modalities_no + 1):
                self.model_names.extend(['G' + str(i)])

            for i in range(1, self.opt.modalities_no + 1 + 1):
                self.model_names.extend(['G5' + str(i)])

        # define networks (both generator and discriminator)
        if isinstance(opt.netG, str):
            opt.netG = [opt.netG] * 4
        if isinstance(opt.net_gs, str):
            opt.net_gs = [opt.net_gs]*5

            
        self.netG1 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG[0], opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.padding)
        self.netG2 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG[1], opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.padding)
        self.netG3 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG[2], opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.padding)
        self.netG4 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG[3], opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.padding)

        # DeepLIIF model currently uses one gs arch because there is only one explicit seg mod output
        self.netG51 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.net_gs[0], opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG52 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.net_gs[1], opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG53 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.net_gs[2], opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG54 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.net_gs[3], opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG55 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.net_gs[4], opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.is_train:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD1 = networks.define_D(opt.input_nc+opt.output_nc , opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD2 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD3 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD4 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            self.netD51 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD52 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD53 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD54 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD55 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
                                          
            # load the teacher model
            self.opt_teacher = get_opt(opt.model_dir_teacher, mode='test')
            self.opt_teacher.gpu_ids = opt.gpu_ids # use student's gpu_ids
            self.nets_teacher = init_nets(opt.model_dir_teacher, eager_mode=True, opt=self.opt_teacher, phase='test')
            

        if self.is_train:
            # define loss functions
            self.criterionGAN_BCE = networks.GANLoss('vanilla').to(self.device)
            self.criterionGAN_lsgan = networks.GANLoss('lsgan').to(self.device)
            self.criterionSmoothL1 = torch.nn.SmoothL1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            params = list(self.netG1.parameters()) + list(self.netG2.parameters()) + list(self.netG3.parameters()) + list(self.netG4.parameters()) + list(self.netG51.parameters()) + list(self.netG52.parameters()) + list(self.netG53.parameters()) + list(self.netG54.parameters()) + list(self.netG55.parameters())
            try:
                self.optimizer_G = get_optimizer(opt.optimizer)(params, lr=opt.lr_g, betas=(opt.beta1, 0.999))
            except:
                print(f'betas are not used for optimizer torch.optim.{opt.optimizer} in generators')
                self.optimizer_G = get_optimizer(opt.optimizer)(params, lr=opt.lr_g)

            params = list(self.netD1.parameters()) + list(self.netD2.parameters()) + list(self.netD3.parameters()) + list(self.netD4.parameters()) + list(self.netD51.parameters()) + list(self.netD52.parameters()) + list(self.netD53.parameters()) + list(self.netD54.parameters()) + list(self.netD55.parameters())
            try:
                self.optimizer_D = get_optimizer(opt.optimizer)(params, lr=opt.lr_d, betas=(opt.beta1, 0.999))
            except:
                print(f'betas are not used for optimizer torch.optim.{opt.optimizer} in discriminators')
                self.optimizer_D = get_optimizer(opt.optimizer)(params, lr=opt.lr_d)

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.criterionVGG = networks.VGGLoss().to(self.device)
            self.criterionKLDiv = torch.nn.KLDivLoss(reduction='batchmean').to(self.device)
            self.softmax = torch.nn.Softmax(dim=-1).to(self.device) # apply softmax on the last dim
            self.logsoftmax = torch.nn.LogSoftmax(dim=-1).to(self.device) # apply log-softmax on the last dim

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.

        :param input (dict): include the input image and the output modalities
        """

        self.real_A = input['A'].to(self.device)

        self.real_B_array = input['B']
        self.real_B_1 = self.real_B_array[0].to(self.device)
        self.real_B_2 = self.real_B_array[1].to(self.device)
        self.real_B_3 = self.real_B_array[2].to(self.device)
        self.real_B_4 = self.real_B_array[3].to(self.device)
        self.real_B_5 = self.real_B_array[4].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B_1 = self.netG1(self.real_A)   # Hematoxylin image generator
        self.fake_B_2 = self.netG2(self.real_A)   # mpIF DAPI image generator
        self.fake_B_3 = self.netG3(self.real_A)   # mpIF Lap2 image generator
        self.fake_B_4 = self.netG4(self.real_A)   # mpIF Ki67 image generator

        self.fake_B_5_1 = self.netG51(self.real_A)      # Segmentation mask generator from IHC input image
        self.fake_B_5_2 = self.netG52(self.fake_B_1)    # Segmentation mask generator from Hematoxylin input image
        self.fake_B_5_3 = self.netG53(self.fake_B_2)    # Segmentation mask generator from mpIF DAPI input image
        self.fake_B_5_4 = self.netG54(self.fake_B_3)    # Segmentation mask generator from mpIF Lap2 input image
        self.fake_B_5_5 = self.netG55(self.fake_B_4)    # Segmentation mask generator from mpIF Lap2 input image
        self.fake_B_5 = torch.stack([torch.mul(self.fake_B_5_1, self.seg_weights[0]),
                                     torch.mul(self.fake_B_5_2, self.seg_weights[1]),
                                     torch.mul(self.fake_B_5_3, self.seg_weights[2]),
                                     torch.mul(self.fake_B_5_4, self.seg_weights[3]),
                                     torch.mul(self.fake_B_5_5, self.seg_weights[4])]).sum(dim=0)
        
        fakes_teacher = run_dask(img=self.real_A, nets=self.nets_teacher, opt=self.opt_teacher, use_dask=False, output_tensor=True)
        for k,v in fakes_teacher.items():
            suffix = k[1:] # starts with G
            suffix = '_'.join(list(suffix)) # 51 -> 5_1
            setattr(self,f'fake_B_{suffix}_teacher',v)

    def backward_D(self):
        """Calculate GAN loss for the discriminators"""
        fake_AB_1 = torch.cat((self.real_A, self.fake_B_1), 1)  # Conditional GANs; feed IHC input and Hematoxtlin output to the discriminator
        fake_AB_2 = torch.cat((self.real_A, self.fake_B_2), 1)  # Conditional GANs; feed IHC input and mpIF DAPI output to the discriminator
        fake_AB_3 = torch.cat((self.real_A, self.fake_B_3), 1)  # Conditional GANs; feed IHC input and mpIF Lap2 output to the discriminator
        fake_AB_4 = torch.cat((self.real_A, self.fake_B_4), 1)  # Conditional GANs; feed IHC input and mpIF Ki67 output to the discriminator

        pred_fake_1 = self.netD1(fake_AB_1.detach())
        pred_fake_2 = self.netD2(fake_AB_2.detach())
        pred_fake_3 = self.netD3(fake_AB_3.detach())
        pred_fake_4 = self.netD4(fake_AB_4.detach())

        fake_AB_5_1 = torch.cat((self.real_A, self.fake_B_5), 1)    # Conditional GANs; feed IHC input and Segmentation mask output to the discriminator
        fake_AB_5_2 = torch.cat((self.real_B_1, self.fake_B_5), 1)  # Conditional GANs; feed Hematoxylin input and Segmentation mask output to the discriminator
        fake_AB_5_3 = torch.cat((self.real_B_2, self.fake_B_5), 1)  # Conditional GANs; feed mpIF DAPI input and Segmentation mask output to the discriminator
        fake_AB_5_4 = torch.cat((self.real_B_3, self.fake_B_5), 1)  # Conditional GANs; feed mpIF Lap2 input and Segmentation mask output to the discriminator
        fake_AB_5_5 = torch.cat((self.real_B_4, self.fake_B_5), 1)  # Conditional GANs; feed mpIF Lap2 input and Segmentation mask output to the discriminator

        pred_fake_5_1 = self.netD51(fake_AB_5_1.detach())
        pred_fake_5_2 = self.netD52(fake_AB_5_2.detach())
        pred_fake_5_3 = self.netD53(fake_AB_5_3.detach())
        pred_fake_5_4 = self.netD54(fake_AB_5_4.detach())
        pred_fake_5_5 = self.netD55(fake_AB_5_5.detach())

        pred_fake_5 = torch.stack(
            [torch.mul(pred_fake_5_1, self.seg_weights[0]),
             torch.mul(pred_fake_5_2, self.seg_weights[1]),
             torch.mul(pred_fake_5_3, self.seg_weights[2]),
             torch.mul(pred_fake_5_4, self.seg_weights[3]),
             torch.mul(pred_fake_5_5, self.seg_weights[4])]).sum(dim=0)

        self.loss_D_fake_1 = self.criterionGAN_BCE(pred_fake_1, False)
        self.loss_D_fake_2 = self.criterionGAN_BCE(pred_fake_2, False)
        self.loss_D_fake_3 = self.criterionGAN_BCE(pred_fake_3, False)
        self.loss_D_fake_4 = self.criterionGAN_BCE(pred_fake_4, False)
        self.loss_D_fake_5 = self.criterionGAN_lsgan(pred_fake_5, False)


        real_AB_1 = torch.cat((self.real_A, self.real_B_1), 1)
        real_AB_2 = torch.cat((self.real_A, self.real_B_2), 1)
        real_AB_3 = torch.cat((self.real_A, self.real_B_3), 1)
        real_AB_4 = torch.cat((self.real_A, self.real_B_4), 1)

        pred_real_1 = self.netD1(real_AB_1)
        pred_real_2 = self.netD2(real_AB_2)
        pred_real_3 = self.netD3(real_AB_3)
        pred_real_4 = self.netD4(real_AB_4)

        real_AB_5_1 = torch.cat((self.real_A, self.real_B_5), 1)
        real_AB_5_2 = torch.cat((self.real_B_1, self.real_B_5), 1)
        real_AB_5_3 = torch.cat((self.real_B_2, self.real_B_5), 1)
        real_AB_5_4 = torch.cat((self.real_B_3, self.real_B_5), 1)
        real_AB_5_5 = torch.cat((self.real_B_4, self.real_B_5), 1)

        pred_real_5_1 = self.netD51(real_AB_5_1)
        pred_real_5_2 = self.netD52(real_AB_5_2)
        pred_real_5_3 = self.netD53(real_AB_5_3)
        pred_real_5_4 = self.netD54(real_AB_5_4)
        pred_real_5_5 = self.netD55(real_AB_5_5)

        pred_real_5 = torch.stack(
            [torch.mul(pred_real_5_1, self.seg_weights[0]),
             torch.mul(pred_real_5_2, self.seg_weights[1]),
             torch.mul(pred_real_5_3, self.seg_weights[2]),
             torch.mul(pred_real_5_4, self.seg_weights[3]),
             torch.mul(pred_real_5_5, self.seg_weights[4])]).sum(dim=0)

        self.loss_D_real_1 = self.criterionGAN_BCE(pred_real_1, True)
        self.loss_D_real_2 = self.criterionGAN_BCE(pred_real_2, True)
        self.loss_D_real_3 = self.criterionGAN_BCE(pred_real_3, True)
        self.loss_D_real_4 = self.criterionGAN_BCE(pred_real_4, True)
        self.loss_D_real_5 = self.criterionGAN_lsgan(pred_real_5, True)

        # combine losses and calculate gradients
        self.loss_D = (self.loss_D_fake_1 + self.loss_D_real_1) * 0.5 * self.loss_D_weights[0] + \
                      (self.loss_D_fake_2 + self.loss_D_real_2) * 0.5 * self.loss_D_weights[1] + \
                      (self.loss_D_fake_3 + self.loss_D_real_3) * 0.5 * self.loss_D_weights[2] + \
                      (self.loss_D_fake_4 + self.loss_D_real_4) * 0.5 * self.loss_D_weights[3] + \
                      (self.loss_D_fake_5 + self.loss_D_real_5) * 0.5 * self.loss_D_weights[4]

        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        fake_AB_1 = torch.cat((self.real_A, self.fake_B_1), 1)
        fake_AB_2 = torch.cat((self.real_A, self.fake_B_2), 1)
        fake_AB_3 = torch.cat((self.real_A, self.fake_B_3), 1)
        fake_AB_4 = torch.cat((self.real_A, self.fake_B_4), 1)

        fake_AB_5_1 = torch.cat((self.real_A, self.fake_B_5), 1)
        fake_AB_5_2 = torch.cat((self.real_B_1, self.fake_B_5), 1)
        fake_AB_5_3 = torch.cat((self.real_B_2, self.fake_B_5), 1)
        fake_AB_5_4 = torch.cat((self.real_B_3, self.fake_B_5), 1)
        fake_AB_5_5 = torch.cat((self.real_B_4, self.fake_B_5), 1)

        pred_fake_1 = self.netD1(fake_AB_1)
        pred_fake_2 = self.netD2(fake_AB_2)
        pred_fake_3 = self.netD3(fake_AB_3)
        pred_fake_4 = self.netD4(fake_AB_4)

        pred_fake_5_1 = self.netD51(fake_AB_5_1)
        pred_fake_5_2 = self.netD52(fake_AB_5_2)
        pred_fake_5_3 = self.netD53(fake_AB_5_3)
        pred_fake_5_4 = self.netD54(fake_AB_5_4)
        pred_fake_5_5 = self.netD55(fake_AB_5_5)
        pred_fake_5 = torch.stack(
            [torch.mul(pred_fake_5_1, self.seg_weights[0]),
             torch.mul(pred_fake_5_2, self.seg_weights[1]),
             torch.mul(pred_fake_5_3, self.seg_weights[2]),
             torch.mul(pred_fake_5_4, self.seg_weights[3]),
             torch.mul(pred_fake_5_5, self.seg_weights[4])]).sum(dim=0)

        self.loss_G_GAN_1 = self.criterionGAN_BCE(pred_fake_1, True)
        self.loss_G_GAN_2 = self.criterionGAN_BCE(pred_fake_2, True)
        self.loss_G_GAN_3 = self.criterionGAN_BCE(pred_fake_3, True)
        self.loss_G_GAN_4 = self.criterionGAN_BCE(pred_fake_4, True)
        self.loss_G_GAN_5 = self.criterionGAN_lsgan(pred_fake_5, True)

        # Second, G(A) = B
        self.loss_G_L1_1 = self.criterionSmoothL1(self.fake_B_1, self.real_B_1) * self.opt.lambda_L1
        self.loss_G_L1_2 = self.criterionSmoothL1(self.fake_B_2, self.real_B_2) * self.opt.lambda_L1
        self.loss_G_L1_3 = self.criterionSmoothL1(self.fake_B_3, self.real_B_3) * self.opt.lambda_L1
        self.loss_G_L1_4 = self.criterionSmoothL1(self.fake_B_4, self.real_B_4) * self.opt.lambda_L1
        self.loss_G_L1_5 = self.criterionSmoothL1(self.fake_B_5, self.real_B_5) * self.opt.lambda_L1

        self.loss_G_VGG_1 = self.criterionVGG(self.fake_B_1, self.real_B_1) * self.opt.lambda_feat
        self.loss_G_VGG_2 = self.criterionVGG(self.fake_B_2, self.real_B_2) * self.opt.lambda_feat
        self.loss_G_VGG_3 = self.criterionVGG(self.fake_B_3, self.real_B_3) * self.opt.lambda_feat
        self.loss_G_VGG_4 = self.criterionVGG(self.fake_B_4, self.real_B_4) * self.opt.lambda_feat
        
        
        # .view(1,1,-1) reshapes the input (batch_size, 3, 512, 512) to (batch_size, 1, 3*512*512)
        # softmax/log-softmax is then applied on the concatenated vector of size (1, 3*512*512)
        # this normalizes the pixel values across all 3 RGB channels
        # the resulting vectors are then used to compute KL divergence loss
        self.loss_G_KLDiv_1 = self.criterionKLDiv(self.logsoftmax(self.fake_B_1.view(1,1,-1)), self.softmax(self.fake_B_1_teacher.view(1,1,-1)))
        self.loss_G_KLDiv_2 = self.criterionKLDiv(self.logsoftmax(self.fake_B_2.view(1,1,-1)), self.softmax(self.fake_B_2_teacher.view(1,1,-1)))
        self.loss_G_KLDiv_3 = self.criterionKLDiv(self.logsoftmax(self.fake_B_3.view(1,1,-1)), self.softmax(self.fake_B_3_teacher.view(1,1,-1)))
        self.loss_G_KLDiv_4 = self.criterionKLDiv(self.logsoftmax(self.fake_B_4.view(1,1,-1)), self.softmax(self.fake_B_4_teacher.view(1,1,-1)))
        self.loss_G_KLDiv_5 = self.criterionKLDiv(self.logsoftmax(self.fake_B_5.view(1,1,-1)), self.softmax(self.fake_B_5_teacher.view(1,1,-1)))
        self.loss_G_KLDiv_5_1 = self.criterionKLDiv(self.logsoftmax(self.fake_B_5_1.view(1,1,-1)), self.softmax(self.fake_B_5_1_teacher.view(1,1,-1)))
        self.loss_G_KLDiv_5_2 = self.criterionKLDiv(self.logsoftmax(self.fake_B_5_2.view(1,1,-1)), self.softmax(self.fake_B_5_2_teacher.view(1,1,-1)))
        self.loss_G_KLDiv_5_3 = self.criterionKLDiv(self.logsoftmax(self.fake_B_5_3.view(1,1,-1)), self.softmax(self.fake_B_5_3_teacher.view(1,1,-1)))
        self.loss_G_KLDiv_5_4 = self.criterionKLDiv(self.logsoftmax(self.fake_B_5_4.view(1,1,-1)), self.softmax(self.fake_B_5_4_teacher.view(1,1,-1)))
        self.loss_G_KLDiv_5_5 = self.criterionKLDiv(self.logsoftmax(self.fake_B_5_5.view(1,1,-1)), self.softmax(self.fake_B_5_5_teacher.view(1,1,-1)))

        self.loss_G = (self.loss_G_GAN_1 + self.loss_G_L1_1 + self.loss_G_VGG_1) * self.loss_G_weights[0] + \
                      (self.loss_G_GAN_2 + self.loss_G_L1_2 + self.loss_G_VGG_2) * self.loss_G_weights[1] + \
                      (self.loss_G_GAN_3 + self.loss_G_L1_3 + self.loss_G_VGG_3) * self.loss_G_weights[2] + \
                      (self.loss_G_GAN_4 + self.loss_G_L1_4 + self.loss_G_VGG_4) * self.loss_G_weights[3] + \
                      (self.loss_G_GAN_5 + self.loss_G_L1_5) * self.loss_G_weights[4] + \
                      (self.loss_G_KLDiv_1 + self.loss_G_KLDiv_2 + self.loss_G_KLDiv_3 + self.loss_G_KLDiv_4 + \
                      self.loss_G_KLDiv_5 + self.loss_G_KLDiv_5_1 + self.loss_G_KLDiv_5_2 + self.loss_G_KLDiv_5_3 + \
                      self.loss_G_KLDiv_5_4 + self.loss_G_KLDiv_5_5) * 10

        # combine loss and calculate gradients
        # self.loss_G = (self.loss_G_GAN_1 + self.loss_G_L1_1) * self.loss_G_weights[0] + \
        #               (self.loss_G_GAN_2 + self.loss_G_L1_2) * self.loss_G_weights[1] + \
        #               (self.loss_G_GAN_3 + self.loss_G_L1_3) * self.loss_G_weights[2] + \
        #               (self.loss_G_GAN_4 + self.loss_G_L1_4) * self.loss_G_weights[3] + \
        #               (self.loss_G_GAN_5 + self.loss_G_L1_5) * self.loss_G_weights[4]
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD1, True)  # enable backprop for D1
        self.set_requires_grad(self.netD2, True)  # enable backprop for D2
        self.set_requires_grad(self.netD3, True)  # enable backprop for D3
        self.set_requires_grad(self.netD4, True)  # enable backprop for D4
        self.set_requires_grad(self.netD51, True)  # enable backprop for D51
        self.set_requires_grad(self.netD52, True)  # enable backprop for D52
        self.set_requires_grad(self.netD53, True)  # enable backprop for D53
        self.set_requires_grad(self.netD54, True)  # enable backprop for D54
        self.set_requires_grad(self.netD55, True)  # enable backprop for D54

        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights

        # update G
        self.set_requires_grad(self.netD1, False)  # D1 requires no gradients when optimizing G1
        self.set_requires_grad(self.netD2, False)  # D2 requires no gradients when optimizing G2
        self.set_requires_grad(self.netD3, False)  # D3 requires no gradients when optimizing G3
        self.set_requires_grad(self.netD4, False)  # D4 requires no gradients when optimizing G4
        self.set_requires_grad(self.netD51, False)  # D51 requires no gradients when optimizing G51
        self.set_requires_grad(self.netD52, False)  # D52 requires no gradients when optimizing G52
        self.set_requires_grad(self.netD53, False)  # D53 requires no gradients when optimizing G53
        self.set_requires_grad(self.netD54, False)  # D54 requires no gradients when optimizing G54
        self.set_requires_grad(self.netD55, False)  # D54 requires no gradients when optimizing G54

        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
    
    def calculate_losses(self):
        """
        Calculate losses but do not optimize parameters. Used in validation loss calculation during training.
        """
        
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD1, True)  # enable backprop for D1
        self.set_requires_grad(self.netD2, True)  # enable backprop for D2
        self.set_requires_grad(self.netD3, True)  # enable backprop for D3
        self.set_requires_grad(self.netD4, True)  # enable backprop for D4
        self.set_requires_grad(self.netD51, True)  # enable backprop for D51
        self.set_requires_grad(self.netD52, True)  # enable backprop for D52
        self.set_requires_grad(self.netD53, True)  # enable backprop for D53
        self.set_requires_grad(self.netD54, True)  # enable backprop for D54
        self.set_requires_grad(self.netD55, True)  # enable backprop for D54

        self.optimizer_D.zero_grad()        # set D's gradients to zero
        self.backward_D()                # calculate gradients for D

        # update G
        self.set_requires_grad(self.netD1, False)  # D1 requires no gradients when optimizing G1
        self.set_requires_grad(self.netD2, False)  # D2 requires no gradients when optimizing G2
        self.set_requires_grad(self.netD3, False)  # D3 requires no gradients when optimizing G3
        self.set_requires_grad(self.netD4, False)  # D4 requires no gradients when optimizing G4
        self.set_requires_grad(self.netD51, False)  # D51 requires no gradients when optimizing G51
        self.set_requires_grad(self.netD52, False)  # D52 requires no gradients when optimizing G52
        self.set_requires_grad(self.netD53, False)  # D53 requires no gradients when optimizing G53
        self.set_requires_grad(self.netD54, False)  # D54 requires no gradients when optimizing G54
        self.set_requires_grad(self.netD55, False)  # D54 requires no gradients when optimizing G54

        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
            

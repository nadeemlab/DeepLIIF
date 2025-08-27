import torch
from .base_model import BaseModel
from . import networks
from .networks import get_optimizer


class DeepLIIFModel(BaseModel):
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
        if not hasattr(self,'mod_id_seg') and hasattr(opt,'mod_id_seg'): # use mod id seg from train opt file if available
            self.mod_id_seg = self.opt.mod_id_seg
        elif not hasattr(self,'mod_id_seg') and not hasattr(opt,'modalities_names'): # backward compatible with models trained before this param was introduced
            self.mod_id_seg = self.opt.modalities_no + 1 # for original DeepLIIF, modalities_no is 4 and the seg mod id is 5
        elif not hasattr(self,'mod_id_seg'):
            self.mod_id_seg = 'S'
        print('Initializing DeepLIIF model with segmentation modality id:',self.mod_id_seg)
        
        if not opt.is_train:
            self.gpu_ids = [] # avoid the models being loaded as DP
        else:
            self.gpu_ids = opt.gpu_ids

        self.loss_names = []
        self.visual_names = ['real_A']
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        for i in range(self.opt.modalities_no):
            self.loss_names.extend([f'G_GAN_{i+1}', f'G_L1_{i+1}', f'D_real_{i+1}', f'D_fake_{i+1}'])
            self.visual_names.extend([f'fake_B_{i+1}', f'real_B_{i+1}'])
        self.loss_names.extend([f'G_GAN_{self.mod_id_seg}',f'G_L1_{self.mod_id_seg}',f'D_real_{self.mod_id_seg}',f'D_fake_{self.mod_id_seg}'])
        
        for i in range(self.opt.modalities_no+1):
            self.visual_names.extend([f'fake_B_{self.mod_id_seg}{i}']) # 0 is used for the base input mod
        self.visual_names.extend([f'fake_B_{self.mod_id_seg}', f'real_B_{self.mod_id_seg}'])

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.is_train:
            self.model_names = []
            for i in range(1, self.opt.modalities_no + 1):
                self.model_names.extend([f'G{i}', f'D{i}'])

            for i in range(self.opt.modalities_no + 1):  # 0 is used for the base input mod
                self.model_names.extend([f'G{self.mod_id_seg}{i}', f'D{self.mod_id_seg}{i}'])
        else:  # during test time, only load G
            self.model_names = []
            for i in range(1, self.opt.modalities_no + 1):
                self.model_names.extend([f'G{i}'])

            for i in range(self.opt.modalities_no + 1):  # 0 is used for the base input mod
                self.model_names.extend([f'G{self.mod_id_seg}{i}'])

        # define networks (both generator and discriminator)
        if isinstance(opt.netG, str):
            opt.netG = [opt.netG] * self.opt.modalities_no
        if isinstance(opt.net_gs, str):
            opt.net_gs = [opt.net_gs] * (self.opt.modalities_no + 1) # +1 for base input mod

        
        for i in range(self.opt.modalities_no):
            setattr(self,f'netG{i+1}',networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG[i], opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.padding))

        # DeepLIIF model currently uses one gs arch because there is only one explicit seg mod output
        for i in range(self.opt.modalities_no+1):
            setattr(self,f'netG{self.mod_id_seg}{i}',networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.net_gs[i], opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids))

        if self.is_train:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            for i in range(self.opt.modalities_no):
                setattr(self,f'netD{i+1}',networks.define_D(opt.input_nc+opt.output_nc , opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids))

            for i in range(self.opt.modalities_no+1):
                setattr(self,f'netD{self.mod_id_seg}{i}',networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids))

        if self.is_train:
            # define loss functions
            self.criterionGAN_BCE = networks.GANLoss('vanilla').to(self.device)
            self.criterionGAN_lsgan = networks.GANLoss('lsgan').to(self.device)
            self.criterionSmoothL1 = torch.nn.SmoothL1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            #params = list(self.netG1.parameters()) + list(self.netG2.parameters()) + list(self.netG3.parameters()) + list(self.netG4.parameters()) + list(self.netG51.parameters()) + list(self.netG52.parameters()) + list(self.netG53.parameters()) + list(self.netG54.parameters()) + list(self.netG55.parameters())
            params = []
            for i in range(self.opt.modalities_no):
                params += list(getattr(self,f'netG{i+1}').parameters())

            for i in range(self.opt.modalities_no+1):
                params += list(getattr(self,f'netG{self.mod_id_seg}{i}').parameters())

            try:
                self.optimizer_G = get_optimizer(opt.optimizer)(params, lr=opt.lr_g, betas=(opt.beta1, 0.999))
            except:
                print(f'betas are not used for optimizer torch.optim.{opt.optimizer} in generators')
                self.optimizer_G = get_optimizer(opt.optimizer)(params, lr=opt.lr_g)

            #params = list(self.netD1.parameters()) + list(self.netD2.parameters()) + list(self.netD3.parameters()) + list(self.netD4.parameters()) + list(self.netD51.parameters()) + list(self.netD52.parameters()) + list(self.netD53.parameters()) + list(self.netD54.parameters()) + list(self.netD55.parameters())
            params = []
            for i in range(self.opt.modalities_no):
                params += list(getattr(self,f'netD{i+1}').parameters())

            for i in range(self.opt.modalities_no+1):
                params += list(getattr(self,f'netD{self.mod_id_seg}{i}').parameters())

            try:
                self.optimizer_D = get_optimizer(opt.optimizer)(params, lr=opt.lr_d, betas=(opt.beta1, 0.999))
            except:
                print(f'betas are not used for optimizer torch.optim.{opt.optimizer} in discriminators')
                self.optimizer_D = get_optimizer(opt.optimizer)(params, lr=opt.lr_d)

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.criterionVGG = networks.VGGLoss().to(self.device)

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.

        :param input (dict): include the input image and the output modalities
        """

        self.real_A = input['A'].to(self.device)

        self.real_B_array = input['B']
        for i in range(self.opt.modalities_no):
            setattr(self,f'real_B_{i+1}',self.real_B_array[i].to(self.device))
        setattr(self,f'real_B_{self.mod_id_seg}',self.real_B_array[self.opt.modalities_no].to(self.device)) # the last one is seg
        
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.fake_B_1 = self.netG1(self.real_A)   # Hematoxylin image generator
        # self.fake_B_2 = self.netG2(self.real_A)   # mpIF DAPI image generator
        # self.fake_B_3 = self.netG3(self.real_A)   # mpIF Lap2 image generator
        # self.fake_B_4 = self.netG4(self.real_A)   # mpIF Ki67 image generator
        
        for i in range(self.opt.modalities_no):
            setattr(self,f'fake_B_{i+1}',getattr(self,f'netG{i+1}')(self.real_A))

        # self.fake_B_5_1 = self.netG51(self.real_A)      # Segmentation mask generator from IHC input image
        # self.fake_B_5_2 = self.netG52(self.fake_B_1)    # Segmentation mask generator from Hematoxylin input image
        # self.fake_B_5_3 = self.netG53(self.fake_B_2)    # Segmentation mask generator from mpIF DAPI input image
        # self.fake_B_5_4 = self.netG54(self.fake_B_3)    # Segmentation mask generator from mpIF Lap2 input image
        # self.fake_B_5_5 = self.netG55(self.fake_B_4)    # Segmentation mask generator from mpIF Lap2 input image
        # self.fake_B_5 = torch.stack([torch.mul(self.fake_B_5_1, self.seg_weights[0]),
        #                              torch.mul(self.fake_B_5_2, self.seg_weights[1]),
        #                              torch.mul(self.fake_B_5_3, self.seg_weights[2]),
        #                              torch.mul(self.fake_B_5_4, self.seg_weights[3]),
        #                              torch.mul(self.fake_B_5_5, self.seg_weights[4])]).sum(dim=0)
        for i in range(self.opt.modalities_no+1):
            if i == 0:
                setattr(self,f'fake_B_{self.mod_id_seg}_{i}',getattr(self,f'netG{self.mod_id_seg}{i}')(self.real_A))
            else:
                setattr(self,f'fake_B_{self.mod_id_seg}_{i}',getattr(self,f'netG{self.mod_id_seg}{i}')(getattr(self,f'fake_B_{i}')))
            
        setattr(self,f'fake_B_{self.mod_id_seg}',torch.stack([torch.mul(getattr(self,f'fake_B_{self.mod_id_seg}_{i}'), self.seg_weights[i]) for i in range(self.opt.modalities_no+1)]).sum(dim=0))

    def backward_D(self):
        """Calculate GAN loss for the discriminators"""
        # fake_AB_1 = torch.cat((self.real_A, self.fake_B_1), 1)  # Conditional GANs; feed IHC input and Hematoxtlin output to the discriminator
        # fake_AB_2 = torch.cat((self.real_A, self.fake_B_2), 1)  # Conditional GANs; feed IHC input and mpIF DAPI output to the discriminator
        # fake_AB_3 = torch.cat((self.real_A, self.fake_B_3), 1)  # Conditional GANs; feed IHC input and mpIF Lap2 output to the discriminator
        # fake_AB_4 = torch.cat((self.real_A, self.fake_B_4), 1)  # Conditional GANs; feed IHC input and mpIF Ki67 output to the discriminator

        # pred_fake_1 = self.netD1(fake_AB_1.detach())
        # pred_fake_2 = self.netD2(fake_AB_2.detach())
        # pred_fake_3 = self.netD3(fake_AB_3.detach())
        # pred_fake_4 = self.netD4(fake_AB_4.detach())

        # self.loss_D_fake_1 = self.criterionGAN_BCE(pred_fake_1, False)
        # self.loss_D_fake_2 = self.criterionGAN_BCE(pred_fake_2, False)
        # self.loss_D_fake_3 = self.criterionGAN_BCE(pred_fake_3, False)
        # self.loss_D_fake_4 = self.criterionGAN_BCE(pred_fake_4, False)
        
        for i in range(self.opt.modalities_no):
            fake_AB = torch.cat((self.real_A, getattr(self,f'fake_B_{i+1}')), 1)
            pred_fake = getattr(self,f'netD{i+1}')(fake_AB.detach())
            setattr(self,f'loss_D_fake_{i+1}',self.criterionGAN_BCE(pred_fake, False))
            #setattr(self,f'fake_AB_{i+1}',torch.cat((self.real_A, getattr(self,f'fake_B_{i+1}')), 1))
            #setattr(self,f'pred_fake_{i+1}',getattr(self,f'netD{i+1}')(getattr))


        # fake_AB_5_1 = torch.cat((self.real_A, self.fake_B_5), 1)    # Conditional GANs; feed IHC input and Segmentation mask output to the discriminator
        # fake_AB_5_2 = torch.cat((self.real_B_1, self.fake_B_5), 1)  # Conditional GANs; feed Hematoxylin input and Segmentation mask output to the discriminator
        # fake_AB_5_3 = torch.cat((self.real_B_2, self.fake_B_5), 1)  # Conditional GANs; feed mpIF DAPI input and Segmentation mask output to the discriminator
        # fake_AB_5_4 = torch.cat((self.real_B_3, self.fake_B_5), 1)  # Conditional GANs; feed mpIF Lap2 input and Segmentation mask output to the discriminator
        # fake_AB_5_5 = torch.cat((self.real_B_4, self.fake_B_5), 1)  # Conditional GANs; feed mpIF Lap2 input and Segmentation mask output to the discriminator
        # 
        # pred_fake_5_1 = self.netD51(fake_AB_5_1.detach())
        # pred_fake_5_2 = self.netD52(fake_AB_5_2.detach())
        # pred_fake_5_3 = self.netD53(fake_AB_5_3.detach())
        # pred_fake_5_4 = self.netD54(fake_AB_5_4.detach())
        # pred_fake_5_5 = self.netD55(fake_AB_5_5.detach())
        # 
        # pred_fake_5 = torch.stack(
        #     [torch.mul(pred_fake_5_1, self.seg_weights[0]),
        #      torch.mul(pred_fake_5_2, self.seg_weights[1]),
        #      torch.mul(pred_fake_5_3, self.seg_weights[2]),
        #      torch.mul(pred_fake_5_4, self.seg_weights[3]),
        #      torch.mul(pred_fake_5_5, self.seg_weights[4])]).sum(dim=0)
        
        l_pred_fake_seg = []
        for i in range(self.opt.modalities_no+1):
            if i == 0:
                fake_AB_seg_i = torch.cat((self.real_A, getattr(self,f'fake_B_{self.mod_id_seg}')), 1)
            else:
                fake_AB_seg_i = torch.cat((getattr(self,f'real_B_{i}'), getattr(self,f'fake_B_{self.mod_id_seg}')), 1)
            
            pred_fake_seg_i = getattr(self,f'netD{self.mod_id_seg}{i}')(fake_AB_seg_i.detach())
            l_pred_fake_seg.append(torch.mul(pred_fake_seg_i, self.seg_weights[i]))
        pred_fake_seg = torch.stack(l_pred_fake_seg).sum(dim=0)

        #self.loss_D_fake_5 = self.criterionGAN_lsgan(pred_fake_5, False)
        setattr(self,f'loss_D_fake_{self.mod_id_seg}',self.criterionGAN_lsgan(pred_fake_seg, False))


        # real_AB_1 = torch.cat((self.real_A, self.real_B_1), 1)
        # real_AB_2 = torch.cat((self.real_A, self.real_B_2), 1)
        # real_AB_3 = torch.cat((self.real_A, self.real_B_3), 1)
        # real_AB_4 = torch.cat((self.real_A, self.real_B_4), 1)
        # 
        # pred_real_1 = self.netD1(real_AB_1)
        # pred_real_2 = self.netD2(real_AB_2)
        # pred_real_3 = self.netD3(real_AB_3)
        # pred_real_4 = self.netD4(real_AB_4)
        # 
        # self.loss_D_real_1 = self.criterionGAN_BCE(pred_real_1, True)
        # self.loss_D_real_2 = self.criterionGAN_BCE(pred_real_2, True)
        # self.loss_D_real_3 = self.criterionGAN_BCE(pred_real_3, True)
        # self.loss_D_real_4 = self.criterionGAN_BCE(pred_real_4, True)
        
        for i in range(self.opt.modalities_no):
            real_AB = torch.cat((self.real_A, getattr(self,f'real_B_{i+1}')), 1)
            pred_real = getattr(self,f'netD{i+1}')(real_AB)
            setattr(self,f'loss_D_real_{i+1}',self.criterionGAN_BCE(pred_real, True))

        # real_AB_5_1 = torch.cat((self.real_A, self.real_B_5), 1)
        # real_AB_5_2 = torch.cat((self.real_B_1, self.real_B_5), 1)
        # real_AB_5_3 = torch.cat((self.real_B_2, self.real_B_5), 1)
        # real_AB_5_4 = torch.cat((self.real_B_3, self.real_B_5), 1)
        # real_AB_5_5 = torch.cat((self.real_B_4, self.real_B_5), 1)
        # 
        # pred_real_5_1 = self.netD51(real_AB_5_1)
        # pred_real_5_2 = self.netD52(real_AB_5_2)
        # pred_real_5_3 = self.netD53(real_AB_5_3)
        # pred_real_5_4 = self.netD54(real_AB_5_4)
        # pred_real_5_5 = self.netD55(real_AB_5_5)
        # 
        # pred_real_5 = torch.stack(
        #     [torch.mul(pred_real_5_1, self.seg_weights[0]),
        #      torch.mul(pred_real_5_2, self.seg_weights[1]),
        #      torch.mul(pred_real_5_3, self.seg_weights[2]),
        #      torch.mul(pred_real_5_4, self.seg_weights[3]),
        #      torch.mul(pred_real_5_5, self.seg_weights[4])]).sum(dim=0)

        l_pred_real_seg = []
        for i in range(self.opt.modalities_no+1):
            if i == 0:
                real_AB_seg_i = torch.cat((self.real_A, getattr(self,f'real_B_{self.mod_id_seg}')), 1)
            else:
                real_AB_seg_i = torch.cat((getattr(self,f'real_B_{i}'), getattr(self,f'real_B_{self.mod_id_seg}')), 1)
            
            pred_real_seg_i = getattr(self,f'netD{self.mod_id_seg}{i}')(real_AB_seg_i)
            l_pred_real_seg.append(torch.mul(pred_real_seg_i, self.seg_weights[i]))
        pred_real_seg = torch.stack(l_pred_real_seg).sum(dim=0)

        #self.loss_D_real_5 = self.criterionGAN_lsgan(pred_real_5, True)
        setattr(self,f'loss_D_real_{self.mod_id_seg}',self.criterionGAN_lsgan(pred_real_seg, True))

        # combine losses and calculate gradients
        # self.loss_D = (self.loss_D_fake_1 + self.loss_D_real_1) * 0.5 * self.loss_D_weights[0] + \
        #               (self.loss_D_fake_2 + self.loss_D_real_2) * 0.5 * self.loss_D_weights[1] + \
        #               (self.loss_D_fake_3 + self.loss_D_real_3) * 0.5 * self.loss_D_weights[2] + \
        #               (self.loss_D_fake_4 + self.loss_D_real_4) * 0.5 * self.loss_D_weights[3] + \
        #               (self.loss_D_fake_5 + self.loss_D_real_5) * 0.5 * self.loss_D_weights[4]
        
        self.loss_D = torch.tensor(0., device=self.device)
        for i in range(self.opt.modalities_no):
            self.loss_D += (getattr(self,f'loss_D_fake_{i+1}') + getattr(self,f'loss_D_real_{i+1}')) * 0.5 * self.loss_D_weights[i]
        self.loss_D += (getattr(self,f'loss_D_fake_{self.mod_id_seg}') + getattr(self,f'loss_D_real_{self.mod_id_seg}')) * 0.5 * self.loss_D_weights[self.opt.modalities_no]

        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        # fake_AB_1 = torch.cat((self.real_A, self.fake_B_1), 1)
        # fake_AB_2 = torch.cat((self.real_A, self.fake_B_2), 1)
        # fake_AB_3 = torch.cat((self.real_A, self.fake_B_3), 1)
        # fake_AB_4 = torch.cat((self.real_A, self.fake_B_4), 1)
        # 
        # pred_fake_1 = self.netD1(fake_AB_1)
        # pred_fake_2 = self.netD2(fake_AB_2)
        # pred_fake_3 = self.netD3(fake_AB_3)
        # pred_fake_4 = self.netD4(fake_AB_4)
        # 
        # self.loss_G_GAN_1 = self.criterionGAN_BCE(pred_fake_1, True)
        # self.loss_G_GAN_2 = self.criterionGAN_BCE(pred_fake_2, True)
        # self.loss_G_GAN_3 = self.criterionGAN_BCE(pred_fake_3, True)
        # self.loss_G_GAN_4 = self.criterionGAN_BCE(pred_fake_4, True)
        
        for i in range(self.opt.modalities_no):
            fake_AB = torch.cat((self.real_A, getattr(self,f'fake_B_{i+1}')), 1)
            pred_fake = getattr(self,f'netD{i+1}')(fake_AB)
            setattr(self,f'loss_G_GAN_{i+1}',self.criterionGAN_BCE(pred_fake, True))

        # fake_AB_5_1 = torch.cat((self.real_A, self.fake_B_5), 1)
        # fake_AB_5_2 = torch.cat((self.real_B_1, self.fake_B_5), 1)
        # fake_AB_5_3 = torch.cat((self.real_B_2, self.fake_B_5), 1)
        # fake_AB_5_4 = torch.cat((self.real_B_3, self.fake_B_5), 1)
        # fake_AB_5_5 = torch.cat((self.real_B_4, self.fake_B_5), 1)
        # 
        # pred_fake_5_1 = self.netD51(fake_AB_5_1)
        # pred_fake_5_2 = self.netD52(fake_AB_5_2)
        # pred_fake_5_3 = self.netD53(fake_AB_5_3)
        # pred_fake_5_4 = self.netD54(fake_AB_5_4)
        # pred_fake_5_5 = self.netD55(fake_AB_5_5)
        # pred_fake_5 = torch.stack(
        #     [torch.mul(pred_fake_5_1, self.seg_weights[0]),
        #      torch.mul(pred_fake_5_2, self.seg_weights[1]),
        #      torch.mul(pred_fake_5_3, self.seg_weights[2]),
        #      torch.mul(pred_fake_5_4, self.seg_weights[3]),
        #      torch.mul(pred_fake_5_5, self.seg_weights[4])]).sum(dim=0)

        l_pred_fake_seg = []
        for i in range(self.opt.modalities_no+1):
            if i == 0:
                fake_AB_seg_i = torch.cat((self.real_A, getattr(self,f'fake_B_{self.mod_id_seg}')), 1)
            else:
                fake_AB_seg_i = torch.cat((getattr(self,f'real_B_{i}'), getattr(self,f'fake_B_{self.mod_id_seg}')), 1)
            
            pred_fake_seg_i = getattr(self,f'netD{self.mod_id_seg}{i}')(fake_AB_seg_i)
            l_pred_fake_seg.append(torch.mul(pred_fake_seg_i, self.seg_weights[i]))
        pred_fake_seg = torch.stack(l_pred_fake_seg).sum(dim=0)

        # self.loss_G_GAN_5 = self.criterionGAN_lsgan(pred_fake_5, True)
        setattr(self,f'loss_G_GAN_{self.mod_id_seg}',self.criterionGAN_lsgan(pred_fake_seg, True))

        # Second, G(A) = B
        # self.loss_G_L1_1 = self.criterionSmoothL1(self.fake_B_1, self.real_B_1) * self.opt.lambda_L1
        # self.loss_G_L1_2 = self.criterionSmoothL1(self.fake_B_2, self.real_B_2) * self.opt.lambda_L1
        # self.loss_G_L1_3 = self.criterionSmoothL1(self.fake_B_3, self.real_B_3) * self.opt.lambda_L1
        # self.loss_G_L1_4 = self.criterionSmoothL1(self.fake_B_4, self.real_B_4) * self.opt.lambda_L1
        # self.loss_G_L1_5 = self.criterionSmoothL1(self.fake_B_5, self.real_B_5) * self.opt.lambda_L1
        
        for i in range(self.opt.modalities_no):
            setattr(self,f'loss_G_L1_{i+1}',self.criterionSmoothL1(getattr(self,f'fake_B_{i+1}'), getattr(self,f'real_B_{i+1}')) * self.opt.lambda_L1)
        setattr(self,f'loss_G_L1_{self.mod_id_seg}',self.criterionSmoothL1(getattr(self,f'fake_B_{self.mod_id_seg}'), getattr(self,f'real_B_{self.mod_id_seg}')) * self.opt.lambda_L1)

        # self.loss_G_VGG_1 = self.criterionVGG(self.fake_B_1, self.real_B_1) * self.opt.lambda_feat
        # self.loss_G_VGG_2 = self.criterionVGG(self.fake_B_2, self.real_B_2) * self.opt.lambda_feat
        # self.loss_G_VGG_3 = self.criterionVGG(self.fake_B_3, self.real_B_3) * self.opt.lambda_feat
        # self.loss_G_VGG_4 = self.criterionVGG(self.fake_B_4, self.real_B_4) * self.opt.lambda_feat
        for i in range(self.opt.modalities_no):
            setattr(self,f'loss_G_VGG_{i+1}',self.criterionVGG(getattr(self,f'fake_B_{i+1}'), getattr(self,f'real_B_{i+1}')) * self.opt.lambda_feat)
        setattr(self,f'loss_G_VGG_{self.mod_id_seg}',self.criterionVGG(getattr(self,f'fake_B_{self.mod_id_seg}'), getattr(self,f'real_B_{self.mod_id_seg}')) * self.opt.lambda_feat)

        # self.loss_G = (self.loss_G_GAN_1 + self.loss_G_L1_1 + self.loss_G_VGG_1) * self.loss_G_weights[0] + \
        #               (self.loss_G_GAN_2 + self.loss_G_L1_2 + self.loss_G_VGG_2) * self.loss_G_weights[1] + \
        #               (self.loss_G_GAN_3 + self.loss_G_L1_3 + self.loss_G_VGG_3) * self.loss_G_weights[2] + \
        #               (self.loss_G_GAN_4 + self.loss_G_L1_4 + self.loss_G_VGG_4) * self.loss_G_weights[3] + \
        #               (self.loss_G_GAN_5 + self.loss_G_L1_5) * self.loss_G_weights[4]
        
        self.loss_G = torch.tensor(0., device=self.device)
        for i in range(self.opt.modalities_no):
            self.loss_G += (getattr(self,f'loss_G_GAN_{i+1}') + getattr(self,f'loss_G_L1_{i+1}') + getattr(self,f'loss_G_VGG_{i+1}')) * self.loss_G_weights[i]
        self.loss_G += (getattr(self,f'loss_G_GAN_{self.mod_id_seg}') + getattr(self,f'loss_G_L1_{self.mod_id_seg}')) * self.loss_G_weights[i]

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
        # self.set_requires_grad(self.netD1, True)  # enable backprop for D1
        # self.set_requires_grad(self.netD2, True)  # enable backprop for D2
        # self.set_requires_grad(self.netD3, True)  # enable backprop for D3
        # self.set_requires_grad(self.netD4, True)  # enable backprop for D4
        # self.set_requires_grad(self.netD51, True)  # enable backprop for D51
        # self.set_requires_grad(self.netD52, True)  # enable backprop for D52
        # self.set_requires_grad(self.netD53, True)  # enable backprop for D53
        # self.set_requires_grad(self.netD54, True)  # enable backprop for D54
        # self.set_requires_grad(self.netD55, True)  # enable backprop for D54
        
        for i in range(self.opt.modalities_no):
            self.set_requires_grad(getattr(self,f'netD{i+1}'), True)
        for i in range(self.opt.modalities_no+1):
            self.set_requires_grad(getattr(self,f'netD{self.mod_id_seg}{i}'), True)

        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights

        # update G
        # self.set_requires_grad(self.netD1, False)  # D1 requires no gradients when optimizing G1
        # self.set_requires_grad(self.netD2, False)  # D2 requires no gradients when optimizing G2
        # self.set_requires_grad(self.netD3, False)  # D3 requires no gradients when optimizing G3
        # self.set_requires_grad(self.netD4, False)  # D4 requires no gradients when optimizing G4
        # self.set_requires_grad(self.netD51, False)  # D51 requires no gradients when optimizing G51
        # self.set_requires_grad(self.netD52, False)  # D52 requires no gradients when optimizing G52
        # self.set_requires_grad(self.netD53, False)  # D53 requires no gradients when optimizing G53
        # self.set_requires_grad(self.netD54, False)  # D54 requires no gradients when optimizing G54
        # self.set_requires_grad(self.netD55, False)  # D54 requires no gradients when optimizing G54
        
        for i in range(self.opt.modalities_no):
            self.set_requires_grad(getattr(self,f'netD{i+1}'), False)
        for i in range(self.opt.modalities_no+1):
            self.set_requires_grad(getattr(self,f'netD{self.mod_id_seg}{i}'), False)

        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
    
    def calculate_losses(self):
        """
        Calculate losses but do not optimize parameters. Used in validation loss calculation during training.
        """
        
        self.forward()                   # compute fake images: G(A)
        # update D
        # self.set_requires_grad(self.netD1, True)  # enable backprop for D1
        # self.set_requires_grad(self.netD2, True)  # enable backprop for D2
        # self.set_requires_grad(self.netD3, True)  # enable backprop for D3
        # self.set_requires_grad(self.netD4, True)  # enable backprop for D4
        # self.set_requires_grad(self.netD51, True)  # enable backprop for D51
        # self.set_requires_grad(self.netD52, True)  # enable backprop for D52
        # self.set_requires_grad(self.netD53, True)  # enable backprop for D53
        # self.set_requires_grad(self.netD54, True)  # enable backprop for D54
        # self.set_requires_grad(self.netD55, True)  # enable backprop for D54
        
        for i in range(self.opt.modalities_no):
            self.set_requires_grad(getattr(self,f'netD{i+1}'), True)
        for i in range(self.opt.modalities_no+1):
            self.set_requires_grad(getattr(self,f'netD{self.mod_id_seg}{i}'), True)

        self.optimizer_D.zero_grad()        # set D's gradients to zero
        self.backward_D()                # calculate gradients for D

        # update G
        # self.set_requires_grad(self.netD1, False)  # D1 requires no gradients when optimizing G1
        # self.set_requires_grad(self.netD2, False)  # D2 requires no gradients when optimizing G2
        # self.set_requires_grad(self.netD3, False)  # D3 requires no gradients when optimizing G3
        # self.set_requires_grad(self.netD4, False)  # D4 requires no gradients when optimizing G4
        # self.set_requires_grad(self.netD51, False)  # D51 requires no gradients when optimizing G51
        # self.set_requires_grad(self.netD52, False)  # D52 requires no gradients when optimizing G52
        # self.set_requires_grad(self.netD53, False)  # D53 requires no gradients when optimizing G53
        # self.set_requires_grad(self.netD54, False)  # D54 requires no gradients when optimizing G54
        # self.set_requires_grad(self.netD55, False)  # D54 requires no gradients when optimizing G54
        
        for i in range(self.opt.modalities_no):
            self.set_requires_grad(getattr(self,f'netD{i+1}'), False)
        for i in range(self.opt.modalities_no+1):
            self.set_requires_grad(getattr(self,f'netD{self.mod_id_seg}{i}'), False)

        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
            


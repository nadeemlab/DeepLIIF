import torch
from torch import nn
import itertools
from ..util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .networks import get_optimizer

# https://github.com/brownvc/R3GAN/blob/main/R3GAN/Trainer.py
def ZeroCenteredGradientPenalty(Samples, Critics):
    
    # print(Samples.requires_grad, Critics.requires_grad)
    Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True)
    return Gradient.square().sum([1, 2, 3])

class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.mod_gen_no = self.opt.modalities_no
        if not hasattr(self.opt,'upsample'):
            self.opt.upsample = 'convtranspose'
        if not hasattr(self.opt,'label_smoothing'):
            self.opt.label_smoothing = 0
        
        use_spectral_norm = self.opt.norm == 'spectral'
        
        self.loss_G_weights = opt.loss_G_weights
        self.loss_D_weights = opt.loss_D_weights
        self.loss_cyc_weights = [1 / self.mod_gen_no] * self.mod_gen_no
        
        self.opt.lambda_identity = 0 # do not use lambda identity for the first trial
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        l_suffix = range(1, self.opt.modalities_no + 1)
        visual_names_A = [f'real_As_{i}' for i in l_suffix] + [f'fake_Bs_{i}' for i in l_suffix] + [f'rec_As_{i}' for i in l_suffix]
        visual_names_B = [f'real_Bs_{i}' for i in l_suffix] + [f'fake_As_{i}' for i in l_suffix] + [f'rec_Bs_{i}' for i in l_suffix]

        # if self.is_train and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
        #     visual_names_A.append('idt_B')
        #     visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.is_train:
            self.model_names = [f'GA_{i}' for i in l_suffix] + [f'GB_{i}' for i in l_suffix] + [f'DA_{i}' for i in l_suffix] + [f'DB_{i}' for i in l_suffix]
        else:  # during test time, only load Gs
            if self.opt.BtoA:
                self.model_names = [f'GB_{i}' for i in l_suffix] 
            else:
                self.model_names = [f'GA_{i}' for i in l_suffix] 

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        if isinstance(opt.net_g, str):
            self.opt.net_g = [self.opt.net_g] * self.mod_gen_no

        self.netGA = nn.ModuleList()
        self.netGB = nn.ModuleList()
        for i in range(self.mod_gen_no):
            if self.is_train or not self.opt.BtoA:
                self.netGA.append(networks.define_G(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, self.opt.net_g[i], self.opt.norm,
                                                 not self.opt.no_dropout, self.opt.init_type, self.opt.init_gain, self.gpu_ids, self.opt.padding, 
                                                upsample=self.opt.upsample))
            if self.is_train or self.opt.BtoA:
                self.netGB.append(networks.define_G(self.opt.output_nc, self.opt.input_nc, self.opt.ngf, self.opt.net_g[i], self.opt.norm,
                                                 not self.opt.no_dropout, self.opt.init_type, self.opt.init_gain, self.gpu_ids, self.opt.padding, 
                                                 upsample=self.opt.upsample))
        
        if self.is_train:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netDA = nn.ModuleList()
            self.netDB = nn.ModuleList()
            for i in range(self.mod_gen_no):
                self.netDA.append(networks.define_D(self.opt.output_nc, self.opt.ndf, self.opt.net_d,
                                                 self.opt.n_layers_D, self.opt.norm, self.opt.init_type, self.opt.init_gain,
                                                 self.gpu_ids))
                self.netDB.append(networks.define_D(self.opt.input_nc, self.opt.ndf, self.opt.net_d,
                                                 self.opt.n_layers_D, self.opt.norm, self.opt.init_type, self.opt.init_gain,
                                                 self.gpu_ids))



        if self.is_train:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pools = [ImagePool(opt.pool_size) for _ in range(self.opt.modalities_no)]  # create image buffer to store previously generated images
            self.fake_B_pools = [ImagePool(opt.pool_size) for _ in range(self.opt.modalities_no)]  # create image buffer to store previously generated images
            
            # define loss functions
            # label smoothing currently only applies to discriminator losses & generatoe of lsgan/vanilla
            self.criterionGAN = networks.GANLoss(opt.gan_mode, label_smoothing=self.opt.label_smoothing).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            
            self.criterionSmoothL1 = torch.nn.SmoothL1Loss()
            self.criterionVGG = networks.VGGLoss().to(self.device)
            
            
            # initialize optimizers
            params = []
            for i in range(len(self.netGA)):
                params += list(self.netGA[i].parameters())
            for i in range(len(self.netGB)):
                params += list(self.netGB[i].parameters())
            try:
                self.optimizer_G = get_optimizer(opt.optimizer)(params, lr=opt.lr_g, betas=(opt.beta1, 0.999))
            except:
                print(f'betas are not used for optimizer torch.optim.{opt.optimizer} in generators')
                self.optimizer_G = get_optimizer(opt.optimizer)(params, lr=opt.lr_g)
            
            params = []        
            for i in range(len(self.netDA)):
                params += list(self.netDA[i].parameters())
            for i in range(len(self.netDB)):
                params += list(self.netDB[i].parameters())
            
            # a smaller learning rate for discriminators to postpone training failure due to discriminators quickly become too strong
            try:
                self.optimizer_D = get_optimizer(opt.optimizer)(params, lr=opt.lr_d, betas=(opt.beta1, 0.999))
            except:
                print(f'betas are not used for optimizer torch.optim.{opt.optimizer} in generators')
                self.optimizer_D = get_optimizer(opt.optimizer)(params, lr=opt.lr_d)
            
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_As = [input['A'].to(self.device) for _ in range(self.opt.modalities_no)]
        self.real_Bs = [x.to(self.device) for x in input['Bs']]
        self.image_paths = input['A_paths']

    def forward(self):
        """
        Run forward pass; called by both functions <optimize_parameters> and <test>.
        During inference, some output list could be empty. For example, if only netGAs are loaded,
        there will not be valid elements in self.rec_As and self.fake_As.
        """
        self.fake_Bs = [netGA(real_A) for netGA, real_A in zip(self.netGA, self.real_As)]  # G_A(A)
        self.rec_As = [netGB(fake_B) for netGB, fake_B in zip(self.netGB, self.fake_Bs)]   # G_B(G_A(A))
        
        self.fake_As = [netGB(real_B) for netGB, real_B in zip(self.netGB, self.real_Bs)]  # G_B(B)
        self.rec_Bs = [netGA(fake_A) for netGA, fake_A in zip(self.netGA, self.fake_As)]   # G_A(G_B(B))
    
      
    def backward_D_basic(self, netD, real, fake, scale_factor=1):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5 * scale_factor
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_Bs = [fake_B_pool.query(fake_B) for fake_B_pool, fake_B in zip(self.fake_B_pools, self.fake_Bs)]
        real_Bs = self.real_Bs
        
        self.loss_D_A = 0
        for i, (netDA, real_B, fake_B) in enumerate(zip(self.netDA, real_Bs, fake_Bs)):
            self.loss_D_A += self.backward_D_basic(netDA, real_B, fake_B, scale_factor=self.loss_D_weights[i])
        #self.loss_D_A.backward()

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_As = [fake_A_pool.query(fake_A) for fake_A_pool, fake_A in zip(self.fake_A_pools, self.fake_As)]
        real_As = self.real_As
            
        self.loss_D_B = 0
        for i, (netDB, real_A, fake_A) in enumerate(zip(self.netDB, real_As, fake_As)):
            self.loss_D_B += self.backward_D_basic(netDB, real_A, fake_A, scale_factor=self.loss_D_weights[i]) 
        #self.loss_D_B.backward()

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        # default lambda values from cyclegan implementation:
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/c3268edd50ec37a81600c9b981841f48929671b8/models/cycle_gan_model.py#L41
        lambda_idt = 0#self.opt.lambda_identity # identity loss is used to preserve color consistency between input and output images, which we do not want to encourage
        lambda_A = 10#self.opt.lambda_A 
        lambda_B = 10#self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
            
        # GAN loss D_A(G_A(A))
        self.loss_G_A = 0
        for i, (netDA, fake_B, real_B) in enumerate(zip(self.netDA, self.fake_Bs, self.real_Bs)):
            self.loss_G_A += self.criterionGAN(netDA(fake_B), True) * self.loss_G_weights[i]
            self.loss_G_A += self.criterionVGG(fake_B, real_B) * self.loss_G_weights[i]
        
        # GAN loss D_B(G_B(B))
        self.loss_G_B = 0
        for i, (netDB, fake_A, real_A) in enumerate(zip(self.netDB, self.fake_As, self.real_As)):
            self.loss_G_B += self.criterionGAN(netDB(fake_A), True) * self.loss_G_weights[i]
            self.loss_G_B += self.criterionVGG(fake_A, real_A) * self.loss_G_weights[i]
            
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = 0
        for i, (rec_A, real_A) in enumerate(zip(self.rec_As, self.real_As)):
            self.loss_cycle_A += self.criterionCycle(rec_A, real_A) * lambda_A * self.loss_cyc_weights[i]
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = 0
        for i, (rec_B, real_B) in enumerate(zip(self.rec_Bs, self.real_Bs)):
            self.loss_cycle_B += self.criterionCycle(rec_B, real_B) * lambda_B * self.loss_cyc_weights[i]
        
        # VGG loss
        # self.loss_G_VGG = self.criterionVGG(self.fake_B_1, self.real_B_1) * self.opt.lambda_feat
        
        # smooth L1
        # self.loss_G_A_L1 = self.criterionSmoothL1(self.fake_B_1, self.real_B_1) * self.opt.lambda_L1
        
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad(self.netDA + self.netDB, False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        
        # D_A and D_B
        self.set_requires_grad(self.netDA + self.netDB, True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

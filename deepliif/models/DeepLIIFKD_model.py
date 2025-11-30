import torch
from .base_model import BaseModel
from . import networks
from .networks import get_optimizer
from . import init_nets, run_dask, get_opt
from torch import nn
from ..util.util import get_input_id, init_input_and_mod_id

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
        self.mod_id_seg, self.input_id = init_input_and_mod_id(opt) # creates self.input_id, self.mod_id_seg
        print(f'Initializing model with segmentation modality id {self.mod_id_seg}, input id {self.input_id}')
        
        if not opt.is_train:
            self.gpu_ids = [] # avoid the models being loaded as DP
        else:
            self.gpu_ids = opt.gpu_ids
        
        self.loss_names = []
        self.visual_names = ['real_A']
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        for i in range(self.opt.modalities_no):
            self.loss_names.extend([f'G_GAN_{i+1}', f'G_L1_{i+1}', f'D_real_{i+1}', f'D_fake_{i+1}', f'G_KLDiv_{i+1}', f'G_KLDiv_{self.mod_id_seg}{i+1}'])
            self.visual_names.extend([f'fake_B_{i+1}', f'fake_B_{i+1}_teacher', f'real_B_{i+1}'])
        self.loss_names.extend([f'G_GAN_{self.mod_id_seg}',f'G_L1_{self.mod_id_seg}',f'D_real_{self.mod_id_seg}',f'D_fake_{self.mod_id_seg}',
                                f'G_KLDiv_{self.mod_id_seg}',f'G_KLDiv_{self.mod_id_seg}{self.opt.modalities_no}'])
        print('self.loss_names',self.loss_names)
        
        for i in range(self.opt.modalities_no+1):
            self.visual_names.extend([f'fake_B_{self.mod_id_seg}{i}', f'fake_B_{self.mod_id_seg}{i}_teacher']) # 0 is used for the base input mod
        self.visual_names.extend([f'fake_B_{self.mod_id_seg}', f'fake_B_{self.mod_id_seg}_teacher', f'real_B_{self.mod_id_seg}',])
        print('self.visual_names',self.visual_names)

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names_g = []
        self.model_names_gs = []
        if self.is_train:
            self.model_names = []
            for i in range(1, self.opt.modalities_no + 1):
                self.model_names.extend([f'G{i}', f'D{i}'])
                self.model_names_g.append(f'G{i}')

            for i in range(self.opt.modalities_no + 1):  # 0 is used for the base input mod
                if self.input_id == '0':
                    self.model_names.extend([f'G{self.mod_id_seg}{i}', f'D{self.mod_id_seg}{i}'])
                    self.model_names_gs.append(f'G{self.mod_id_seg}{i}')
                else:
                    self.model_names.extend([f'G{self.mod_id_seg}{i+1}', f'D{self.mod_id_seg}{i+1}'])
                    self.model_names_gs.append(f'G{self.mod_id_seg}{i+1}')
        else:  # during test time, only load G
            self.model_names = []
            for i in range(1, self.opt.modalities_no + 1):
                self.model_names.extend([f'G{i}'])
                self.model_names_g.append(f'G{i}')

            #input_id = get_input_id(os.path.join(opt.checkpoints_dir, opt.name))
            if self.input_id == '0':
                for i in range(self.opt.modalities_no + 1):  # 0 is used for the base input mod
                    self.model_names.extend([f'G{self.mod_id_seg}{i}'])
                    self.model_names_gs.append(f'G{self.mod_id_seg}{i}')
            else:
                for i in range(self.opt.modalities_no + 1):  # old setting, 1 is used for the base input mod
                    self.model_names.extend([f'G{self.mod_id_seg}{i+1}'])
                    self.model_names_gs.append(f'G{self.mod_id_seg}{i+1}')
        print('self.model_names',self.model_names)
        
        # define networks (both generator and discriminator)
        if isinstance(opt.netG, str):
            opt.netG = [opt.netG] * self.opt.modalities_no
        if isinstance(opt.net_gs, str):
            opt.net_gs = [opt.net_gs] * (self.opt.modalities_no + 1) # +1 for base input mod

            
        for i,model_name in enumerate(self.model_names_g):
            setattr(self,f'net{model_name}',networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG[i], opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.padding))

        # DeepLIIF model currently uses one gs arch because there is only one explicit seg mod output
        for i,model_name in enumerate(self.model_names_gs):
            setattr(self,f'net{model_name}',networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.net_gs[i], opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids))

        if self.is_train:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.model_names_d = [f'D{i+1}' for i in range(self.opt.modalities_no)]
            if self.input_id == '0':
                self.model_names_ds = [f'D{self.mod_id_seg}{i}' for i in range(self.opt.modalities_no+1)]
            else:
                self.model_names_ds = [f'D{self.mod_id_seg}{i+1}' for i in range(self.opt.modalities_no+1)]
            for model_name in self.model_names_d:
                setattr(self,f'net{model_name}',networks.define_D(opt.input_nc+opt.output_nc , opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids))

            for model_name in self.model_names_ds:
                setattr(self,f'net{model_name}',networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids))

            # load the teacher model
            self.opt_teacher = get_opt(opt.model_dir_teacher, mode='test')
            self.opt_teacher.gpu_ids = opt.gpu_ids # use student's gpu_ids
            self.nets_teacher = init_nets(opt.model_dir_teacher, eager_mode=True, opt=self.opt_teacher, phase='test')
            
            # TODO: modify model names to be consistent with the current deepliifkd model names
            # otherwise it may be tricky to pair the loss terms?
            self.opt_teacher.mod_id_seg, self.opt_teacher.input_id = init_input_and_mod_id(self.opt_teacher)
            
            

        if self.is_train:
            # define loss functions
            self.criterionGAN_BCE = networks.GANLoss('vanilla').to(self.device)
            self.criterionGAN_lsgan = networks.GANLoss('lsgan').to(self.device)
            self.criterionSmoothL1 = torch.nn.SmoothL1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            #params = list(self.netG1.parameters()) + list(self.netG2.parameters()) + list(self.netG3.parameters()) + list(self.netG4.parameters()) + list(self.netG51.parameters()) + list(self.netG52.parameters()) + list(self.netG53.parameters()) + list(self.netG54.parameters()) + list(self.netG55.parameters())
            params = []
            for model_name in self.model_names_g + self.model_names_gs:
                params += list(getattr(self,f'net{model_name}').parameters())

            try:
                self.optimizer_G = get_optimizer(opt.optimizer)(params, lr=opt.lr_g, betas=(opt.beta1, 0.999))
            except:
                print(f'betas are not used for optimizer torch.optim.{opt.optimizer} in generators')
                self.optimizer_G = get_optimizer(opt.optimizer)(params, lr=opt.lr_g)

            #params = list(self.netD1.parameters()) + list(self.netD2.parameters()) + list(self.netD3.parameters()) + list(self.netD4.parameters()) + list(self.netD51.parameters()) + list(self.netD52.parameters()) + list(self.netD53.parameters()) + list(self.netD54.parameters()) + list(self.netD55.parameters())
            params = []
            for model_name in self.model_names_d + self.model_names_ds:
                params += list(getattr(self,f'net{model_name}').parameters())

            try:
                self.optimizer_D = get_optimizer(opt.optimizer)(params, lr=opt.lr_d, betas=(opt.beta1, 0.999))
            except:
                print(f'betas are not used for optimizer torch.optim.{opt.optimizer} in discriminators')
                self.optimizer_D = get_optimizer(opt.optimizer)(params, lr=opt.lr_d)

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            print('self.device',self.device)
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
        
        for i,model_name in enumerate(self.model_names_gs):
            if i == 0:
                setattr(self,f'fake_B_{self.mod_id_seg}_{i}',getattr(self,f'net{model_name}')(self.real_A))
            else:
                setattr(self,f'fake_B_{self.mod_id_seg}_{i}',getattr(self,f'net{model_name}')(getattr(self,f'fake_B_{i}')))
            
        setattr(self,f'fake_B_{self.mod_id_seg}',torch.stack([torch.mul(getattr(self,f'fake_B_{self.mod_id_seg}_{i}'), self.seg_weights[i]) for i in range(self.opt.modalities_no+1)]).sum(dim=0))

        fakes_teacher = run_dask(img=self.real_A, nets=self.nets_teacher, opt=self.opt_teacher, use_dask=False, output_tensor=True)
        #print(f'Checking seg mod id for teacher model: current id is {self.opt_teacher.mod_id_seg}, id to map to is {self.mod_id_seg}')
        for k,v in fakes_teacher.items():
            suffix = k[1:] # starts with G
            l_suffix = list(suffix)
            if l_suffix[0] == str(self.opt_teacher.mod_id_seg): # mod_id_seg might be integer
                if l_suffix[0] != str(self.mod_id_seg):
                    l_suffix[0] = str(self.mod_id_seg)
            #suffix = '_'.join(list(suffix)) # 51 -> 5_1
            suffix = '_'.join(l_suffix) # 51 -> 5_1
            #print(f'Loaded teacher model: fake_B_{suffix}_teacher')
            setattr(self,f'fake_B_{suffix}_teacher',v.to(self.device))

    def backward_D(self):
        """Calculate GAN loss for the discriminators"""
        
        for i,model_name in enumerate(self.model_names_d):
            fake_AB = torch.cat((self.real_A, getattr(self,f'fake_B_{i+1}')), 1)
            pred_fake = getattr(self,f'net{model_name}')(fake_AB.detach())
            setattr(self,f'loss_D_fake_{i+1}',self.criterionGAN_BCE(pred_fake, False))
        
        l_pred_fake_seg = []
        for i,model_name in enumerate(self.model_names_ds):
            if i == 0:
                fake_AB_seg_i = torch.cat((self.real_A, getattr(self,f'fake_B_{self.mod_id_seg}')), 1)
            else:
                fake_AB_seg_i = torch.cat((getattr(self,f'real_B_{i}'), getattr(self,f'fake_B_{self.mod_id_seg}')), 1)
            
            pred_fake_seg_i = getattr(self,f'net{model_name}')(fake_AB_seg_i.detach())
            l_pred_fake_seg.append(torch.mul(pred_fake_seg_i, self.seg_weights[i]))
        pred_fake_seg = torch.stack(l_pred_fake_seg).sum(dim=0)
        
        setattr(self,f'loss_D_fake_{self.mod_id_seg}',self.criterionGAN_lsgan(pred_fake_seg, False))
        
        for i,model_name in enumerate(self.model_names_d):
            real_AB = torch.cat((self.real_A, getattr(self,f'real_B_{i+1}')), 1)
            pred_real = getattr(self,f'net{model_name}')(real_AB)
            setattr(self,f'loss_D_real_{i+1}',self.criterionGAN_BCE(pred_real, True))
        
        l_pred_real_seg = []
        for i,model_name in enumerate(self.model_names_ds):
            if i == 0:
                real_AB_seg_i = torch.cat((self.real_A, getattr(self,f'real_B_{self.mod_id_seg}')), 1)
            else:
                real_AB_seg_i = torch.cat((getattr(self,f'real_B_{i}'), getattr(self,f'real_B_{self.mod_id_seg}')), 1)
            
            pred_real_seg_i = getattr(self,f'net{model_name}')(real_AB_seg_i)
            l_pred_real_seg.append(torch.mul(pred_real_seg_i, self.seg_weights[i]))
        pred_real_seg = torch.stack(l_pred_real_seg).sum(dim=0)
        
        #self.loss_D_real_5 = self.criterionGAN_lsgan(pred_real_5, True)
        setattr(self,f'loss_D_real_{self.mod_id_seg}',self.criterionGAN_lsgan(pred_real_seg, True))

        # combine losses and calculate gradients
        self.loss_D = torch.tensor(0., device=self.device)
        for i in range(self.opt.modalities_no):
            self.loss_D += (getattr(self,f'loss_D_fake_{i+1}') + getattr(self,f'loss_D_real_{i+1}')) * 0.5 * self.loss_D_weights[i]
        self.loss_D += (getattr(self,f'loss_D_fake_{self.mod_id_seg}') + getattr(self,f'loss_D_real_{self.mod_id_seg}')) * 0.5 * self.loss_D_weights[self.opt.modalities_no]

        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        
        for i,model_name in enumerate(self.model_names_d):
            fake_AB = torch.cat((self.real_A, getattr(self,f'fake_B_{i+1}')), 1)
            pred_fake = getattr(self,f'net{model_name}')(fake_AB)
            setattr(self,f'loss_G_GAN_{i+1}',self.criterionGAN_BCE(pred_fake, True))

        l_pred_fake_seg = []
        for i,model_name in enumerate(self.model_names_ds):
            if i == 0:
                fake_AB_seg_i = torch.cat((self.real_A, getattr(self,f'fake_B_{self.mod_id_seg}')), 1)
            else:
                fake_AB_seg_i = torch.cat((getattr(self,f'real_B_{i}'), getattr(self,f'fake_B_{self.mod_id_seg}')), 1)
            
            pred_fake_seg_i = getattr(self,f'net{model_name}')(fake_AB_seg_i)
            l_pred_fake_seg.append(torch.mul(pred_fake_seg_i, self.seg_weights[i]))
        pred_fake_seg = torch.stack(l_pred_fake_seg).sum(dim=0)
        
        setattr(self,f'loss_G_GAN_{self.mod_id_seg}',self.criterionGAN_lsgan(pred_fake_seg, True))

        # Second, G(A) = B
        for i in range(self.opt.modalities_no):
            setattr(self,f'loss_G_L1_{i+1}',self.criterionSmoothL1(getattr(self,f'fake_B_{i+1}'), getattr(self,f'real_B_{i+1}')) * self.opt.lambda_L1)
        setattr(self,f'loss_G_L1_{self.mod_id_seg}',self.criterionSmoothL1(getattr(self,f'fake_B_{self.mod_id_seg}'), getattr(self,f'real_B_{self.mod_id_seg}')) * self.opt.lambda_L1)
        
        for i in range(self.opt.modalities_no):
            setattr(self,f'loss_G_VGG_{i+1}',self.criterionVGG(getattr(self,f'fake_B_{i+1}'), getattr(self,f'real_B_{i+1}')) * self.opt.lambda_feat)
        setattr(self,f'loss_G_VGG_{self.mod_id_seg}',self.criterionVGG(getattr(self,f'fake_B_{self.mod_id_seg}'), getattr(self,f'real_B_{self.mod_id_seg}')) * self.opt.lambda_feat)
        
        # .view(1,1,-1) reshapes the input (batch_size, 3, 512, 512) to (batch_size, 1, 3*512*512)
        # softmax/log-softmax is then applied on the concatenated vector of size (1, 3*512*512)
        # this normalizes the pixel values across all 3 RGB channels
        # the resulting vectors are then used to compute KL divergence loss
        # self.loss_G_KLDiv_1 = self.criterionKLDiv(self.logsoftmax(self.fake_B_1.view(1,1,-1)), self.softmax(self.fake_B_1_teacher.view(1,1,-1)))
        # self.loss_G_KLDiv_2 = self.criterionKLDiv(self.logsoftmax(self.fake_B_2.view(1,1,-1)), self.softmax(self.fake_B_2_teacher.view(1,1,-1)))
        # self.loss_G_KLDiv_3 = self.criterionKLDiv(self.logsoftmax(self.fake_B_3.view(1,1,-1)), self.softmax(self.fake_B_3_teacher.view(1,1,-1)))
        # self.loss_G_KLDiv_4 = self.criterionKLDiv(self.logsoftmax(self.fake_B_4.view(1,1,-1)), self.softmax(self.fake_B_4_teacher.view(1,1,-1)))
        # self.loss_G_KLDiv_5 = self.criterionKLDiv(self.logsoftmax(self.fake_B_5.view(1,1,-1)), self.softmax(self.fake_B_5_teacher.view(1,1,-1)))
        
        for i in range(self.opt.modalities_no):
            setattr(self,f'loss_G_KLDiv_{i+1}',self.criterionKLDiv(self.logsoftmax(getattr(self,f'fake_B_{i+1}').view(1,1,-1)), self.softmax(getattr(self,f'fake_B_{i+1}_teacher').view(1,1,-1))))
        setattr(self,f'loss_G_KLDiv_{self.mod_id_seg}',self.criterionKLDiv(self.logsoftmax(getattr(self,f'fake_B_{self.mod_id_seg}').view(1,1,-1)), self.softmax(getattr(self,f'fake_B_{self.mod_id_seg}_teacher').view(1,1,-1))))
        
        # self.loss_G_KLDiv_5_1 = self.criterionKLDiv(self.logsoftmax(self.fake_B_5_1.view(1,1,-1)), self.softmax(self.fake_B_5_1_teacher.view(1,1,-1)))
        # self.loss_G_KLDiv_5_2 = self.criterionKLDiv(self.logsoftmax(self.fake_B_5_2.view(1,1,-1)), self.softmax(self.fake_B_5_2_teacher.view(1,1,-1)))
        # self.loss_G_KLDiv_5_3 = self.criterionKLDiv(self.logsoftmax(self.fake_B_5_3.view(1,1,-1)), self.softmax(self.fake_B_5_3_teacher.view(1,1,-1)))
        # self.loss_G_KLDiv_5_4 = self.criterionKLDiv(self.logsoftmax(self.fake_B_5_4.view(1,1,-1)), self.softmax(self.fake_B_5_4_teacher.view(1,1,-1)))
        # self.loss_G_KLDiv_5_5 = self.criterionKLDiv(self.logsoftmax(self.fake_B_5_5.view(1,1,-1)), self.softmax(self.fake_B_5_5_teacher.view(1,1,-1)))
        
        for i in range(self.opt.modalities_no+1):
            setattr(self,f'loss_G_KLDiv_{self.mod_id_seg}{i}',self.criterionKLDiv(self.logsoftmax(getattr(self,f'fake_B_{self.mod_id_seg}_{i}').view(1,1,-1)), self.softmax(getattr(self,f'fake_B_{self.mod_id_seg}_{i}_teacher').view(1,1,-1))))
        #setattr(self,f'loss_G_KLDiv_{self.mod_id_seg}{self.opt.modalities_no+1}',self.criterionKLDiv(self.logsoftmax(getattr(self,f'fake_B_{self.mod_id_seg}_{self.opt.modalities_no+1}').view(1,1,-1)), self.softmax(getattr(self,f'fake_B_{self.mod_id_seg}_teacher').view(1,1,-1))))


        # self.loss_G = (self.loss_G_GAN_1 + self.loss_G_L1_1 + self.loss_G_VGG_1) * self.loss_G_weights[0] + \
        #               (self.loss_G_GAN_2 + self.loss_G_L1_2 + self.loss_G_VGG_2) * self.loss_G_weights[1] + \
        #               (self.loss_G_GAN_3 + self.loss_G_L1_3 + self.loss_G_VGG_3) * self.loss_G_weights[2] + \
        #               (self.loss_G_GAN_4 + self.loss_G_L1_4 + self.loss_G_VGG_4) * self.loss_G_weights[3] + \
        #               (self.loss_G_GAN_5 + self.loss_G_L1_5) * self.loss_G_weights[4] + \
        #               (self.loss_G_KLDiv_1 + self.loss_G_KLDiv_2 + self.loss_G_KLDiv_3 + self.loss_G_KLDiv_4 + \
        #               self.loss_G_KLDiv_5 + self.loss_G_KLDiv_5_1 + self.loss_G_KLDiv_5_2 + self.loss_G_KLDiv_5_3 + \
        #               self.loss_G_KLDiv_5_4 + self.loss_G_KLDiv_5_5) * 10
        
        self.loss_G = torch.tensor(0., device=self.device)
        for i in range(self.opt.modalities_no):
            self.loss_G += (getattr(self,f'loss_G_GAN_{i+1}') + getattr(self,f'loss_G_L1_{i+1}') + getattr(self,f'loss_G_VGG_{i+1}')) * self.loss_G_weights[i]
        self.loss_G += (getattr(self,f'loss_G_GAN_{self.mod_id_seg}') + getattr(self,f'loss_G_L1_{self.mod_id_seg}')) * self.loss_G_weights[i]
        
        factor_KLDiv = 10
        for i in range(self.opt.modalities_no):
            self.loss_G += (getattr(self,f'loss_G_KLDiv_{i+1}') + getattr(self,f'loss_G_KLDiv_{self.mod_id_seg}{i+1}')) * factor_KLDiv
        self.loss_G += getattr(self,f'loss_G_KLDiv_{self.mod_id_seg}') * factor_KLDiv
        if self.input_id == '0':
            self.loss_G += getattr(self,f'loss_G_KLDiv_{self.mod_id_seg}0') * factor_KLDiv
        else:
            self.loss_G += getattr(self,f'loss_G_KLDiv_{self.mod_id_seg}{self.opt.modalities_no+1}') * factor_KLDiv


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
        for model_name in self.model_names_d + self.model_names_ds:
            self.set_requires_grad(getattr(self,f'net{model_name}'), True)

        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights

        # update G
        for model_name in self.model_names_d + self.model_names_ds:
            self.set_requires_grad(getattr(self,f'net{model_name}'), False)

        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
    
    def calculate_losses(self):
        """
        Calculate losses but do not optimize parameters. Used in validation loss calculation during training.
        """
        
        self.forward()                   # compute fake images: G(A)
        # update D
        for model_name in self.model_names_d + self.model_names_ds:
            self.set_requires_grad(getattr(self,f'net{model_name}'), True)

        self.optimizer_D.zero_grad()        # set D's gradients to zero
        self.backward_D()                # calculate gradients for D

        # update G
        for model_name in self.model_names_d + self.model_names_ds:
            self.set_requires_grad(getattr(self,f'net{model_name}'), False)

        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
            

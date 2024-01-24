"""This package options includes option modules: training options, test options, and basic options (used in both training and test)."""

from pathlib import Path
import os
from ..util.util import mkdirs

def read_model_params(file_addr):
    with open(file_addr) as f:
        lines = f.readlines()
    param_dict = {}
    for line in lines:
        if ':' in line:
            key = line.split(':')[0].strip()
            val = line.split(':')[1].split('[')[0].strip()
            param_dict[key] = val
    return param_dict

class Options:
    def __init__(self, d_params=None, path_file=None, mode='train'):
        assert d_params is not None or path_file is not None, "either d_params or path_file should be provided"
        assert d_params is None or path_file is None, "only one source can be provided, either being d_params or path_file"
        assert mode in ['train','test'], 'mode should be one of ["train", "test"]'
        
        if path_file:
            d_params = read_model_params(path_file)
     
        for k,v in d_params.items():
            try:
                if k not in ['phase']: # e.g., k = 'phase', v = 'train', eval(v) is a function rather than a string
                    setattr(self,k,eval(v)) # to parse int/float/tuple etc. from string
                else:
                    setattr(self,k,v)
            except:
                setattr(self,k,v)            

        if mode == 'train':
            self.is_train = True
            self.netG = 'resnet_9blocks'
            self.netD = 'n_layers'
            self.n_layers_D = 4
            self.lambda_L1 = 100
            self.lambda_feat = 100
        else:
            self.phase = 'test'
            self.is_train = False
            self.input_nc = 3
            self.output_nc = 3
            self.ngf = 64
            self.norm = 'batch'
            self.use_dropout = True
            #self.padding_type = 'zero' # some models use reflect etc. which adds additional randomness 
            #self.padding = 'zero'
            self.use_dropout = False #if self.no_dropout == 'True' else True
            
            # reset checkpoints_dir and name based on the model directory
            # when base model is initialized: self.save_dir = os.path.join(opt.checkpoints_dir, opt.name) 
            model_dir = Path(path_file).parent
            self.checkpoints_dir = str(model_dir.parent)
            self.name = str(model_dir.name)
            
            #self.gpu_ids = [] # gpu_ids is only used by eager mode, set to empty / cpu to be the same as the old settings; non-eager mode will use all gpus
            
            # to account for old settings where gpu_ids value is an integer, not a tuple
            if isinstance(self.gpu_ids,int):
                self.gpu_ids = (self.gpu_ids,)
            
            # to account for old settings before modalities_no was introduced
            if not hasattr(self,'modalities_no') and hasattr(self,'targets_no'):
                self.modalities_no = self.targets_no - 1
                del self.targets_no
            
            # to account for old settings: same as in cli.py train
            if not hasattr(self,'seg_no'):
              if self.model == 'DeepLIIF':
                  self.seg_no = 1
              elif self.model == 'DeepLIIFExt':
                  if self.seg_gen:
                      self.seg_no = self.modalities_no
                  else:
                      self.seg_no = 0
              elif self.model == 'SDG':
                  self.seg_no = 0
                  self.seg_gen = False
              else:
                  raise Exception(f'seg_gen cannot be automatically determined for {opt.model}')
            
            # to account for old settings: prior to SDG, our models only have 1 input image
            if not hasattr(self,'input_no'):
                self.input_no = 1
            
            # to account for old settings: before adding scale_size
            if not hasattr(self, 'scale_size'):
                if self.model in ['DeepLIIF','SDG']:
                    self.scale_size = 512
                elif self.model == 'DeepLIIFExt':
                    self.scale_size = 1024
                else:
                  raise Exception(f'scale_size cannot be automatically determined for {opt.model}')

            
    
    def _get_kwargs(self):
        common_attr = ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__']
        l_args = [x for x in dir(self) if x not in common_attr]
        return {k:getattr(self,k) for k in l_args}

def format_options(opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    return message
    
def print_options(opt, save=False):
    """Print (and save) options

    It will print both current options and default values(if different).
    If save=True, it will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = format_options(opt)
    print(message)
    
    # save to the disk
    if save:
        save_options(opt)

def save_options(opt):
    message = format_options(opt)
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message+'\n')

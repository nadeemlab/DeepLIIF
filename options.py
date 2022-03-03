class Options:
    def __init__(self, dataroot, name, gpu_ids, checkpoints_dir, targets_no, input_nc, output_nc, ngf, ndf, net_d,
                 net_g, n_layers_d, norm, init_type, init_gain, no_dropout, direction, serial_batches, num_threads,
                 batch_size, load_size, crop_size, max_dataset_size, preprocess, no_flip, display_winsize, epoch,
                 load_iter, verbose, lambda_l1, is_train, display_freq, display_ncols, display_id, display_server, display_env,
                 display_port, update_html_freq, print_freq, no_html, save_latest_freq, save_epoch_freq, save_by_iter,
                 continue_train, epoch_count, phase, lr_policy, n_epochs, n_epochs_decay, beta1, lr, lr_decay_iters,
                 remote_transfer_cmd, dataset_mode, padding):
        self.dataroot = dataroot
        self.name = name
        self.gpu_ids = gpu_ids
        self.checkpoints_dir = checkpoints_dir
        self.targets_no = targets_no
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.ndf = ndf
        self.net_d = net_d
        self.net_g = net_g
        self.n_layers_d = n_layers_d
        self.norm = norm
        self.init_type = init_type
        self.init_gain = init_gain
        self.no_dropout = no_dropout
        self.direction = direction
        self.serial_batches = serial_batches
        self.num_threads = num_threads
        self.batch_size = batch_size
        self.load_size = load_size
        self.crop_size = crop_size
        self.max_dataset_size = max_dataset_size
        self.preprocess = preprocess
        self.no_flip = no_flip
        self.display_winsize = display_winsize
        self.epoch = epoch
        self.load_iter = load_iter
        self.verbose = verbose
        self.lambda_l1 = lambda_l1
        self.is_train = is_train
        self.display_freq = display_freq
        self.display_ncols = display_ncols
        self.display_id = display_id
        self.display_server = display_server
        self.display_env = display_env
        self.display_port = display_port
        self.update_html_freq = update_html_freq
        self.print_freq = print_freq
        self.no_html = no_html
        self.save_latest_freq = save_latest_freq
        self.save_epoch_freq = save_epoch_freq
        self.save_by_iter = save_by_iter
        self.continue_train = continue_train
        self.epoch_count = epoch_count
        self.phase = phase
        self.lr_policy = lr_policy
        self.n_epochs = n_epochs
        self.n_epochs_decay = n_epochs_decay
        self.beta1 = beta1
        self.lr = lr
        self.lr_decay_iters = lr_decay_iters
        self.dataset_mode = dataset_mode
        self.padding = padding
        self.remote_transfer_cmd = remote_transfer_cmd

        self.isTrain = True
        self.netG = 'resnet_9blocks'
        self.netD = 'n_layers'
        self.n_layers_D = 4
        self.lambda_L1 = 100
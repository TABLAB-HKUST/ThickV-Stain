import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

from torch.autograd import Variable
import torch.nn.functional as F

class MultiScaleModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D', 'G', 'idt',]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if self.isTrain:
            self.visual_names = ['input_A',  'fake_B',]  #'input_B', 'real_B','B_global',
        else:
            self.visual_names = ['fake_B']  #,'input_A','fake_1','fake_2','fake_3'

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G']  #, 'M', 'F'
        else:  # during test time, only load Gs
            self.model_names = ['G']  #, 'M', 'F'

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        # print('********************', opt.norm_G)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm_G,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt)
        
        if self.isTrain:  # define discriminators
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm_D, opt.init_type, opt.init_gain, self.gpu_ids, opt)


        if self.isTrain:
            if opt.lambda_idt > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters()), lr=opt.G_lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters()), lr=opt.D_lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.input_A = input['A' if AtoB else 'B'].to(self.device)
        self.input_B = input['B' if AtoB else 'A'].to(self.device)
        
      
        if not self.isTrain:
            self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.label_A = input['A_class'].to(self.device)  #.type(torch.LongTensor)
        self.label_B = input['B_class'].to(self.device)
        # # self.fake_label = torch.zeros(self.opt.batch_size, 1, dtype=torch.float32)   #torch.Size([8]) torch.Size([8, 1])
        self.fake_label = Variable(torch.zeros_like(self.label_A.data).cuda(), requires_grad=False) #changed on 0415 10:23
        
        if self.opt.class_block_use and self.isTrain:
            if self.opt.netD == 'm_scale':
                label_A = input['A_class'].to(self.device)  #.type(torch.LongTensor)
                label_B = input['B_class'].to(self.device)  # tensor([1, 7, 3, 2, 5, 0, 2, 6]  torch.Size([8])
                self.fake_label = torch.zeros(self.opt.batch_size,dtype=torch.int64).to(self.device)
                self.nll_weight = torch.tensor([3.,1.,1.]).to(self.device)  # for lung
                ndim = (self.opt.class_nums + 1) * 1 
                # convert label to one-hot vector
                self.label_A = torch.zeros(self.opt.batch_size, ndim, dtype=torch.float32)   # 3 is class nums
                self.label_B = torch.zeros(self.opt.batch_size, ndim, dtype=torch.float32)   
                self.fake_label = torch.zeros(self.opt.batch_size, ndim, dtype=torch.float32)           

                for i in range(self.opt.batch_size):
                    self.label_A[i,label_A[i]+1] = 1
                    self.label_B[i,label_B[i]+1] = 1
                    self.fake_label[i,0] = 1      

                self.label_A = self.label_A.unsqueeze(2).unsqueeze(3).cuda()
                self.label_B = self.label_B.unsqueeze(2).unsqueeze(3).cuda()
                self.fake_label = self.fake_label.unsqueeze(2).unsqueeze(3).cuda()
                # print('self.label_A------------',self.label_A.size() )
            else:
                self.label_A = input['A_class'].to(self.device)  #.type(torch.LongTensor)
                self.label_B = input['B_class'].to(self.device)


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        self.real_B = self.input_B
        self.real_A = self.input_A
        self.fake_B = self.netG(self.input_A)
          


    def criterionS(self, input, target):
        if self.opt.netD == 'm_scale':
            logsoft = torch.nn.LogSoftmax(dim=1)  
            s = logsoft(input)    # after softmax torch.Size([8, num_cls, 15, 15])
            gamma = 0 
            # return torch.mean(torch.sum(-target *((1.01-torch.exp(s))**gamma)* s, dim=1))
            return torch.mean(torch.sum(-target * s, dim=1))  # target * s torch.Size([8, num_cls, 15, 15])      
        else:
            m = torch.nn.LogSoftmax()
            nll_loss = torch.nn.NLLLoss() # 178 751 447 
            loss = nll_loss(m(input), target)
            return loss

    def cal_mscale_real(self, netD, real, real_label):
        outs0 = netD(real)
        loss_D_real = 0
        for out0 in outs0: 
            if self.opt.class_block_use:
                loss_D_real += self.criterionS(out0, real_label)
            else:
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss_D_real += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1)) 
        return loss_D_real

    def cal_mscale_fake(self, netD, fake, fake_label):
        outs0 = netD(fake)
        loss_D_fake = 0
        for out0 in outs0: 
            if self.opt.class_block_use:
                loss_D_fake += self.criterionS(out0, fake_label)
            else:
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                loss_D_fake += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0)) 
        return loss_D_fake



    def backward_D(self):
        """Calculate GAN loss for discriminator D_A"""

        # fake_B = self.fake_B_pool.query(self.fake_B)
        fake_B = self.fake_B
        if self.opt.netD == 'm_scale':
            loss_D_real = self.cal_mscale_real(self.netD, self.real_B, self.label_B)
            loss_D_fake = self.cal_mscale_fake(self.netD, fake_B.detach(), self.fake_label)
            
        if self.opt.netD == 'basic':
            pred_real = self.netD(self.real_B)
            pred_fake = self.netD(fake_B.detach())
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)
        self.loss_D = (loss_D_real + loss_D_fake) * 0.5
            
        self.loss_D.backward()



    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_idt
        # Identity loss
        if lambda_idt > 0:
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            if self.opt.netG == 'm_scale':
                self.idt_B = self.netG(self.real_B.detach())
            else:
                self.idt_B = self.netG(self.real_B.detach())
            
            self.loss_idt = self.criterionIdt(self.idt_B, self.real_B.detach()) * lambda_idt
        else:
            self.loss_idt = 0
        
        # GAN loss D_A(G_A(A)) and D_B(G_B(B)) 
        if self.opt.netD == 'm_scale':
            self.loss_G = self.cal_mscale_real(self.netD, self.fake_B, self.label_A)
        if self.opt.netD == 'basic':
            self.loss_G = self.criterionGAN(self.netD(self.fake_B), True)
        

        # combined loss and calculate gradients
        self.loss_G = self.loss_G +  self.loss_idt 
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
          
        # D_A and D_B
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D()      # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights

       

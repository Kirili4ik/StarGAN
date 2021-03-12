import torch
from torch import nn
from my_utils import compute_gradient_penalty, generate_target_label


def conv_norm_act(in_size, out_size, kernel_size=3, stride=1, padding=None, bias=False,
                  is_instance=True,
                  act=None):
    
    padding=(kernel_size - 1) // 2 if padding is None else padding
    
    return nn.Sequential(
        nn.Conv2d(in_size, out_size, kernel_size, 
                  stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_size, affine=True, track_running_stats=True) if is_instance else nn.Identity(),
        act if act is not None else nn.Identity()
    )


def deconv_norm_act(in_size, out_size, kernel_size, stride, padding=None, act=nn.ReLU()):
    
    padding=(kernel_size - 1) // 2 if padding is None else padding
    
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(in_size, out_size, kernel_size, 
                  stride=1, padding=padding, bias=False),
        # nn.ConvTranspose2d(in_size, out_size, kernel_size,
        #                    stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_size, affine=True, track_running_stats=True),
        act
    )



class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        n_ch = self.config.n_channels
        relu = nn.ReLU()
        
        self.encoder = nn.Sequential(
            conv_norm_act(3 + config.n_domains, n_ch,   kernel_size=7, act=relu),
            conv_norm_act(n_ch,                 n_ch*2, kernel_size=3, stride=2, act=relu),
            conv_norm_act(n_ch*2,               n_ch*4, kernel_size=3, stride=2, act=relu)
        )
        self.neck = nn.ModuleList([]) 
        for i in range(6):
            self.neck.append(conv_norm_act(n_ch*4, n_ch*4, act=relu))
            self.neck.append(conv_norm_act(n_ch*4, n_ch*4, is_instance=False))
            
        self.decoder = nn.Sequential(
            deconv_norm_act(n_ch*4, n_ch*2, kernel_size=3, stride=2, act=relu),
            deconv_norm_act(n_ch*2, n_ch,   kernel_size=3, stride=2, act=relu),
            conv_norm_act(  n_ch,   3,      kernel_size=7, stride=1, act=nn.Tanh())
        )
        
        
    def forward(self, x, labels):
        labels = labels.unsqueeze(2).unsqueeze(3)
        labels = labels.repeat(1, 1, x.size(2), x.size(3))
        concat = torch.cat((x, labels), dim=1)     # Do we want to autograd through this?
        encoded = self.encoder(concat)
        for layer in self.neck:
            resid_res = layer(encoded)
            encoded = (encoded + resid_res)        # Do we want to divide it by sqrt(2)?
        decoded = self.decoder(encoded)
        return decoded



class Critic(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        n_ch = self.config.n_channels
        lrelu = nn.LeakyReLU(0.01)
        
        self.input_layer = conv_norm_act(3, n_ch, kernel_size=4, stride=2, bias=True,
                                         act=lrelu, is_instance=False)
        self.hidden = nn.Sequential(
            conv_norm_act(n_ch,    n_ch*2,  kernel_size=4, stride=2, bias=True, act=lrelu, is_instance=False),
            conv_norm_act(n_ch*2,  n_ch*4,  kernel_size=4, stride=2, bias=True, act=lrelu, is_instance=False),
            conv_norm_act(n_ch*4,  n_ch*8,  kernel_size=4, stride=2, bias=True, act=lrelu, is_instance=False),
            conv_norm_act(n_ch*8,  n_ch*16, kernel_size=4, stride=2, bias=True, act=lrelu, is_instance=False),
            conv_norm_act(n_ch*16, n_ch*32, kernel_size=4, stride=2, bias=True, act=lrelu, is_instance=False)
        )
        self.output_src = conv_norm_act(n_ch*32, 1, kernel_size=3, is_instance=False)
        self.output_cls = conv_norm_act(n_ch*32, self.config.n_domains,
                                        kernel_size=self.config.image_size // 64, is_instance=False)
        
        
    def forward(self, x):
        input_res = self.input_layer(x)
        hidden_res = self.hidden(input_res)
        D_src = self.output_src(hidden_res)
        D_cls = self.output_cls(hidden_res)
        
        return D_src, D_cls



class StarGAN:
    
    def __init__(self, config):
        self.config = config 
        self.G = Generator(config)
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=config.lr_start, betas=config.betas)
        self.D = Critic(config)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=config.lr_start, betas=config.betas)
        
        checkpoint = torch.load('checkpoint_1')
        self.G.load_state_dict(checkpoint['G_st_dict'])
        self.D.load_state_dict(checkpoint['D_st_dict'])
        self.lambda_gp = config.lambda_gp
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec

    
    def train(self):
        self.G.train()
        self.D.train()
     
    
    def eval(self):
        self.G.eval()
        self.D.eval()

        
    def to(self, device):
        self.D.to(device)
        self.G.to(device)
        
    
    def reset_grad(self):
        self.G_optimizer.zero_grad()
        self.D_optimizer.zero_grad()

        
    def trainG(self, real_x, real_label, target_label):
        '''
            Need: D_fake_cls, real_label, target_label
        '''
        
        # disable backprop for D for saving Time and Resourses
        #for param in self.D.parameters():
        #    param.requires_grad = False
        #self.G_optimizer.zero_grad()
        
        ### APPLY MODELS
        fake_x = self.G(real_x, target_label)
        D_fake_src, D_fake_cls = self.D(fake_x)
        
        
        # Domain Classification loss
        L_fake_cls = F.binary_cross_entropy_with_logits(D_fake_cls.squeeze(), target_label.squeeze())  # c from target domain
        
        # Reconstruction loss
        G_G_x = self.G(fake_x, real_label)
        L_rec = F.l1_loss(G_G_x, real_x)

        # Adversarial loss
        G_src_loss = -torch.mean(D_fake_src)

        # Full loss
        L_G = G_src_loss + self.lambda_cls * L_fake_cls + self.lambda_rec * L_rec
        self.reset_grad()
        L_G.backward()
        self.G_optimizer.step()
        return G_src_loss.item(), self.lambda_cls * L_fake_cls.item(), self.lambda_rec * L_rec.item(), L_G.item()

        
    def trainD(self, real_x, real_label, target_label):
        '''
            Need: D_real_cls, D_real_src, D_fake_src, real_label
        '''
        
        # disable backprop for G for saving Time and Resourses
        #for param in self.G.parameters():
        #    param.requires_grad = False
        
        ### APPLY MODELS
        D_real_src, D_real_cls = self.D(real_x)
        fake_x                 = self.G(real_x, target_label)
        D_fake_src, D_fake_cls = self.D(fake_x.detach())
        
        # Grad Penalty
        gradient_penalty = compute_gradient_penalty(self.D, real_x.data, fake_x.data)
        
        # Domain Classification loss
        L_real_cls = F.binary_cross_entropy_with_logits(D_real_cls.squeeze(), real_label.squeeze())    # c' from true domain
        
        # Adversarial loss
        D_src_loss = -torch.mean(D_real_src) + torch.mean(D_fake_src) \
                                                      + self.lambda_gp * gradient_penalty
        
        # Full loss
        L_D = D_src_loss + self.lambda_cls * L_real_cls
        self.reset_grad()
        L_D.backward()
        self.D_optimizer.step()
        return D_src_loss.item(), self.lambda_cls * L_real_cls.item(), self.lambda_gp * gradient_penalty.item(), L_D.item()


    def generate(self, image, label=None):
        if label is None:
            label = generate_target_label(self.config.batch_size, self.config.n_domains, self.config.device)
        return self.G(image, label)
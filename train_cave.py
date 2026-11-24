
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from scipy.io import loadmat
from thop import profile, clever_format
import os
from calculate_metrics import *
import torch
from loss import *
from einops import rearrange
from multiprocessing import Pool
from torch.utils.data import Dataset, DataLoader
from data_load import kaistDATAprocess

# from torch import nn
from tqdm import tqdm
import time
import pandas as pd
import torch.utils.data as data
from Utils import *
import math
import numpy as np
import hdf5storage as h5
from torch.cuda import is_available as is_cuda_available
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_r():
    r_path = "./r.mat"
    data = sio.loadmat(r_path)
    r = data['r'][...]
    r = np.array(r,dtype=np.float32)
    R = torch.from_numpy(r)
    R = R.unsqueeze(0)
    return R
def mkdir(path):

    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("训练文件夹为：{}".format(path))
    else:
        print('已存在{}'.format(path))

def reconstruction(net2, HSI_LR, MSI,HRHSI, downsample_factor, training_size, stride,val_loss):
    index_matrix = torch.zeros((HSI_LR.shape[1], MSI.shape[2], MSI.shape[3])).cuda()
    abundance_t = torch.zeros((HSI_LR.shape[1], MSI.shape[2], MSI.shape[3])).cuda()
    a = []
    for j in range(0, MSI.shape[2] - training_size + 1, stride):
        a.append(j)
    a.append(MSI.shape[2] - training_size)
    b = []
    for j in range(0, MSI.shape[3] - training_size + 1, stride):
        b.append(j)
    b.append(MSI.shape[3] - training_size)
    for j in a:
        for k in b:
            temp_hrms = MSI[:, :, j:j + training_size, k:k + training_size]
            temp_lrhs = HSI_LR[:, :, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                        int(k / downsample_factor):int((k + training_size) / downsample_factor)]
            temp_hrhs = HRHSI[:, :, j:j + training_size, k:k + training_size]
            LR_up = torch.nn.functional.interpolate(temp_lrhs, scale_factor=8, mode='bilinear') # 
            with torch.no_grad():

                # lr = optimizer.param_groups[0]['lr']
                # r = read_r().cuda()
                # LR_up = torch.nn.functional.interpolate(temp_lrhs, scale_factor=8, mode='bilinear') # MS
                # B, C, H, W = temp_hrms.shape
                # HS1 = rearrange(temp_hrms, 'b c h w -> b (h w) c')
                # HS1 = torch.matmul(HS1, r)
                # HS1 = rearrange(HS1, 'b (h w) c -> b c h w', h=H, w=W)
#                 out = net2(HS1-LR_up, temp_lrhs, temp_hrms)
                out,*_ = net2(temp_lrhs, temp_hrms)
               
                loss_temp = loss_func(out, temp_hrhs.cuda())
                val_loss.update(loss_temp)
                HSI = out.squeeze()
                # 去掉维数为一的维度
                HSI = torch.clamp(HSI, 0, 1)
                abundance_t[:, j:j + training_size, k:k + training_size] = abundance_t[:, j:j + training_size,
                                                                           k:k + training_size] + HSI
                index_matrix[:, j:j + training_size, k:k + training_size] = 1 + index_matrix[:, j:j + training_size,
                                                                                k:k + training_size]

    HSI_recon = abundance_t / index_matrix
    assert torch.isnan(HSI_recon).sum() == 0
    return HSI_recon


if __name__ == '__main__':
    # 路径参数

    root= "./checkpoint_cave/"

    from Model.model import Net
    model_name = 'model'
    model =Net(31,3, 8).cuda()
    
#     from Compare.Otias import otias_c_x8
#     model_name = 'OTIAS'
#     model = otias_c_x8(n_select_bands=3, n_bands=31, feat_dim=128,
#                            guide_dim=128, sz=64).cuda()
    
    
    dataset= 'CAVE'
    data_path = './Data/CAVE/Train/HSI/'
    test_path = './Data/CAVE/Test/HSI/'
    
#     dataset= 'KAIST'
#     root= "./checkpoint_kaist/"
#     data_path = './Data/KAIST/train/'
#     test_path = './Data/KAIST/test/'
    
    path = root + model_name
    print(path)
    mkdir(os.path.join(root,model_name))

    # 训练参数
    loss_func = nn.L1Loss(reduction='mean').cuda()
    R = create_F()
    PSF = fspecial('gaussian', 8, 3)
    downsample_factor = 8
    training_size = 64  # 训练和测试裁剪块的大小

    EPOCH =1500
    weight_decay=1e-8
    BATCH_SIZE = 16

    psnr_optimal = 44
    rmse_optimal = 1.5
    val_epoch = 25                     # 前50epoch不测试
    val_interval = 50        # 每隔val_interval epoch测试一次
    checkpoint_interval = 100
    

    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
    ## New model
    print("===> New Model")
    
    print(model_name)

#     model =Net(31,3, 8).cuda()

    # set the number of parallel GPUs
    print("===> Setting GPU")
#     model = dataparallel(model ,1)

    
#     测试参数量
    a,b=64,64
    hsi = torch.randn(2, 31,a//8,b//8).cuda()
    msi = torch.randn(2, 3, a,b).cuda()
    HR = torch.randn(2, 31, a,b).cuda()
    # 注册 ReLU hook，防止 AttributeError

    flops, params,DIC= profile(model, inputs=(hsi, msi),ret_layer_info=True)
    # flops, params,DIC= profile(model, inputs=(msi,hsi),ret_layer_info=True)
    flops, params= clever_format([flops, params], "%.3f")
    print(flops, params)
    
#     torch.cuda.synchronize()
#     start = time.time()

#     output = model(msi,HR,hsi)

#     torch.cuda.synchronize()
#     print("Forward Time:", time.time() - start)
    
#     model = otias_c_x8(n_select_bands=3, n_bands=31, feat_dim=128,
#                            guide_dim=128, sz=64).cuda()
    #数据集处理
    if dataset=='CAVE' :
        stride = 32
        stride1 = 32
        train_data = kaistDATAprocess(data_path, R, training_size, stride, downsample_factor, PSF, num=20,dataset='cave')
    #数据集处理
    if dataset=='KAIST' :
        stride = 64   # 滑动窗口方式裁剪，stride
        stride1 = 64
        train_data =kaistDATAprocess(data_path, R, training_size, stride, downsample_factor, PSF, num=20,dataset='kaist')
    print('len(dataset):', len(train_data))
    train_loader = data.DataLoader(dataset=train_data, num_workers=8, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)


    
    # 模型初始化
     # Initialize weight
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
        if isinstance(layer, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(layer.weight)

#     optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    LR = 2e-4
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
#     scheduler = MultiStepLR(optimizer, milestones=[500,1000], gamma=0.75)
    initial_epoch = findLastCheckpoint(save_dir=root+model_name)
    initial_epoch = 0
    loss_eval =999
    if initial_epoch > 0:
        print('resuming by loading epoch %04d' % initial_epoch)
        load_ckpt = torch.load(os.path.join(root+model_name, 'model_%04d.pth' % initial_epoch))
        load_weights_dict = {
            k: v
            for k, v in load_ckpt['parameter'].items()
            if k in model.state_dict() and model.state_dict()[k].numel() == v.numel()
        }

# 可选：打印缺失的键（未被加载的键）
        missing_keys = [k for k in model.state_dict() if k not in load_weights_dict]
        print(f"Missing keys: {missing_keys}")
        load_weights_dict = load_ckpt['parameter']

        model.load_state_dict(load_weights_dict,strict =False) 
        optimizer.load_state_dict(load_ckpt['optimizer'])                                     # 加载优化器状态
#         scheduler.load_state_dict(load_ckpt['scheduler'])
        
    
    # 创建excel
    df = pd.DataFrame(columns=['epoch', 'lr', 'train_loss', 'val_psnr', 'val_sam'])  # 列名
    excel_name=dataset+'new_record.csv'
    excel_path=os.path.join(path,excel_name)
    if initial_epoch == 0:
        df.to_csv(excel_path,  index=False)
        step = 0

    
    for epoch in range(initial_epoch+1, EPOCH+1):
        model.train()
        loss_all = []
        loop = tqdm(train_loader, total=len(train_loader))
        for HR, RGB, LR in loop:
            HR = HR.to(device)
            RGB = RGB.to(device)
            LR = LR.to(device)
            lr = optimizer.param_groups[0]['lr']
            r = read_r().to(device)
            LR_up = torch.nn.functional.interpolate(LR, scale_factor=8, mode='bilinear') # 
            RGB_down = torch.nn.functional.interpolate(RGB, scale_factor=1 / 8, mode='bilinear')  # MS
            
            B, C, H, W = RGB.shape

            output, pseudo_hsi ,pseudo_msi = model(LR, RGB)
#             output = model(LR, RGB)
                                                  
            loss_1 = loss_func(pseudo_hsi, LR)
            loss_2 = loss_func(pseudo_msi, RGB_down)
            loss = loss_func(output, HR)

            total_loss, low_loss, high_loss = calculate_swt_frequency_loss(
                output, HR, 
                wavelet='haar',    # 使用Daubechies 1小波
                level=1,          # 1层分解
                low_loss_weight=0.1,  # 低频损失权重
                high_loss_weight=0.2,  # 高频损失权重 (增加高频重要性)
                loss_type='mse'   # 使用MSE损失
            )

            loss_temp = loss + 0.1 *loss_1 +  0.1 * loss_2 + total_loss 
#             loss_temp = loss  + total_loss 
            
            loss_all.append(np.array(loss_temp.detach().cpu().numpy()))
            optimizer.zero_grad()
            
            loss_temp.backward()
            optimizer.step()
            loop.set_description(f'Epoch [{epoch}/{EPOCH}]')
            loop.set_postfix({'loss': '{0:1.8f}'.format(np.mean(loss_all)), "lr": '{0:1.6f}'.format(lr)})
#         scheduler.step()

        if ((epoch % val_interval == 0) and (epoch >= val_epoch))  or epoch==EPOCH or epoch ==1:
            model.eval()
            checkpoint = {'parameter': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
#                           'scheduler': scheduler.state_dict(),
                          'epoch': epoch,
                          'loss' : loss_eval
                         }
            torch.save(checkpoint, root+model_name+'/model_%04d.pth' % (epoch ))
            val_loss=AverageMeter()
            SAM = Loss_SAM_gpu()
            RMSE = Loss_RMSE()
            PSNR = Loss_PSNR()
            sam = AverageMeter()
            rmse = AverageMeter()
            psnr = AverageMeter()
            
            imglist = os.listdir(test_path)
            with torch.no_grad():
                for i in range(0, len(imglist)):
                    print(i)
                    img = h5.loadmat(test_path + imglist[i])
                    if dataset == 'Harvard':
                        img1 = img["ref"]
                        img1 = img1 / img1.max()
                    if dataset == 'CAVE':
                        img1=img['b']
                        img1 = img1 / img1.max()
                    if dataset == 'KAIST':
                        img1=img['HSI']
                        img1 = img1 / img1.max()
                    print(img1.shape)
                    HRHSI = torch.Tensor(np.transpose(img1, (2, 0, 1)))

                    w, h = int(HRHSI.shape[1] / downsample_factor), int(HRHSI.shape[2] / downsample_factor)
                    HRHSI = HRHSI[:, :w * downsample_factor, :h * downsample_factor]

                    MSI = torch.tensordot(torch.Tensor(R), HRHSI, dims=([1], [0]))
                    HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
                    MSI_1 = torch.unsqueeze(MSI, 0)
                    HSI_LR1 = torch.unsqueeze(torch.Tensor(HSI_LR), 0)  # 加维度 (b,c,h,w)
                    # 计算val_loss用的，防止出错单独拿出来
                    to_fet_loss_hr_hsi=torch.unsqueeze(torch.Tensor(HRHSI), 0)
                    HRHSI = HRHSI.to(device)
                    HSI_LR1 = HSI_LR1.to(device)
                    MSI_1 = MSI_1.to(device)
                    # prediction = model(HSI_LR1.cuda(), MSI_1.cuda())
                    prediction = reconstruction(model, HSI_LR1.cuda(), MSI_1.cuda(), to_fet_loss_hr_hsi,downsample_factor, training_size, stride1,val_loss)
                    sam.update(SAM(HRHSI.permute(1, 2, 0), prediction.squeeze().permute(1, 2, 0)))
                    psnr.update(PSNR(HRHSI.permute(1, 2, 0), prediction.squeeze().permute(1, 2, 0)))

            
                print(f"Epoch {epoch}, Val PSNR: {psnr.avg}, SAM: {sam.avg}")
                val_list = [epoch, lr,np.mean(loss_all), psnr.avg.cpu().detach().numpy(), sam.avg]

                val_data = pd.DataFrame([val_list])
                val_data.to_csv(excel_path,mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了
                time.sleep(0.1)



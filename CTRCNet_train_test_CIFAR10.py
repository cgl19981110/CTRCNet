import os
import torch
import time
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from skimage.metrics import peak_signal_noise_ratio as compute_pnsr
from keras.datasets import cifar100
from CTRCNet_models import CTRCNet

os.environ['CUDA_VISIBLE_DEVICES']='0'

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

# Note that the original data is downloaded from keras.datasets, not from torch.utils.data
def Load_cifar10_data():
    x_train = np.load(r'D:\BaiduSyncdisk\paper\CTRCNet\CTRCNet_main\data\cifar10_raw\x_train.npy')
    x_test = np.load(r'D:\BaiduSyncdisk\paper\CTRCNet\CTRCNet_main\data\cifar10_raw\x_test.npy')
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_test = np.transpose(x_test, (0, 3, 1, 2))
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    return x_train, x_test

# Note that the original data is downloaded from keras.datasets, not from torch.utils.data
def Load_cifar100_data():
    # x_train = np.load('data/cifar100_raw/train')
    # x_test = np.load('data/cifar100_raw/test')
    (x_train, y_train_), (x_test, y_test_) = cifar100.load_data()
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_test = np.transpose(x_test, (0, 3, 1, 2))
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    return x_train, x_test

def Img_transform(test_rec):
    test_rec = test_rec.permute(0, 2, 3, 1)
    test_rec = test_rec.cpu().detach().numpy()
    test_rec = test_rec*255
    test_rec = test_rec.astype(np.uint8)
    return test_rec

def Compute_batch_PSNR(test_input, test_rec):
    psnr_i1 = np.zeros((test_input.shape[0]))
    for j in range(0, test_input.shape[0]):
        psnr_i1[j] = compute_pnsr(test_input[j, :], test_rec[j, :])
    psnr_ave = np.mean(psnr_i1)
    return psnr_ave

class DatasetFolder(Dataset):
    def __init__(self, matData):
        self.matdata = matData
    def __getitem__(self, index):
        return self.matdata[index]
    def __len__(self):
        return self.matdata.shape[0]

BATCH_SIZE = 128
EPOCHS = 150
LEARNING_RATE = 0.0001
PRINT_RREQ = 391
CHANNEL = 'AWGN'           # Choose AWGN or Fading
IMG_SIZE = [3, 32, 32]
N_channels = 256
Kernel_sz = 5
current_epoch = 0
enc_out_shape = [48, IMG_SIZE[1]//4, IMG_SIZE[2]//4]
KSZ = str(Kernel_sz)+'x'+str(Kernel_sz)+'_'

Continue_Train = False
data_train = True  #True: training  and  False test


def worker_init_fn_seed(worker_id):
    seed = 10
    seed += worker_id
    np.random.seed(seed)

if data_train == True:
    x_train, x_test = Load_cifar10_data()
    train_dataset = DatasetFolder(x_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, worker_init_fn=worker_init_fn_seed, pin_memory=True)
    test_dataset = DatasetFolder(x_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
else:
    _, x_test = Load_cifar100_data()
    test_dataset = DatasetFolder(x_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)


if __name__ == '__main__':
    CTRCNet = CTRCNet(enc_out_shape, Kernel_sz, N_channels).cuda()
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(CTRCNet.parameters(), lr=LEARNING_RATE)

    bestLoss = 1e3  
    if Continue_Train == True:
        CTRCNet.load_state_dict(torch.load('./JSCC_models/CTRCNet_'+KSZ+CHANNEL+'_'+str(N_channels)+'_FCF_CIFAR10.pth.tar')['state_dict'])
        current_epoch = 0

    if data_train == True:
        for epoch in range(current_epoch, EPOCHS):
            if epoch == 120:
                LEARNING_RATE =0.00001
            CTRCNet.train()
            begin_time = time.time()
            for i, x_input in enumerate(train_loader):
                torch.cuda.empty_cache()
                x_input = x_input.cuda()

                SNR_TRAIN = torch.randint(-3, 12, (x_input.shape[0], 1)).cuda()
                CR = 0.1+0.9*torch.rand(x_input.shape[0], 1).cuda()
                begin = time.time()
                x_rec = CTRCNet(x_input, SNR_TRAIN, CHANNEL)

                # x_rec = CTRCNet(x_input, SNR_TRAIN, CR, CHANNEL)

                loss = criterion(x_input, x_rec)
                loss = loss.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % PRINT_RREQ == 0:
                    print('Epoch: [{0}][{1}/{2}]\t' 'Loss {loss:.4f}\t'.format(epoch, i, len(train_loader), loss=loss.item()))
            end = time.time()

            # Model Evaluation
            CTRCNet.eval()
            totalLoss = 0
            total_psnr = 0
            with torch.no_grad():
                for i, test_input in enumerate(test_loader):
                    test_input = test_input.cuda()
                    SNR_TEST = torch.randint(-3, 24, (test_input.shape[0], 1)).cuda()
                    CR = 0.1+0.9*torch.rand(test_input.shape[0], 1).cuda()
                    test_rec = CTRCNet(test_input, SNR_TEST, CHANNEL)
                    totalLoss += criterion(test_input, test_rec).item() * test_input.size(0)

                    test_input = Img_transform(test_input)
                    test_rec = Img_transform(test_rec)
                    psnr_ave = Compute_batch_PSNR(test_input, test_rec)
                averageLoss = totalLoss / (len(test_dataset))
                total_psnr += psnr_ave
                averagePSNR = (total_psnr / i)*100

                # print('snr=', SNR_TEST, 'cr=', CR)
                print('averageLoss=', averageLoss, 'PSNR = ' + str(averagePSNR))
            end_time = time.time()
            print(f"Epoch: [{epoch}]time={end_time-begin_time}")

            if averageLoss < bestLoss:
                # Model saving
                if not os.path.exists('./JSCC_models'):
                    os.makedirs('./JSCC_models')
                torch.save({'state_dict': CTRCNet.state_dict(), }, './JSCC_models/CTRCNet_'+KSZ+CHANNEL+'_'+str(N_channels)+'_FCF_CIFAR10.pth.tar')
                # print('Model saved')
                bestLoss = averageLoss
            torch.cuda.empty_cache()

    else:
        CR_INDEX = torch.Tensor([8, 6, 4, 2]).int()

        for m in range(0, 4):
            cr = 1 / CR_INDEX[m]

            CTRCNet.load_state_dict(torch.load('./JSCC_models/CTRCNet_'+KSZ+CHANNEL+'_'+str(N_channels)+'_FCF_CIFAR10.pth.tar')['state_dict'])
            for k in range(0, 7):
                print('Evaluating CTRCNet with CR = ' + str(cr.item()) + ' and SNR = ' + str(3 * k - 3) + 'dB')

                total_psnr = 0
                CTRCNet.eval()
                with torch.no_grad():
                    tex_begin = time.time()
                    for i, test_input in enumerate(test_loader):
                        SNR = (3 * k - 3) * torch.ones((test_input.shape[0], 1))
                        CR = cr * torch.ones((test_input.shape[0], 1))
                        SNR = SNR.cuda()
                        CR = CR.cuda()
                        test_input = test_input.cuda()
                        test_rec = CTRCNet(test_input, SNR, CR, CHANNEL)
                        test_input = Img_transform(test_input)
                        test_rec = Img_transform(test_rec)
                        psnr_ave = Compute_batch_PSNR(test_input, test_rec)
                        total_psnr += psnr_ave
                    averagePSNR = total_psnr / i
                    tex_end = time.time()
                    print('PSNR = ' + str(averagePSNR))
                    print(f"Epoch: time={tex_end-tex_begin}")
    print('Training for CTRCNet is finished!')







































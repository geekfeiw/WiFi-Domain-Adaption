import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
import sys
import glob
import hdf5storage
from random import shuffle
import time
import os


# get each minibatch by file names, names shuffled
def getMinibatchforPair(file_names, scene_label_start, scene_label_end, hm_depth=44, hm_height=46, hm_width=82):
    file_num = len(file_names)
    if file_num%2 == 0:
        # data_diff = torch.zeros(int(file_num/2), 30 * 5 * 2, 3, 3)
        csi_data = torch.zeros(int(file_num), 30 * 5, 3, 3)
        scene_diff_label = torch.zeros(int(file_num/2))
        mpi_hm_label = torch.zeros(int(file_num), hm_depth, hm_height, hm_width)
        for i in range(int(file_num/2)):
            data1 = hdf5storage.loadmat(file_names[2*i], variable_names={'csi_serial', 'heatmapsmpi'})
            data2 = hdf5storage.loadmat(file_names[2*i+1], variable_names={'csi_serial', 'heatmapsmpi'})

            csi_data[2*i, :, :, :] = torch.from_numpy(data1['csi_serial']).type(torch.FloatTensor).view(-1, 3, 3)
            csi_data[2*i+1, :, :, :] = torch.from_numpy(data2['csi_serial']).type(torch.FloatTensor).view(-1, 3, 3)

            mpi_hm_label[2*i:, :, :, :] = torch.from_numpy(data1['heatmapsmpi']).type(torch.FloatTensor)
            mpi_hm_label[2*i+1:, :, :, :] = torch.from_numpy(data2['heatmapsmpi']).type(torch.FloatTensor)

            scene_diff_label[i] = int(file_names[2*i][scene_label_start:scene_label_end] == file_names[2*i+1][scene_label_start:scene_label_end])

    else:
        file_num = file_num - 1
        csi_data = torch.zeros(int(file_num), 30 * 5, 3, 3)
        mpi_hm_label = torch.zeros(int(file_num), hm_depth, hm_height, hm_width)
        scene_diff_label = torch.zeros(int(file_num / 2))

        for i in range(int(file_num / 2)):
            data1 = hdf5storage.loadmat(file_names[2 * i], variable_names={'csi_serial', 'heatmapsmpi'})
            data2 = hdf5storage.loadmat(file_names[2 * i + 1], variable_names={'csi_serial', 'heatmapsmpi'})

            csi_data[2 * i, :, :, :] = torch.from_numpy(data1['csi_serial']).type(torch.FloatTensor).view(-1, 3, 3)
            csi_data[2 * i + 1, :, :, :] = torch.from_numpy(data2['csi_serial']).type(torch.FloatTensor).view(-1, 3, 3)

            mpi_hm_label[2 * i:, :, :, :] = torch.from_numpy(data1['heatmapsmpi']).type(torch.FloatTensor)
            mpi_hm_label[2 * i + 1:, :, :, :] = torch.from_numpy(data2['heatmapsmpi']).type(torch.FloatTensor)

            scene_diff_label[i] = int(file_names[2 * i][scene_label_start:scene_label_end] == file_names[2 * i + 1][scene_label_start:scene_label_end])

    return csi_data, mpi_hm_label, scene_diff_label


def getConfidence(weight, k=1, b=1):
    weight = k * F.relu(weight)
    weight = weight + b
    return weight


batch_size = 32
num_epochs = 20
learning_rate = 0.001


from models.data_transformer import DT
from models.pair_discriminator import ResNet, ResidualBlock
from models.pose_unet import UNet

dataTrans = DT(n_channels=150).cuda()
pairDiscri = ResNet(ResidualBlock, [2, 2, 2, 2], number_class=2).cuda()
poseNet = UNet().cuda()

optimizer_DT = torch.optim.Adam(dataTrans.parameters(), lr=learning_rate)
optimizer_PD = torch.optim.Adam(pairDiscri.parameters(), lr=learning_rate)
optimizer_PN = torch.optim.Adam(poseNet.parameters(), lr=learning_rate)

criterion_L2 = nn.MSELoss(reduction='mean').cuda()
criterion_BCE = nn.BCEWithLogitsLoss(reduction='mean').cuda()
criterion_CE = nn.CrossEntropyLoss(reduction='mean').cuda()


scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_DT, milestones=[5, 10, 12, 15, 18, 25, 30], gamma=0.5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_PN, milestones=[5, 10, 12, 15, 18, 25, 30], gamma=0.5)


dataTrans.train()
pairDiscri.eval()
poseNet.train()

mats = glob.glob('E:/wifiposedata/train80/*.mat')
scene_label_start = len('E:/wifiposedata/train80/')
scene_label_end = len('E:/wifiposedata/train80/sep12set1_')

# mats = mats[::10000]   # downsampling 1/10 them

mats_num = len(mats)
batch_num = int(np.floor(mats_num/batch_size))
mats_num = len(mats)
batch_num = int(np.floor(mats_num/batch_size))

loss_joints_list = []
loss_pafs_list = []


for epoch_index in range(num_epochs):
    scheduler.step()
    dataTrans.train()

    start = time.time()
    # shuffling dataset
    shuffle(mats)
    loss_x = 0
    # in each minibatch
    for batch_index in range(batch_num):
        if batch_index < batch_num:
            file_names = mats[batch_index*batch_size:(batch_index+1)*batch_size]
        else:
            file_names = mats[batch_num*batch_size:]

        csi_data, heatmap_label, scene_diff_label = getMinibatchforPair(file_names, scene_label_start, scene_label_end)
        csi_data = Variable(csi_data.cuda())
        heatmap_label = Variable(heatmap_label.cuda())
        scene_diff_label = Variable(scene_diff_label.type(torch.LongTensor).cuda())

        optimizer_DT.zero_grad()
        optimizer_PN.zero_grad()
        optimizer_PD.zero_grad()

        trans_data = dataTrans(csi_data)
        trans_data = trans_data.view(int(trans_data.shape[0]/2), -1, 3, 3)
        pair_predict = pairDiscri(trans_data)

        loss_pd = criterion_CE(pair_predict, scene_diff_label)

        # loss.backward()
        # optimizer_DT.step()

        predict_hm, predict_bg, predict_paf = poseNet(csi_data)
        predict_heat = torch.cat((predict_hm, predict_bg), dim=1)

        hm_conf = getConfidence(heatmap_label[:, 0:16, :, :], k=1, b=1)
        paf_conf = getConfidence(heatmap_label[:, 16:, :, :], k=1, b=0.3)
        # heat_conf = torch.cat((hm_conf, torch.ones([hm_conf.shape[0], 1, 46, 82]).cuda()), dim=1)

        loss_heatmap = criterion_L2(torch.mul(predict_heat, hm_conf), torch.mul(heatmap_label[:, 0:16, :, :], hm_conf))
        loss_pafs = criterion_L2(torch.mul(predict_paf, paf_conf), torch.mul(heatmap_label[:, 16:, :, :], paf_conf))
        # loss_mask = criterion_BCE(predict_mask, mask_label)

        # loss = loss_heatmap + loss_pafs + 0.05*loss_mask
        loss = loss_pd + loss_heatmap + loss_pafs

        loss_joints_list.append(loss_heatmap.item())
        loss_pafs_list.append(loss_pafs.item())

        loss.backward()

        optimizer_DT.step()
        optimizer_PD.step()

    endl = time.time()
    # torch.save(net, 'E:/wifiposedata/weights/' + pkl_name + '_epoch' + str(epoch_index) + '.pkl')
    print('Costing time:', (endl-start)/60)



# torch.save(net, 'E:/wifiposedata/weights/' + pkl_name + '.pkl')
#
# hdf5storage.savemat('E:/wifiposedata/weights/' + pkl_name + '_loss_joints.mat', {'loss_joints_list': loss_joints_list})
# hdf5storage.savemat('E:/wifiposedata/weights/' + pkl_name + '_loss_pafs.mat', {'loss_pafs_list': loss_pafs_list})
#
# print('cool')
#
# # plt.plot(loss_joints_list)
# # plt.plot(loss_pafs_list)
# # plt.show()
#
#
# mats = glob.glob('E:/wifiposedata/train80/*.mat')
# # mats = mats[::1000]   # downsampling 1/10 them
# #
# mats_num = len(mats)
# batch_num = int(np.floor(mats_num/batch_size))

# os.makedirs('E:/result/'+pkl_name)
#
# net = net.cuda().eval()
# # in each minibatch
# for batch_index in range(batch_num+1):
#     if batch_index < batch_num:
#         file_names = mats[batch_index*batch_size:(batch_index+1)*batch_size]
#     else:
#         file_names = mats[batch_num*batch_size:]
#
#     csi_data, heatmap_label = getMinibatch(file_names)
#
#     with torch.no_grad():
#         csi_data = Variable(csi_data.cuda())
#         # 46x82
#
#         predict_hm, predict_bg, predict_paf = net(csi_data)
#         predict_label = torch.cat((predict_hm, predict_bg, predict_paf), dim=1)
#
#         for i in range(len(csi_data)):
#             hdf5storage.savemat('E:/result/'+pkl_name+'/out_'+file_names[i][24:], {'heatmapsmpi': predict_label[i,:,:,:].view(1,1,44,46,82).cpu().numpy()})
#
#
# mats = glob.glob('E:/wifiposedata/test20/oct17set10*.mat')
# # mats = mats[::1000]   # downsampling 1/10 them
# #
# mats_num = len(mats)
# batch_num = int(np.floor(mats_num/batch_size))
#
# os.makedirs('E:/result/'+pkl_name)
#
# net = net.cuda().eval()
# # in each minibatch
# for batch_index in range(batch_num+1):
#     if batch_index < batch_num:
#         file_names = mats[batch_index*batch_size:(batch_index+1)*batch_size]
#     else:
#         file_names = mats[batch_num*batch_size:]
#
#     csi_data, heatmap_label = getMinibatch(file_names)
#
#     with torch.no_grad():
#         csi_data = Variable(csi_data.cuda())
#         # 46x82
#
#         predict_hm, predict_bg, predict_paf = net(csi_data)
#         predict_label = torch.cat((predict_hm, predict_bg, predict_paf), dim=1)
#
#         for i in range(len(csi_data)):
#             hdf5storage.savemat('E:/result/'+pkl_name+'/out_'+file_names[i][23:], {'heatmapscoco18': predict_label[i,:,:,:].view(1,1,57,46,82).cpu().numpy()})

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Dataset
import torch as T
import configparser
from torchinfo import summary
from astropy.timeseries import LombScargle
from PyAstronomy.pyTiming import pyPeriod
from sklearn.model_selection import KFold


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):

        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,**kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self,x):

        return F.relu(self.bn(self.conv(x)))

class LinearDrop(nn.Module):

    def __init__(self, in_channels, out_channels, drop_ratio):

        super(LinearDrop, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.drop = nn.Dropout(p=drop_ratio)

    def forward(self,x):

        return F.relu(self.linear(self.drop(x)))




class Shell_Net_v3(nn.Module):
    def __init__(self, num_classes):
        super(Shell_Net_v3, self).__init__()
        self.conv = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=112, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(112),
            nn.Conv2d(in_channels=112, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=48, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.AdaptiveAvgPool2d((5,5))

        )


        self.flatten = nn.Flatten()

        self.linear1 = nn.Sequential(
            nn.Linear(1200, 64),
            nn.ReLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(64, 96),
            nn.ReLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(96, num_classes)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(1200, 64),
            nn.ReLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(64, 96),
            nn.ReLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(96, num_classes)
        )

        self.linear3 = nn.Sequential(
            nn.Linear(1200, 64),
            nn.ReLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(64, 96),
            nn.ReLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(96, num_classes)
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.flatten(out)
        out_rv = self.linear1(out)
        out_fwhm = self.linear2(out)
        out_span = self.linear3(out)

        return out_rv, out_fwhm, out_span



class Shell_Net_optuna_mad(nn.Module):
    def __init__(self, num_classes):
        super(Shell_Net_optuna_mad, self).__init__()
        self.conv = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=112, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(112),
            nn.AdaptiveAvgPool2d((3,3))

        )


        self.flatten = nn.Flatten()

        self.linear1 = nn.Sequential(
            nn.Linear(1008, 480),
            nn.ReLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(480, 512),
            nn.ReLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(32, num_classes)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(1008, 480),
            nn.ReLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(480, 512),
            nn.ReLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(32, num_classes)
        )

        self.linear3 = nn.Sequential(
            nn.Linear(1008, 480),
            nn.ReLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(480, 512),
            nn.ReLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.flatten(out)
        out_rv = self.linear1(out)
        out_fwhm = self.linear2(out)
        out_span = self.linear3(out)

        return out_rv, out_fwhm, out_span



class Shell_Net_optuna_activity(nn.Module):
    def __init__(self, num_classes):
        super(Shell_Net_optuna_activity, self).__init__()
        self.conv = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.AdaptiveAvgPool2d((3,3))

        )


        self.flatten = nn.Flatten()

        self.linear1 = nn.Sequential(
            nn.Linear(432, 352),
            nn.ReLU(),
            nn.Dropout(p = 0.2),
            nn.Linear(352, 416),
            nn.ReLU(),
            nn.Dropout(p = 0.2),
            nn.Linear(416, num_classes)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(432, 352),
            nn.ReLU(),
            nn.Dropout(p = 0.2),
            nn.Linear(352, 416),
            nn.ReLU(),
            nn.Dropout(p = 0.2),
            nn.Linear(416, num_classes)
        )

        self.linear3 = nn.Sequential(
            nn.Linear(432, 352),
            nn.ReLU(),
            nn.Dropout(p = 0.2),
            nn.Linear(352, 416),
            nn.ReLU(),
            nn.Dropout(p = 0.2),
            nn.Linear(416, num_classes)
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.flatten(out)
        out_rv = self.linear1(out)
        out_fwhm = self.linear2(out)
        out_span = self.linear3(out)

        return out_rv, out_fwhm, out_span

class ShellDataset(T.utils.data.Dataset):
    def __init__(self, data_frame_x, data_frame_y, data_frame_z, data_frame_m):
        x_train_cpu = data_frame_x.reshape(-1,1,10,10)
        y_train_cpu = data_frame_y.reshape(-1,1)
        z_train_cpu = data_frame_z.reshape(-1,1)
        m_train_cpu = data_frame_m.reshape(-1,1)

        self.x_data = T.tensor(x_train_cpu, dtype=T.float32).to(device)
        self.y_data = T.tensor(y_train_cpu, dtype=T.float32).to(device)
        self.z_data = T.tensor(z_train_cpu, dtype=T.float32).to(device)
        self.m_data = T.tensor(m_train_cpu, dtype=T.float32).to(device)


    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if T.is_tensor(idx):
            idx = idx.tolist()
        train_x_gpu = self.x_data[idx, :,:,:]
        train_y_gpu = self.y_data[idx,:]
        train_z_gpu = self.z_data[idx,:]
        train_m_gpu = self.m_data[idx,:]
        sample =  { 'shells' : train_x_gpu, 'rvs' : train_y_gpu, 'fwhm': train_z_gpu, 'span':train_m_gpu}
        return sample



batch_size = 8
device = T.device("cuda")


framework = 'activity'

stage = 'matching_mad'


for i_grid in np.arange(180):


    prefix =
    prefix_input =
    info_limits_file =
    train_limits_file =



    info_planet =  np.load(info_limits_file)
    train_b = np.load(train_limits_file)['Temp_map_vel']
    info_indicator = np.load(info_limits_file)


    train_x = np.zeros((train_b.shape[0], 1, train_b.shape[1], train_b.shape[2]))
    train_x[:,0,:,:] = train_b


    train_RV_old =   info_planet['RV_planets']-info_planet['RV_planets'].min()
    mean_train_rv = np.mean(train_RV_old)
    std_train_rv = np.std(train_RV_old)
    train_RV = (train_RV_old.copy() - np.mean(train_RV_old))/std_train_rv*100
    train_y = np.zeros((train_x.shape[0], 1))
    train_y[:,0] = train_RV


    jdb_array = info_planet['jdbs'].reshape((-1,1))

    train_fwhm_old =  info_indicator['FWHM_planets']
    mean_train_fwhm = np.mean(train_fwhm_old)
    std_train_fwhm = np.std(train_fwhm_old)

    train_fwhm = (train_fwhm_old.copy() - np.mean(train_fwhm_old))/std_train_fwhm*100
    train_z = np.zeros((train_x.shape[0], 1))
    train_z[:,0] = train_fwhm


    train_span_old =  info_indicator['span_planets']
    mean_train_span = np.mean(train_span_old)
    std_train_span = np.std(train_span_old)

    train_span = (train_span_old.copy() - np.mean(train_span_old))/std_train_span*100
    train_m = np.zeros((train_x.shape[0], 1))
    train_m[:,0] = train_span




    k_folds = 10
    kfold = KFold(n_splits=k_folds, shuffle=True)

    jdb_list = []

    raw_rv_list = []
    pred_rv_list = []
    residual_rv_list = []

    raw_fwhm_list = []
    pred_fwhm_list = []
    residual_fwhm_list = []

    raw_span_list = []
    pred_span_list = []
    residual_span_list = []

    num_epochs = 150
    accuracy_array = np.zeros((k_folds,num_epochs))
    accuracy_array_rv = np.zeros((k_folds,num_epochs))
    accuracy_array_fwhm = np.zeros((k_folds,num_epochs))
    accuracy_array_span = np.zeros((k_folds,num_epochs))

    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_x)):

        print(f'FOLD {fold}')
        print('--------------------------------')

        train_x_fold = train_x[train_ids,:,:,:]
        train_y_fold = train_y[train_ids,:]
        train_z_fold = train_z[train_ids,:]
        train_m_fold = train_m[train_ids,:]

        print('--------------------------------')
        test_x_fold = train_x[test_ids,:,:,:]
        test_y_fold = train_y[test_ids,:]
        test_z_fold = train_z[test_ids,:]
        test_m_fold = train_m[test_ids,:]

        jdb_array_fold = jdb_array[test_ids,:]


        train_set_loader = T.utils.data.DataLoader(ShellDataset(train_x_fold, train_y_fold, train_z_fold, train_m_fold),batch_size=batch_size, shuffle=True)
        test_set_loader = T.utils.data.DataLoader(ShellDataset(test_x_fold, test_y_fold, test_z_fold, test_m_fold),batch_size=batch_size, shuffle=False)

        training_steps = len(train_set_loader.dataset)// batch_size
        test_steps = len(test_set_loader.dataset)// batch_size


        if framework == 'old':
            model = Shell_Net_v3(1)
        elif framework == 'mad':
            model = Shell_Net_optuna_mad(1)
        else:
            model =  Shell_Net_optuna_activity(1)

        criterion1 = nn.MSELoss()
        criterion2 = nn.MSELoss()
        criterion3 = nn.MSELoss()


        if framework == 'old':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=2.28e-5)
        elif framework == 'mad':
            optimizer = torch.optim.Adam(model.parameters(), lr=1.801e-05)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=1.8474e-05)


        if torch.cuda.is_available():
            model = model.cuda()
            criterion1 = criterion1.cuda()
            criterion2 = criterion2.cuda()
            criterion3 = criterion3.cuda()



        summary(model, input_size= (batch_size,1 ,10, 10))

        H = {"train_loss": [],"val_loss": []}


        for epoch in range(num_epochs):
            model.train()
            totalTrainLoss = 0
            totalValLoss = 0
            totalValLoss_rv = 0
            totalValLoss_fwhm = 0
            totalValLoss_span = 0

            for (batch_idx, training_sets) in enumerate(train_set_loader):

                (x_train_dev, y_train_dev, z_train_dev, m_train_dev) = (training_sets['shells'], training_sets['rvs'], training_sets['fwhm'], training_sets['span'])

                optimizer.zero_grad()
                output_train_y, output_train_z, output_train_m = model(x_train_dev)
                loss_train_y = criterion1(output_train_y, y_train_dev)
                loss_train_z = criterion1(output_train_z, z_train_dev)
                loss_train_m = criterion1(output_train_m, m_train_dev)

                loss_train_final = loss_train_y+loss_train_z+loss_train_m

                loss_train_final.backward()
                optimizer.step()
                totalTrainLoss += loss_train_final

            with torch.no_grad():

                model.eval()
                for (batch_idx, test_sets) in enumerate(test_set_loader):

                    (x_test_dev, y_test_dev, z_test_dev, m_test_dev) = (test_sets['shells'], test_sets['rvs'], test_sets['fwhm'], test_sets['span'])
                    pred_y, pred_z, pred_m = model(x_test_dev)


                    loss_val_y = criterion1(pred_y, y_test_dev)
                    loss_val_z = criterion2(pred_z, z_test_dev)
                    loss_val_m = criterion3(pred_m, m_test_dev)
                    valloss_final = (loss_val_y+loss_val_z+loss_val_m)/3.0

                    totalValLoss += valloss_final
                    totalValLoss_rv += loss_val_y
                    totalValLoss_fwhm += loss_val_z
                    totalValLoss_span += loss_val_m

            avgTrainLoss = totalTrainLoss / training_steps
            avgValLoss = totalValLoss / test_steps
            avgValLoss_rv = totalValLoss_rv / test_steps
            avgValLoss_fwhm = totalValLoss_fwhm / test_steps
            avgValLoss_span = totalValLoss_span / test_steps

            print("[INFO] EPOCH: {}/{}".format(epoch + 1, num_epochs))
            print("Train loss: {:.6f} ".format(avgTrainLoss))
            print("Val loss: {:.6f}\n".format(avgValLoss))

            H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            H["val_loss"].append(avgValLoss.cpu().detach().numpy())

            accuracy_array[fold,epoch] = avgValLoss.cpu().detach().numpy()
            accuracy_array_rv[fold,epoch] = avgValLoss_rv.cpu().detach().numpy()
            accuracy_array_fwhm[fold,epoch] = avgValLoss_fwhm.cpu().detach().numpy()
            accuracy_array_span[fold,epoch] = avgValLoss_span.cpu().detach().numpy()


        with torch.no_grad():
            model.eval()

            preds_RV = []
            preds_fwhm = []
            preds_span = []

            for (batch_idx, test_sets) in enumerate(test_set_loader):

                (x_test_dev, y_test_dev, z_test_dev, m_test_dev) = (test_sets['shells'], test_sets['rvs'], test_sets['fwhm'], test_sets['span'])
                pred_y, pred_z, pred_m = model(x_test_dev)

                preds_RV.extend( pred_y[:,0].cpu().detach().numpy() )
                preds_fwhm.extend(pred_z[:,0].cpu().detach().numpy() )
                preds_span.extend(pred_m[:,0].cpu().detach().numpy() )

        final_RV = (np.array(preds_RV)*std_train_rv/100+mean_train_rv).reshape(-1)
        y_fold = (np.array(test_y_fold)*std_train_rv/100+mean_train_rv).reshape(-1)
        RV_diff = y_fold - final_RV


        final_fwhm = (np.array(preds_fwhm)*std_train_fwhm/100+mean_train_fwhm).reshape(-1)
        z_fold = (np.array(test_z_fold)*std_train_fwhm/100+mean_train_fwhm).reshape(-1)
        fwhm_diff = z_fold - final_fwhm


        final_span = (np.array(preds_span)*std_train_span/100+mean_train_span).reshape(-1)
        m_fold = (np.array(test_m_fold)*std_train_span/100+mean_train_span).reshape(-1)
        span_diff = m_fold - final_span

        jdb_list.append(jdb_array_fold.reshape(-1))
        raw_rv_list.append(y_fold)
        pred_rv_list.append(final_RV)
        residual_rv_list.append(RV_diff)

        raw_fwhm_list.append(z_fold)
        pred_fwhm_list.append(final_fwhm)
        residual_fwhm_list.append(fwhm_diff)


        raw_span_list.append(m_fold)
        pred_span_list.append(final_span)
        residual_span_list.append(span_diff)






    accuracy_avg_fold = np.mean(accuracy_array,axis=0)
    accuracy_avg_fold_rv = np.mean(accuracy_array_rv,axis=0)
    accuracy_avg_fold_fwhm = np.mean(accuracy_array_fwhm,axis=0)
    accuracy_avg_fold_span = np.mean(accuracy_array_span,axis=0)


    accuracy_out_frame = {}
    accuracy_out_frame['rv'] = accuracy_avg_fold_rv
    accuracy_out_frame['fwhm'] = accuracy_avg_fold_fwhm
    accuracy_out_frame['span'] = accuracy_avg_fold_span
    accuracy_out_frame['accuracy'] = accuracy_avg_fold

    prefix_out =
    accuracy_name_limit =
    np.savez(accuracy_name_limit,**accuracy_out_frame)

    data_out_frame = {}
    data_out_frame['injected'] = np.array(raw_rv_list,dtype='double')
    data_out_frame['predicted'] = np.array(pred_rv_list,dtype='double')
    data_out_frame['residual'] = np.array(residual_rv_list,dtype='double')

    data_out_frame['injected_fwhm'] = np.array(raw_fwhm_list,dtype='double')
    data_out_frame['predicted_fwhm'] = np.array(pred_fwhm_list,dtype='double')
    data_out_frame['residual_fwhm'] = np.array(residual_fwhm_list,dtype='double')

    data_out_frame['injected_span'] = np.array(raw_span_list,dtype='double')
    data_out_frame['predicted_span'] = np.array(pred_span_list,dtype='double')
    data_out_frame['residual_span'] = np.array(residual_span_list,dtype='double')

    data_out_frame['jds'] =  np.array(jdb_list,dtype='double')

    result_name_limit =
    np.savez(result_name_limit,**data_out_frame)

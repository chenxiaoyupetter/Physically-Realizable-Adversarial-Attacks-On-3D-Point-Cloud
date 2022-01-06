from pytorch.renderer import nmr
import torch
import torch.autograd as autograd
import argparse
import cv2
from c2p_segmentation import *
import loss_LiDAR
import numpy as np
import cluster
import os
from xyz2grid import *
import render
from plyfile import *
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from open3d import geometry
from FPFH import FPFH
from tqdm import tqdm
import random


class attack_msf():
    def __init__(self, args):
        self.args = args
        self.num_pos = 1
        self.threshold = 0.4
        self.root_path = '/home/car/桌面/pythonProject/MSF-ADV-master/data/'
        self.pclpath = 'pcd/'
        self.rotation = torch.tensor(np.array([[1., 0., 0.],
                                               [0., 0., -1.],
                                               [0., 1., 0.]]), dtype=torch.float)
        self.protofile = self.root_path + 'deploy.prototxt'
        self.weightfile = self.root_path + 'deploy.caffemodel'
        self.outputs = ['instance_pt', 'category_score', 'confidence_score',
                   'height_pt', 'heading_pt', 'class_score']
        self.esp = args.epsilon
        self.direction_val, self.dist_val = self.load_const_features('./data/features_1.out')

    def load_const_features(self, fname):
        print("Loading dircetion, dist")
        features_filename = fname

        features = np.loadtxt(features_filename)
        features = np.swapaxes(features, 0, 1)
        features = np.reshape(features, (1, 512, 512, 8))

        direction = np.reshape(features[:, :, :, 3], (1, 512, 512, 1))
        dist = np.reshape(features[:, :, :, 6], (1, 512, 512, 1))
        return torch.tensor(direction).cuda().float(), torch.tensor(dist).cuda().float()

    def model_val_lidar(self, protofile, weightfile):
        net = CaffeNet(protofile, phase='TEST')
        # torch.cuda.set_device(0)
        net.cuda()
        net.load_weights(weightfile)
        net.set_train_outputs(outputs)
        net.set_eval_outputs(outputs)
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
        return net

    def load_LiDAR_model(self, ):
        self.LiDAR_model = generatePytorch(self.protofile, self.weightfile).cuda()
        self.LiDAR_model_val = self.model_val_lidar(self.protofile, self.weightfile)

    def load_pc_mesh(self, path):
        PCL_path = path

        # loading ray_direction and distance for the background pcd
        self.PCL = loadPCL(PCL_path, True)
        x_final = torch.FloatTensor(self.PCL[:, 0]).cuda()
        y_final = torch.FloatTensor(self.PCL[:, 1]).cuda()
        z_final = torch.FloatTensor(self.PCL[:, 2]).cuda()
        self.i_final = torch.FloatTensor(self.PCL[:, 3]).cuda()
        self.ray_direction, self.length = render.get_ray(x_final, y_final, z_final)

    def set_learning_rate(self, optimizer, learning_rate):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    def load_bg(self, path, h=416, w=416):
        background = cv2.imread(path)
        background = cv2.resize(background, (h, w))
        background = background[:, :, ::-1] / 255.0
        self.background = background.astype(np.float32)

    def l2_loss(self, desk_t, desk_v, ori_desk_t, ori_desk_v):
        t_loss = torch.nn.functional.mse_loss(desk_t, ori_desk_t)
        v_loss = torch.nn.functional.mse_loss(desk_v, ori_desk_v)
        return v_loss, t_loss

    def load_mesh(self, path, r,x_of=10,y_of=0):
        #x_of =  random.uniform(10,11)
        #y_of = random.uniform(-0.3,0.3)
        print("x_of",x_of,"y_of",y_of)
        z_of = -1.73 + r / 2.
        plydata = PlyData.read(path)
        x = torch.FloatTensor(plydata['vertex']['x']) * r
        y = torch.FloatTensor(plydata['vertex']['y']) * r
        z = torch.FloatTensor(plydata['vertex']['z']) * r
        self.object_v = torch.stack([x, y, z], dim=1).cuda()

        self.object_f = plydata['face'].data['vertex_indices']
        self.object_f = torch.tensor(np.vstack(self.object_f)).cuda()

        r = R.from_euler('zxy', [10, 80, 4], degrees=True)
        lidar_rotation = torch.tensor(r.as_matrix(), dtype=torch.float).cuda()
        rotation = lidar_rotation.cuda()
        self.object_v = self.object_v.cuda()
        self.object_v = self.object_v.permute(1, 0)
        self.object_v = torch.matmul(rotation, self.object_v)
        self.object_v = self.object_v.permute(1, 0)
        self.object_v[:, 0] += x_of
        self.object_v[:, 1] += y_of
        self.object_v[:, 2] += z_of

        self.object_ori = self.object_v.clone()

    def preprocess_point_cloud(pcd, voxel_size):
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(0.0001)

        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh
    def rendering_img(self, ppath):

        u_offset = 0
        v_offset = 0

        lr = 0.03
        best_it = 1e10
        num_class = 80
        threshold = 0.5
        batch_size = 1
        icp = FPFH(e=0.1, div=2, nneighbors=8, rad=2)
        self.object_v.requires_grad = True
        bx = self.object_v.clone().detach().requires_grad_()
        sample_diff = np.random.uniform(-0.001, 0.001, self.object_v.shape)
        sample_diff = torch.tensor(sample_diff).cuda().float()
        sample_diff.clamp_(-self.args.epsilon, self.args.epsilon)
        self.object_v.data = sample_diff + bx
        iteration = self.args.iteration
        # pc = render.render(self.ray_direction, self.length, self.object_v, self.object_f, self.i_final)
        ori_fpfh = icp.solve(self.object_v)
        if self.args.opt == 'Adam':
            from torch.optim import Adam
            opt = Adam([self.object_v], lr=lr, amsgrad=True)
        self.object_v = self.object_v.cuda()
        for it in tqdm(range(iteration)):

            if it % 200 == 0:
                lr = lr / 10.0
                print("lr",lr)
            l_c_c_ori = self.object_ori
            self.object_f = self.object_f.cuda()
            self.i_final = self.i_final.cuda()

            self.object_v = self.object_v.cuda()
            # self.object_v = self.random_obj(self.object_v)
            adv_total_loss = None

            point_cloud = render.render(self.ray_direction, self.length, self.object_v, self.object_f, self.i_final)

            recent_fpfh = icp.solve(self.object_v)
            fpfh_score = torch.norm(ori_fpfh-recent_fpfh, dim=1).mean()
            #print(f"fpfh_score: {fpfh_score}")
            # distance = []
            # for i in range(len(recent_fpfh)):
            #     dist = []
            #     for j in range(len(ori_fpfh)):
            #         dist.append(np.linalg.norm(recent_fpfh[i] - ori_fpfh[j]))
            #     distance.append(dist)
            # fpfh_score = np.array(distance)
            # fpfh_score = np.min(fpfh_score, axis=1)
            a=6
            if a ==5:
                pc = geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(self.PCL[:, :3])
                o3d.visualization.draw_geometries([pc])
                pc.points = o3d.utility.Vector3dVector(point_cloud.cpu().detach().numpy()[:, :3])
                o3d.visualization.draw_geometries([pc])

            grid = xyzi2grid_v2(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], point_cloud[:, 3])

            featureM = gridi2feature_v2(grid, self.direction_val, self.dist_val)

            outputPytorch = self.LiDAR_model(featureM)

            lossValue,loss_object = loss_LiDAR.lossRenderAttack(outputPytorch,
                                                     self.object_v,
                                                     self.object_ori,
                                                     self.object_f,
                                                     0.05)
            total_loss = lossValue + fpfh_score*0.001
            if best_it > loss_object.data.cpu() or it == 0:
                best_it = loss_object.data.cpu().clone()
                best_vertex = self.object_v.data.cpu().clone()
                best_face = self.object_f.data.cpu().clone()
                best_out_lidar = outputPytorch[:]
                pc_ = point_cloud[:, :3].cpu().detach().numpy()
                vertice = best_vertex.numpy()
                face = best_face.numpy()
                pp = ppath.split('/')[-1].split('.bin')[0]
                render.savemesh(self.args.object, self.args.object_save + pp + str(it) + '_v2.ply', vertice, face, r=0.75)

            print('Iteration {} of {}: Loss={}'.format(it, iteration, total_loss.data.cpu().numpy()))
            self.object_v = self.object_v.cuda()
            print(self.args.opt)
            if self.args.opt == "Adam":
                opt.zero_grad()
                total_loss.backward(retain_graph=True)
                opt.step()
            else:
                pgd_grad = autograd.grad([total_loss.sum()], [self.object_v])[0]
                with torch.no_grad():
                    loss_grad_sign = pgd_grad.sign()
                    self.object_v.data.add_(-lr, loss_grad_sign)
                    diff = self.object_v - bx
                    diff.clamp_(-self.esp, self.esp)
                    self.object_v.data = diff + bx
                del pgd_grad
                del diff

            if it < iteration - 1:
                del total_loss
                del featureM
                del grid
                del point_cloud


        print('best iter: {}'.format(best_it))
        diff = self.object_v - bx
        vertice = best_vertex.numpy()
        face = best_face.numpy()
        pp = ppath.split('/')[-1].split('.bin')[0]
        render.savemesh(self.args.object, self.args.object_save + pp + '_v2.ply', vertice, face, r=0.75)

        print('x range: ', vertice[:, 0].max() - vertice[:, 0].min())
        print('y range: ', vertice[:, 1].max() - vertice[:, 1].min())
        print('z range: ', vertice[:, 2].max() - vertice[:, 2].min())
        ######################
        PCLConverted = mapPointToGrid(pc_)

        print('------------  Pytorch Output ------------')
        obj, label_map = cluster.cluster(best_out_lidar[1].cpu().detach().numpy(),
                                         best_out_lidar[2].cpu().detach().numpy(),
                                         best_out_lidar[3].cpu().detach().numpy(),
                                         best_out_lidar[0].cpu().detach().numpy(),
                                         best_out_lidar[5].cpu().detach().numpy())

        obstacle, cluster_id_list = twod2threed(obj, label_map, self.PCL, PCLConverted)
        self.pc_save = pc_
        self.best_vertex = best_vertex.numpy()
        self.benign = bx.clone().data.cpu().numpy()

    def savemesh(self, path_r, path_w, vet, r):
        plydata = PlyData.read(path_r)
        plydata['vertex']['x'] = vet[:, 0] / r
        plydata['vertex']['y'] = vet[:, 1] / r
        plydata['vertex']['z'] = vet[:, 2] / r

        plydata.write(path_w)
        return

    def set_neighbor_graph(self, f, vn, degree=1):
        max_len = 0
        face = f.cpu().data.numpy()
        vn = vn.data.cpu().tolist()
        for i in range(len(face)):
            v1, v2, v3 = face[i]
            for v in [v1, v2, v3]:
                vn[v].append(v2)
                vn[v].append(v3)
                vn[v].append(v1)

        # two degree
        for i in range(len(vn)):
            vn[i] = list(set(vn[i]))
        for de in range(degree - 1):
            vn2 = [[] for _ in range(len(vn))]
            for i in range(len(vn)):
                for item in vn[i]:
                    vn2[i].extend(vn[item])

            for i in range(len(vn2)):
                vn2[i] = list(set(vn2[i]))
            vn = vn2
        max_len = 0
        len_matrix = []
        for i in range(len(vn)):
            vn[i] = list(set(vn[i]))
            len_matrix.append(len(vn[i]))

        idxs = np.argsort(len_matrix)[::-1][:len(len_matrix) // 1]
        max_len = len_matrix[idxs[0]]
        print("max_len: ", max_len)

        vns = np.zeros((len(idxs), max_len))
        # for i in range( len(vn)):
        for i0, i in enumerate(idxs):
            for j in range(max_len):
                if j < len(vn[i]):
                    vns[i0, j] = vn[i][j]
                else:
                    vns[i0, j] = i
        return vns, idxs

    def read_cali(self, path):
        file1 = open(path, 'r')
        Lines = file1.readlines()
        for line in Lines:
            if 'R:' in line:
                rotation = line.split('R:')[-1]
            if 'T:' in line:
                translation = line.split('T:')[-1]
        tmp_r = rotation.split(' ')
        tmp_r.pop(0)
        tmp_r[-1] = tmp_r[-1].split('\n')[0]
        # print(tmp_r)
        rota_matrix = []

        for i in range(3):
            tt = []
            for j in range(3):
                tt.append(float(tmp_r[i * 3 + j]))
            rota_matrix.append(tt)
        self.rota_matrix = np.array(rota_matrix)
        tmp_t = translation.split(' ')
        tmp_t.pop(0)
        tmp_t[-1] = tmp_t[-1].split('\n')[0]
        # print(tmp_t)
        trans_matrix = [float(tmp_t[i]) for i in range(3)]
        self.trans_matrix = np.array(trans_matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-obj', '--obj', dest='object', default="./object/cube.ply")
    parser.add_argument('-obj_save' ,'--obj_save', dest='object_save', default="./object/obj_save")
    parser.add_argument('-lidar', '--lidar', dest='lidar', default='./data/lidar.bin')
    parser.add_argument('-cam', '--cam', dest='cam', default='./data/cam.png')
    parser.add_argument('-cali', '--cali', dest='cali', default='./data/cali.txt')
    parser.add_argument('-o', '--opt', dest='opt', default="Adam")  # pgd
    parser.add_argument('-e', '--epsilon', dest='epsilon', type=float, default=0.2)
    parser.add_argument('-it', '--iteration', dest='iteration', type=int, default=500)
    args = parser.parse_args()

    obj = attack_msf(args)
    # obj.load_model_()
    obj.load_LiDAR_model()
    obj.read_cali(args.cali)
    obj.load_mesh(args.object, 0.75)
    # obj.load_bg(args.cam)
     # obj.init_render()
    obj.load_pc_mesh(args.lidar)
    obj.rendering_img(args.lidar)
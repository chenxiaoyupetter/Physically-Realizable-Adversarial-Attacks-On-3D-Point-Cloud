import torch
import sys
import torch.nn.functional as F


def square_distance(src, dst):  # 一个n点点云和一个m点点云，最后生成n×m个欧式距离平方

    N, _ = src.shape
    M, _ = dst.shape
    # src *= (-2)
    # dst *= (-2)
    dist = torch.mm(src, dst.T)
    dist *= (-2)
    # src /= (-2)
    # dst /= (-2)
    dist += torch.sum(src ** 2, -1).view(N, 1)
    dist += torch.sum(dst ** 2, -1).view(1, M)

    return dist


def chamfer_distance(points, adv_points):

    adv_dist = square_distance(points, adv_points)
    min_dist = adv_dist.min(-1)[0]
    ch_dist = min_dist.mean(-1)

    return ch_dist


def hausdorff_distance(points, adv_points):

    adv_dist = square_distance(points, adv_points)
    min_dist = adv_dist.min(-1)[0]
    ha_dist = min_dist.max(-1)[0]

    return ha_dist

#'instance_pt', 'category_score', 'confidence_score','height_pt', 'heading_pt', 'class_score']
def lossRenderAttack(outputPytorch, vertex, vertex_og, face, mu):
    #a = torch.sum(torch.eq(vertex, vertex_og))
    #b = a/19854
    face = face.long()
    loss_center = (vertex.mean(0) - vertex_og.mean(0)) ** 2
    loss_center = loss_center.sum()

    inv_res_x = 0.5 * float(512) / 60

    x_var = vertex[:, 0]
    y_var = vertex[:, 1]

    fx = torch.floor(x_var * 512.0 / 120 + 512.0 / 2).long()
    fy = torch.floor(y_var * 512.0 / 120 + 512.0 / 2).long()

    mask = torch.zeros((512, 512)).cuda().index_put((fx, fy), torch.ones(fx.shape).cuda())
    mask1 = torch.where(torch.mul(mask, outputPytorch[1]) >= 0.5, torch.ones_like(mask), torch.zeros_like(mask))

    loss_object = torch.sum(torch.mul(mask1, outputPytorch[2])) / (torch.sum(mask1 + 0.000000001))
    #loss1 = torch.sum(torch.mul(mask, outputPytorch[1]))
    # class_probs = (torch.mul(mask1, outputPytorch[5]).sum(2).sum(2) / torch.sum(mask1 + 0.000000001))[0]
    # loss_class = class_probs[0] - class_probs[1]
    # print class_probs

    # loss_distance_1 = torch.sum(torch.sqrt(torch.pow(vertex[:, 0] - vertex_og[:, 0] + sys.float_info.epsilon, 2) +
    #                                        torch.pow(vertex[:, 1] - vertex_og[:, 1] + sys.float_info.epsilon, 2) +
    #                                        torch.pow(vertex[:, 2] - vertex_og[:, 2] + sys.float_info.epsilon,
    #                                                  2)))  # + sys.float_info.epsilon to prevent zero gradient
    # print(vertex_og.shape)
    zmin = vertex_og[:, 2].min()
    loss_z = (vertex[:, 2].min() - zmin) ** 2

    def calc_dis(vertex):
        meshes = torch.nn.functional.embedding(face, vertex)
        edge_1 = meshes[:, 1] - meshes[:, 0]
        edge_2 = meshes[:, 2] - meshes[:, 0]
        edge_3 = meshes[:, 1] - meshes[:, 2]

        dis = torch.stack([torch.sqrt(torch.pow(edge_1[:, 0], 2) +
                                      torch.pow(edge_1[:, 1], 2) +
                                      torch.pow(edge_1[:, 2], 2)), torch.sqrt(torch.pow(edge_2[:, 0], 2) +
                                                                              torch.pow(edge_2[:, 1], 2) +
                                                                              torch.pow(edge_2[:, 2], 2)),
                           torch.sqrt(torch.pow(edge_3[:, 0], 2) +
                                      torch.pow(edge_3[:, 1], 2) +
                                      torch.pow(edge_3[:, 2], 2))], dim=1)

        return dis

    dis = calc_dis(vertex)
    dis_og = calc_dis(vertex_og)

    loss_distance_2 = torch.sum((dis - dis_og) ** 2)
    #loss_distance_chamfer = chamfer_distance(dis, dis_og)
    #loss_distance_hausdorff = hausdorff_distance(dis, dis_og)

    beta = 0.5
    labda = 100.0

    # loss_distance = loss_distance_1 + beta * loss_distance_2
    # loss_distance = loss_distance_2
    # loss = mu * loss_distance + loss_object
    # loss_distance_ = F.relu(loss_distance - 1.0)
    # loss_distance = 0.0
    loss = 5 * loss_object + beta * loss_center + loss_z + loss_distance_2*0.6
    #print("b",b)
    #loss =  5.0 *loss_object
    print("loss_object",loss_object.detach().cpu(),
          "loss_distance_2",loss_distance_2.detach().cpu(),
          "beta * loss_center",beta * loss_center.detach().cpu(),
          "loss_z",loss_z.detach().cpu())
    # loss_class += mu*loss_distance
    # loss_distance = loss_distance

    # return loss, loss_object, loss_distance,
    return loss, loss_object#, loss_distance, loss_center, loss_z
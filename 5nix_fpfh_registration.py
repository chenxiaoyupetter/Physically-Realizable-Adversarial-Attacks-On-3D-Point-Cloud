import pointclouds_preprocess as pp
import open3d as o3d 
import numpy as np
import copy
import os


target = pp.prepare_dataset("../../data/5nix_compound/5nix_pocket_pc.ply")
source = pp.prepare_dataset("../../data/5nix_compound/5nix_ligand_pc.ply")
source_model = pp.prepare_dataset("../../data/5nix_compound/model_ligand.ply")


voxel_size = 0.5


source.downsampling(voxel_size)
target.downsampling(voxel_size)
source.estimate_normal(voxel_size * 2)
target.estimate_normal(voxel_size * 2)

os.makedirs("result/rmse_data", exist_ok=True)


for fpfh_radius_size in range(5, 51, 5):

    source.calculate_fpfh(fpfh_radius_size)
    target.calculate_fpfh(fpfh_radius_size)

    # --- 推定したリガンドのポーズと真のポーズとのＲＳＭＥをファイルに書き込み
    f = open('result/rmse_data/result_rmse_radius%d.txt' % fpfh_radius_size, mode="w")

    # --- 結果画像を格納するためのフォルダーを作成
    os.makedirs("result/fpfh_radius_%d" %fpfh_radius_size, exist_ok=True)

    # --- 進捗表示
    print(":: Iteration  ")
    for i in range(100):
        # --- ＦＰＦＨ特徴量をマッチングさせる
        result = pp.execute_global_registration(source.pcd_down, 
                                                target.pcd_down, 
                                                source.pcd_fpfh, 
                                                target.pcd_fpfh, 
                                                voxel_size)

        # --- 進捗表示
        # print("-", end='')
        if (i % 10) == 0:
            print("%d" % i)

        # print(":: ", result)

        # --- 画像保存（色替え、向き変え）
        target.pcd.paint_uniform_color([0, 0.651, 0.929])
        source.pcd.paint_uniform_color([1, 0.706, 0])
        source.pcd.transform(result.transformation)

        # --- 推定したリガンドのポーズと真のポーズとのＲＳＭＥ
        rmse = np.average(source.pcd.compute_point_cloud_distance(source_model.pcd))
        f.write(str(rmse)+"\n") 

        if rmse < 4.0:
            # --- 画像保存（ウィンドウ出さずに保存）
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(source.pcd)
            vis.add_geometry(target.pcd)
            # - 結果画像の向きを変える試み
            # ctr = vis.get_view_control()
            # ctr.translate(90.0, 0.0)
            # -
            vis.update_geometry()
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image("result/fpfh_radius_%d" %fpfh_radius_size + "/test_pic%d.jpg" % i)
            vis.destroy_window()

        # --- 向きを元に戻す
        source.pcd.transform(np.linalg.inv(result.transformation))

    # --- 進捗表示
    print()

f.close()
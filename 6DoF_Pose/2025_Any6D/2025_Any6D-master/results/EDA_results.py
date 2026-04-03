import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import os.path as osp

# ==========================================
# 1. 핵심 유틸리티 함수 (Data Loaders)
# ==========================================

def load_vertices(path, downsample_rate=20):
    """ OBJ 파일에서 Vertex 좌표만 추출 """   
    vertices = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])]) # (x,y,z)
    
    #-- 시각화 속도를 위해 샘플링 (10개 중 1개)
    return np.array(vertices)[::downsample_rate] # return (N, 3)

def project_points(K, R, t, points_3d):
    """ 
    [수학적 검증용]

    3D 점들에 회전(R)과 이동(t)을 적용한 후, 카메라 행렬(K)로 투영 
    """
    # 1. Pose 적용 (World -> Camera) 카메라 좌표계로 변환 
    # (N, 3) x (3, 3).T + (3,)
    points_cam = (R @ points_3d.T).T + t.reshape(1, 3)
    
    # 2. 2D 투영 (Camera -> Image)
    points_2d_homo = (K @ points_cam.T).T
    
    # 3. 정규화 (Z로 나누기) [u, v] = [x/z,y/z]
    valid_mask = points_2d_homo[:, 2] > 0.001 # 카메라 앞쪽 점만
    points_2d = points_2d_homo[valid_mask, :2] / points_2d_homo[valid_mask, 2:3]
    
    return points_2d

def project_direct(K, points_3d):
    """ 
    [결과물 검증용]
    이미 변환된(Transformed) 점들을 바로 투영 (카메라 앞쪽 점만 사용)
    """
    points_2d_homo = (K @ points_3d.T).T
    valid_mask = points_2d_homo[:, 2] > 0.001  # Z(깊이가 0보다 큰 점만 사용(= 카메라 뒤에 있는 점 무시)
    points_2d = points_2d_homo[valid_mask, :2] / points_2d_homo[valid_mask, 2:3] # 정규화(투영); Homogeneous에서 실제 이미지 좌표로 변환 
                                                                                 # (u,v) = (x/z, y/z)
    return points_2d

# ==========================================
# 2. 메인 실행 로직
# ==========================================

def main():
    # --- [Step 1] 경로 설정 (질문자님 코드 반영) ---
    root = osp.dirname(osp.abspath(__file__))
    result_dir = osp.join(root, "demo_mustard") 
    

    # 파일 경로 정의
    img_path = osp.join(result_dir, 'color.png')
    
    
    # 원본 모델 (데모 데이터 폴더에 있는 것)
    original_mesh_path = osp.join(result_dir, 'refine_init_mesh_demo.obj') 
    
    # 결과물들
    final_mesh_path = osp.join(result_dir, 'final_mesh_demo.obj')
    pred_pose_path = osp.join(result_dir, 'demo_mustard_initial_pose.txt')
    gt_pose_path = osp.join(result_dir, 'demo_mustard_gt_pose.txt') # from demo_data/labels.npz
    k_path = osp.join(result_dir, 'K.txt') 

    # --- [Step 2] 데이터 로드 및 검증 ---
    
    # 1. 이미지
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # 2. 카메라 행렬 (K)
    # K는 카메라 내부 파라미터(intrinsic matrix)입니다.
    # 예시:
    # [[fx,  0, cx],
    #  [ 0, fy, cy],
    #  [ 0,  0,  1]]
    K = np.loadtxt(k_path)  # for 3D -> 2D projection

    # 3. 3D 모델 PointClouds (원본 vs 결과물)
    verts_original = load_vertices(original_mesh_path) # 원본 (0,0,0 중심)
    verts_final = load_vertices(final_mesh_path)       # 결과물 (이미 이동됨)

    # 4. 포즈 파일
    pred_pose = np.loadtxt(pred_pose_path) # Any6D 예측값
    gt_pose = np.loadtxt(gt_pose_path)     # 정답값


    # --- [Step 3] 시각화 및 검증 ---
    vis_img = img.copy()

    print("\n🔍 검증 시작...")

    # (A) Ground Truth (빨간색)
    # -> 원본 메쉬 좌표계 이슈로 인해 실제 물체와 어긋날 수 있음 (정상)
    if gt_pose is not None and len(verts_original) > 0:
        pts_gt = project_points(K, gt_pose[:3,:3], gt_pose[:3,3], verts_original)
        for p in pts_gt:
            if 0 <= p[0] < w and 0 <= p[1] < h:
                cv2.circle(vis_img, (int(p[0]), int(p[1])), 1, (255, 0, 0), -1) 
        print("   Checking GT... [Red Dots]")

    # (B) 수학적 계산 검증 (파란색, 큰 점)
    # -> Logic: 원본(refine_init_mesh_demo.obj) * 예측포즈(pred_pose.txt)
    if len(verts_original) > 0:
        pts_calc = project_points(K, pred_pose[:3,:3], pred_pose[:3,3], verts_original)
        for p in pts_calc:
            if 0 <= p[0] < w and 0 <= p[1] < h:
                cv2.circle(vis_img, (int(p[0]), int(p[1])), 1, (0, 0, 255), -1) # Blue
        print("   Checking Pose Calculation... [Blue Dots]")

    # (C) 결과 메쉬 검증 (초록색, 작은 점)
    # -> Logic: 결과파일(final_mesh...obj) 직접 투영
    if len(verts_final) > 0:
        pts_direct = project_direct(K, verts_final)
        for p in pts_direct:
            if 0 <= p[0] < w and 0 <= p[1] < h:
                cv2.circle(vis_img, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1) # Green
        print("   Checking Final Mesh File... [Green Dots]")

    # --- [Step 4] 결과 저장 ---
    plt.figure(figsize=(12, 8))
    plt.imshow(vis_img)
    plt.title("Verification: Blue(Calculation) & Green(Final Mesh) should OVERLAP.\nRed is GT (Reference).")
    plt.axis('off')
    
    save_name = 'verified_result.png'
    plt.savefig(save_name)
    print(f"\n✅ 검증 완료! '{save_name}' 이미지를 확인하세요.")
    print("   👉 파란색 점 위에 초록색 점이 덮어씌워져 있다면, 모든 데이터가 완벽하게 일치하는 것입니다.")

if __name__ == "__main__":
    main()
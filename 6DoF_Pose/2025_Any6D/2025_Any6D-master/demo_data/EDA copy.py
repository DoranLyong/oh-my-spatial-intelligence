import os.path as osp

import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_obj_vertices(path, downsample_rate=10):
    """
    .obj 파일에서 Vertex(점) 정보만 읽어옵니다.
    빠른 시각화를 위해 점 개수를 줄여서(downsample) 가져옵니다.
    """
    vertices = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                # v x y z r g b 형식이거나 v x y z 형식
                parts = line.strip().split()
                # x, y, z 좌표만 float으로 변환
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    vertices = np.array(vertices)
    return vertices[::downsample_rate] # 점 개수 줄이기

def get_camera_intrinsic_from_yaml(yaml_path):
    """
    YAML 파일을 파싱하여 Color 카메라의 Intrinsic Matrix(K)를 만듭니다.
    외부 라이브러리 없이 텍스트 파싱으로 처리합니다.
    """
    params = {}
    with open(yaml_path, 'r') as f:
        lines = f.readlines()
        
    # 'color:' 섹션 찾기
    in_color_section = False
    for line in lines:
        line = line.strip()
        if line.startswith('color:'):
            in_color_section = True
            continue
        if line.startswith('depth:'):
            in_color_section = False
            
        if in_color_section:
            if line.startswith('fx:'): params['fx'] = float(line.split(':')[1])
            if line.startswith('fy:'): params['fy'] = float(line.split(':')[1])
            if line.startswith('ppx:'): params['ppx'] = float(line.split(':')[1])
            if line.startswith('ppy:'): params['ppy'] = float(line.split(':')[1])
            
    # Intrinsic Matrix K 구성
    # [[fx, 0, ppx],
    #  [0, fy, ppy],
    #  [0,  0,   1]]
    K = np.array([
        [params['fx'], 0, params['ppx']],
        [0, params['fy'], params['ppy']],
        [0, 0, 1]
    ])
    return K

def project_points(K, R, t, points_3d):
    """
    3D 점들을 2D 이미지 평면으로 투영(Projection)합니다.
    공식: p_2d = K * (R * p_3d + t)
    """
    # 3D 점들을 회전 및 이동 변환 (Camera Coordinate로 변환)
    # points_3d: (N, 3), R: (3, 3), t: (3,)
    points_cam = (R @ points_3d.T).T + t # (N, 3)
    
    # 2D 픽셀 좌표로 투영 (Homogeneous division)
    points_2d_homo = (K @ points_cam.T).T # (N, 3)
    points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3] # z로 나누기
    
    return points_2d

def visualize_analysis():
    root = osp.dirname(osp.abspath(__file__))

    # 1. 데이터 로드
    img_rgb = cv2.cvtColor(cv2.imread(osp.join(root,'color.png')), cv2.COLOR_BGR2RGB)        
    depth = cv2.imread(osp.join(root, 'depth.png'), cv2.IMREAD_UNCHANGED)
    
    labels = np.load(osp.join(root, 'labels.npz'))
    # labels.npz 구조: 'pose_y', 'pose_m', 'joint_2d', 'seg' ...
    
    K = get_camera_intrinsic_from_yaml(osp.join(root, '836212060125_640x480.yml'))
    mustard_verts = load_obj_vertices(osp.join(root, 'mustard.obj'), downsample_rate=20)

    # ---------------------------------------------------------
    # 시각화 1: RGB 이미지 위에 Hand Joint와 Object Pose 그리기
    # ---------------------------------------------------------
    vis_img = img_rgb.copy()
    
    # (A) Hand Joints 시각화 (녹색 점)
    if 'joint_2d' in labels:
        joints = labels['joint_2d'][0] # (21, 2)
        for j in joints:
            cv2.circle(vis_img, (int(j[0]), int(j[1])), 5, (0, 255, 0), -1)
        
        # 간단한 손가락 연결 (Skeleton) - 주요 관절만 연결 예시
        # 보통 MANO 순서는: 0(손목), 1-4(검지), 5-8(중지)... 등등이나 여기선 점만 표시
        
    # (B) Object (Mustard) Pose 시각화 (빨간색 점)
    # pose_y는 (4, 3, 4) 형태로 4개의 물체 포즈가 들어있음.
    # Mustard 병에 해당하는 포즈를 찾아야 함. 보통 YCB 데이터셋 ID와 매칭됨.
    # 여기서는 4개 포즈를 모두 그려보고, 영상 내 물체와 겹치는 것을 확인합니다.
    poses = labels['pose_y'] 
    
    colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)] # R, B, Y, C
    
    # 모든 포즈를 투영해 봅니다.
    for i, pose in enumerate(poses):
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        # 카메라 앞(z > 0)에 있고, 이미지 범위 내에 들어오는 포즈만 그리기 위한 간단한 체크
        if t[2] > 0: 
            pts_2d = project_points(K, R, t, mustard_verts)
            
            # 이미지 밖으로 나가는 점이 너무 많으면 건너뛰기 (유효성 검사)
            valid_pts = (pts_2d[:,0] >= 0) & (pts_2d[:,0] < 640) & \
                        (pts_2d[:,1] >= 0) & (pts_2d[:,1] < 480)
            
            if np.sum(valid_pts) > 10: # 유효한 점이 좀 있다면 그림
                # 해당 포즈의 점들 그리기
                for pt in pts_2d[valid_pts]:
                    cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 1, colors[i % 4], -1)

    # ---------------------------------------------------------
    # 시각화 2: Depth Map (컬러맵 적용)
    # ---------------------------------------------------------
    # Depth 값을 0~255로 정규화하여 시각화
    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = np.uint8(depth_vis)
    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)

    # ---------------------------------------------------------
    # 시각화 3: Segmentation Mask
    # ---------------------------------------------------------
    seg_mask = labels['seg']
    # 마스크 ID 별로 랜덤 컬러 적용
    unique_ids = np.unique(seg_mask)
    seg_vis = np.zeros_like(img_rgb)
    
    for uid in unique_ids:
        if uid == 0: continue # 배경 무시
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        seg_vis[seg_mask == uid] = color

    # ---------------------------------------------------------
    # 결과 합치기 및 저장
    # ---------------------------------------------------------
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("RGB + Hand(Green) + Object(Red/Dots)")
    plt.imshow(vis_img)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Depth Map (Colormap)")
    plt.imshow(depth_colormap)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Segmentation Mask")
    plt.imshow(seg_vis)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('analysis_result.png')
    print("분석 결과 이미지 'analysis_result.png'가 저장되었습니다.")

if __name__ == "__main__":
    visualize_analysis()
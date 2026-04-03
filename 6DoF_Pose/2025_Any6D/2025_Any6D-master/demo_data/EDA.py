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

def project_and_draw(img, K, R, t, points_3d, color=(0, 255, 0)):
    """ 3D 점들을 이미지에 투영하여 그립니다. """
    points_cam = (R @ points_3d.T).T + t
    points_2d_homo = (K @ points_cam.T).T
    points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
    
    h, w, _ = img.shape
    valid = (points_2d[:,0] >= 0) & (points_2d[:,0] < w) & \
            (points_2d[:,1] >= 0) & (points_2d[:,1] < h)
            
    vis_img = img.copy()
    for pt in points_2d[valid]:
        cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 1, color, -1)
    return vis_img

def visualize_analysis():
    root = osp.dirname(osp.abspath(__file__))

    # 1. 데이터 로드
    img_rgb = cv2.cvtColor(cv2.imread(osp.join(root,'color.png')), cv2.COLOR_BGR2RGB)        
    depth = cv2.imread(osp.join(root, 'depth.png'), cv2.IMREAD_UNCHANGED)    
    labels = np.load(osp.join(root, 'labels.npz'))
    mustard_verts = load_obj_vertices(osp.join(root, 'mustard.obj'), downsample_rate=20)
    K = get_camera_intrinsic_from_yaml(osp.join(root, '836212060125_640x480.yml'))
    
    
    
    # 2. [수정됨] 머스타드 병 찾기 (가장 큰 객체 ID 추출)
    seg = labels['seg']
    ids, counts = np.unique(seg, return_counts=True)
    # 배경(0)과 무효값(255) 제외하고 가장 큰 영역을 가진 ID 찾기
    valid_mask = (ids != 0) & (ids != 255)
    target_id = ids[valid_mask][np.argmax(counts[valid_mask])]
    
    # 3. [수정됨] 해당 ID의 Pose Index 찾기
    # pose_y는 seg에 등장하는 객체 순서대로 저장되어 있다고 가정
    obj_indices = ids[valid_mask] # 예: [5, 6, 13, 18]
    pose_idx = np.where(obj_indices == target_id)[0][0] # ID 5는 0번째 인덱스
    
    target_pose = labels['pose_y'][pose_idx] # (3, 4) or (4, 4)
    print(f"Target Object ID: {target_id} (Mustard Bottle), Pose Index: {pose_idx}")

    # 4. 시각화
    # 주의: labels.npz의 pose는 YCB 원본 기준, mustard.obj는 데모용(Centered) 기준이라
    # 시각적으로는 어긋나 보일 수 있음 (정상)
    vis_img = project_and_draw(img_rgb, K, 
                               target_pose[:3, :3], target_pose[:3, 3], 
                               mustard_verts, color=(255, 0, 0))

    # 결과 저장
    plt.figure(figsize=(10, 8))
    plt.imshow(vis_img)
    plt.title(f"Mustard Bottle Visualization (ID: {target_id})\n*Note: Misalignment due to Mesh Coordinate mismatch is expected")
    plt.axis('off')
    plt.savefig('mustard_analysis.png')
    print("분석 완료: 'mustard_analysis.png' 저장됨")

if __name__ == "__main__":
    visualize_analysis()
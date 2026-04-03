import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_cube(ax, center, size=1.0, color='b', label=''):
    """ 큐브(육면체)를 그리는 함수 """
    r = [-size/2, size/2]
    # 큐브의 8개 꼭짓점 생성
    pts = np.array([[x, y, z] for x in r for y in r for z in r])
    pts += center # 중심 이동
    
    # 꼭짓점 그리기
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=color, s=50, label=label)
    
    # 모서리 연결 (시각화용)
    for i in range(8):
        for j in range(i+1, 8):
            if np.sum(np.abs(pts[i]-pts[j])) == size: # 인접한 꼭짓점만 연결
                ax.plot([pts[i,0], pts[j,0]], [pts[i,1], pts[j,1]], [pts[i,2], pts[j,2]], c=color, alpha=0.5)
    return pts

def get_rotation_z(theta):
    """ Z축 기준 회전 행렬 """
    rad = np.radians(theta)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def visualize_canonical_concept():
    fig = plt.figure(figsize=(12, 6))
    
    # ------------------------------------------------
    # 상황 1: Canonical Space (중심이 0,0,0)
    # ------------------------------------------------
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_title("1. Canonical Space (Center=0,0,0)\nRotation applies 'In-Place'")
    ax1.set_xlim(-2, 2); ax1.set_ylim(-2, 2); ax1.set_zlim(-2, 2)
    ax1.scatter([0], [0], [0], c='k', marker='x', s=100, label='Origin(0,0,0)') # 원점 표시

    # (A) 원래 물체 (파랑)
    center_canonical = np.array([0, 0, 0])
    pts_can = plot_cube(ax1, center_canonical, color='blue', label='Original')

    # (B) 45도 회전 (빨강)
    R = get_rotation_z(45)
    pts_can_rotated = (R @ pts_can.T).T
    
    # 회전된 점들 그리기
    ax1.scatter(pts_can_rotated[:,0], pts_can_rotated[:,1], pts_can_rotated[:,2], c='red', s=50, label='Rotated 45°')
    # 모서리 연결
    for i in range(8):
        for j in range(i+1, 8):
            if np.abs(np.linalg.norm(pts_can[i]-pts_can[j]) - 1.0) < 0.1:
                ax1.plot([pts_can_rotated[i,0], pts_can_rotated[j,0]], 
                         [pts_can_rotated[i,1], pts_can_rotated[j,1]], 
                         [pts_can_rotated[i,2], pts_can_rotated[j,2]], c='red', alpha=0.5)

    ax1.legend()

    # ------------------------------------------------
    # 상황 2: Non-Canonical (중심이 2,2,0)
    # ------------------------------------------------
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title("2. World/Offset Space (Center=2,2,0)\nRotation causes 'Orbit' (Position Change!)")
    ax2.set_xlim(-1, 4); ax2.set_ylim(-1, 4); ax2.set_zlim(-2, 2)
    ax2.scatter([0], [0], [0], c='k', marker='x', s=100, label='Origin(0,0,0)')

    # (A) 원래 물체 (파랑) - 원점에서 떨어져 있음
    center_world = np.array([2, 0, 0])
    pts_world = plot_cube(ax2, center_world, color='blue', label='Original')

    # (B) 45도 회전 (빨강) - 똑같은 행렬 R을 곱함
    pts_world_rotated = (R @ pts_world.T).T
    
    # 회전된 점들 그리기
    ax2.scatter(pts_world_rotated[:,0], pts_world_rotated[:,1], pts_world_rotated[:,2], c='red', s=50, label='Rotated 45°')
     # 모서리 연결
    for i in range(8):
        for j in range(i+1, 8):
            if np.abs(np.linalg.norm(pts_world[i]-pts_world[j]) - 1.0) < 0.1:
                ax2.plot([pts_world_rotated[i,0], pts_world_rotated[j,0]], 
                         [pts_world_rotated[i,1], pts_world_rotated[j,1]], 
                         [pts_world_rotated[i,2], pts_world_rotated[j,2]], c='red', alpha=0.5)
    
    # 이동 경로 화살표
    ax2.quiver(2, 0, 0, pts_world_rotated.mean(0)[0]-2, pts_world_rotated.mean(0)[1], 0, color='g', linestyle='--', arrow_length_ratio=0.1)
    ax2.text(1, 1, 0, "Orbit Movement", color='g')

    ax2.legend()
    plt.tight_layout()
    plt.savefig('canonical_explanation.png')
    print("Canonical Space 설명 이미지가 저장되었습니다.")

if __name__ == "__main__":
    visualize_canonical_concept()
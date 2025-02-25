#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('TkAgg')  # TkAgg 백엔드 사용
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
import matplotlib.ticker as ticker

# 파일 경로 설정
file_path1 = "/home/teamai/data/finalplot/diff1000ep240/just/v_new_data.nc"
file_path2 = "/home/teamai/data/finalplot/diff1000ep240/shift_np.pi_2/v_new_data.nc"

# 분석에 사용할 시간 인덱스 (예: 20000번째 시간)
time_index = 20000

# 파일에서 u, v, flow_mask를 읽고 vorticity 계산 함수
def process_file(file_path, time_index):
    ds = Dataset(file_path, 'r')
    u = ds.variables['u_new'][time_index, :, :]
    v = ds.variables['v_new'][time_index, :, :]
    flow_mask = ds.variables['flow_mask'][time_index, :, :]
    ds.close()
    # 중앙 차분법을 사용한 vorticity 계산: ζ = ∂v/∂x - ∂u/∂y
    dv_dx = np.gradient(v, axis=1)
    du_dy = np.gradient(u, axis=0)
    vorticity = dv_dx - du_dy
    return u, v, flow_mask, vorticity

# 각 파일 처리
u1, v1, flow_mask1, vort1 = process_file(file_path1, time_index)
u2, v2, flow_mask2, vort2 = process_file(file_path2, time_index)

# flow_mask 처리: 내부 영역만 (경계 제거)
def process_flow_mask(flow_mask):
    mask = (flow_mask == 0)
    interior_mask = binary_erosion(mask, structure=np.ones((3,3)))
    mask_region = np.where(interior_mask, 1, np.nan)
    return mask_region

mask_region1 = process_flow_mask(flow_mask1)
mask_region2 = process_flow_mask(flow_mask2)

# 서브플롯 생성 (1행 2열)
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)

# 각 서브플롯에 대해 vorticity contour 및 streamlines 그리기
idx=0
for ax, vort, u, v, mask_region, title in zip(
        axs, [vort1, vort2], [u1, u2], [v1, v2], [mask_region1, mask_region2],
        ['Vorticity (Just)', 'Vorticity (Shifted: $\pi/2$)']):
    
    # vorticity contour 플롯: 범위는 예시로 -0.1 ~ 0.1 사용 (필요시 조정)
    levels = np.linspace(-0.1, 0.1, 21)
    cf = ax.contourf(vort, levels=levels, cmap='RdBu_r', extend='both')
    
    # flow_mask 내부 영역 회색 오버레이 (경계선 없이)
    height, width = vort.shape
    ax.imshow(mask_region, cmap='gray', alpha=0.9, interpolation='none',
              origin='lower', extent=[0, width, 0, height], zorder=10)
    
    # x, y 좌표 생성 및 streamlines 그리기
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    ax.streamplot(X, Y, u, v, color='k', density=1.5, linewidth=1)
    
    ax.set_title(title, fontsize=16, pad=12)
    if idx==1:
        ax.set_xlabel('X (m)', fontsize=14, labelpad=10)
    else:
        idx=idx+1
    ax.set_ylabel('Y (m)', fontsize=14, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_aspect('equal', adjustable='box')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

# 공통 y축 범위 설정
axs[0].set_ylim(0, height)
axs[1].set_ylim(0, height)

# 전체 색상바를 위한 영역 생성
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
cbar = fig.colorbar(cf, cax=cbar_ax)
cbar.set_label('Vorticity (1/s)', fontsize=14)
cbar.ax.tick_params(labelsize=12)

# 전체 레이아웃 조정 및 파일 저장 (화면에 표시하지 않음)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig('vorticity_subplots.png', dpi=300, bbox_inches='tight')
plt.close()


import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 필드 크기
width = 300
height = 100

# 20x20 박스 (검정색) 중심: (height//2, 3*width//4)
center_y = height // 2        # 50
center_x = 3 * width // 4     # 225
box_size = 20
half_box = box_size // 2
box_left = center_x - half_box  # 225 - 10 = 215
box_bottom = center_y - half_box  # 50 - 10 = 40

# 분석 영역 (analysis area)
spongebuf = 25
beforebox = 214
analysis_top = 3 + spongebuf       # 28
analysis_bottom = height - (spongebuf + 3)  # 100 - 28 = 72
analysis_left = 5
analysis_right = beforebox         # 214

analysis_width = analysis_right - analysis_left  # 214 - 5 = 209
analysis_height = analysis_bottom - analysis_top # 72 - 28 = 44

# 관측점 (observation points)
x_obs = [213, 213, 208, 208, 195, 195, 190, 190, 185, 185]
y_obs = [45, 55, 40, 60, 35, 65, 35, 65, 35, 65]

# Figure 생성
fig, ax = plt.subplots(figsize=(10, 4))

# 축 한계 설정 (원점: 왼쪽 아래)
ax.set_xlim(0, width)
ax.set_ylim(0, height)

# 분석 영역 사각형 (초록색 실선)
analysis_rect = patches.Rectangle((analysis_left, analysis_top), analysis_width, analysis_height,
                                  linewidth=2, edgecolor='black', facecolor='none', linestyle='-')
ax.add_patch(analysis_rect)
# 분석 영역 내부 중앙에 텍스트 라벨 추가 (분석 영역 index 포함)
ax.text(analysis_left + analysis_width/2, analysis_top + analysis_height/2, 'Analysis Area',
        color='black', fontsize=12, fontweight='bold',
        ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# 20x20 박스 (검정색 채움)
box_rect = patches.Rectangle((box_left, box_bottom), box_size, box_size,
                             linewidth=1, edgecolor='black', facecolor='black')
ax.add_patch(box_rect)

# 관측점 표시 (자주색 원형 마커, 검정 테두리)
ax.plot(x_obs, y_obs, 'o', color='magenta', markersize=8, markeredgecolor='black', label='Observation Points')

# 학술용 제목, 축 라벨, 범례 등
ax.set_title('Analysis Area, Box, and Observation Points', fontsize=16)
ax.set_xlabel('X (m)', fontsize=14)
ax.set_ylabel('Y (m)', fontsize=14)
ax.legend(fontsize=12, loc='upper right', frameon=True)
ax.grid(True, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('field_analysis_observation.png', dpi=300, bbox_inches='tight')
plt.show()


loadindex=130
loaddatetime=
expname="diff1000ep$loadindex"

# 기준 경로
BASE_DIR="/home/teamai/data/$expname"
JUST_DIR="$BASE_DIR/just"
SHIFT_DIR="$BASE_DIR/shift"

# 기준 경로 확인 및 생성
if [ ! -d "$BASE_DIR" ]; then
    echo "Creating $BASE_DIR and its subdirectories..."
    mkdir -p "$JUST_DIR" "$SHIFT_DIR"
    echo "Directories created: $BASE_DIR, $JUST_DIR, $SHIFT_DIR"
else
    echo "$BASE_DIR already exists. No directories created."
fi

# 원본 파일
just_FILE="../fluid_2d/just_5p.py"
# 수정된 파일 저장 경로
justexp_FILE="../fluid_2d/just_now.py"
# expname을 diff1000ep50으로 수정하고 결과를 새로운 파일에 저장
sed "s/expname/$expname/" $just_FILE > $justexp_FILE
# 수정된 파일 실행
CUDA_VISIBLE_DEVICES=1  python $justexp_FILE --mu=1.7894e-05 --rho=1.225 --dt=4 --load_index=$loadindex --load_date_time=2024-11-19\ 18:26:48 --average_sequence_length=30000 --n_forcing=10


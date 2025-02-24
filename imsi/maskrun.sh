#!/bin/bash

loadindex=7
shiftphase="np.pi/2"
shiftphase_safe=$(echo "$shiftphase" | sed 's/\//_/g')
expname="diff100ep$loadindex"

# 기준 경로
BASE_DIR="/home/teamai/data/$expname"
JUST_DIR="$BASE_DIR/just"
SHIFT_DIR="$BASE_DIR/shift_$shiftphase_safe"

# 기준 경로 확인 및 생성
if [ ! -d "$BASE_DIR" ]; then
    echo "Creating $BASE_DIR and its subdirectories..."
    mkdir -p "$JUST_DIR" 
    echo "Directories created: $BASE_DIR, $JUST_DIR"
else
    echo "$BASE_DIR already exists. No directories created."
fi
# 기준 경로 확인 및 생성
if [ ! -d "$SHIFT_DIR" ]; then
    echo "Creating $SHIFT_DIR and its subdirectories..."
    mkdir -p "$SHIFT_DIR" 
    echo "Directories created: $SHIFT_DIR"
else
    echo "$SHIFT_DIR already exists. No directories created."
fi

# 원본 파일
just_FILE="just_mask.py"
# 수정된 파일 저장 경로
justexp_FILE="just_now.py"
# expname을 diff1000ep50으로 수정하고 결과를 새로운 파일에 저장
sed "s/expname/$expname/" $just_FILE > $justexp_FILE
# 수정된 파일 실행
#CUDA_VISIBLE_DEVICES=1  python $justexp_FILE --mu=0.1 --rho=4.0 --dt=4 --load_index=$loadindex --load_date_time=2025-01-09\ 15:59:21 --average_sequence_length=30000 --n_forcing=10
#CUDA_VISIBLE_DEVICES=1  python $justexp_FILE --mu=0.1 --rho=4.0 --dt=4 --load_index=$loadindex --load_date_time=2025-01-03\ 17:29:09 --average_sequence_length=250000 --n_forcing=10
#CUDA_VISIBLE_DEVICES=1  python $justexp_FILE --mu=0.1 --rho=4.0 --dt=4 --load_index=$loadindex --load_date_time=2024-12-31\ 09:53:33 --average_sequence_length=30000 --n_forcing=10
#CUDA_VISIBLE_DEVICES=1  python $justexp_FILE --mu=0.1 --rho=4.0 --dt=4 --load_index=$loadindex --load_date_time=2024-12-20\ 15:32:44 --average_sequence_length=30000 --n_forcing=10
CUDA_VISIBLE_DEVICES=1  python $justexp_FILE --mu=1.7894e-05 --rho=1.225 --dt=4 --load_index=$loadindex --load_date_time=2025-02-11\ 10:07:59 --average_sequence_length=30000 --n_forcing=10


find_FILE="findtargetfreq_tosh.py"
findexp_FILE="findtargetfreq_now.py"

sed "s/expname/$expname/" $find_FILE > $findexp_FILE
# Python 코드 실행 및 출력 캡처
output=$(python3 $findexp_FILE)

# Python 출력에서 targetfreq 추출
targetfreq=$(echo "$output" | grep -oP "TARGET_FREQ=\K.*")

# 결과 출력
echo "Target Frequency: $targetfreq"


# 원본 파일
shift_FILE="shift_tosh.py"
# 수정된 파일 저장 경로
shift1_FILE="imsi_shift.py"
shift2_FILE="imsi2_shift.py"
shift3_FILE="shift_now.py"
sed "s/expname/$expname/" $shift_FILE > $shift1_FILE
sed "s/targetfreq/$targetfreq/" $shift1_FILE > $shift2_FILE
sed "s|shiftphase|$shiftphase|" $shift2_FILE > $shift3_FILE

python $shift3_FILE

# 원본 파일
forc_FILE="forcing_mask.py"
# 수정된 파일 저장 경로
forc1_FILE="imsi_forcing.py"
forcexp_FILE="forcing_now.py"
# expname을 diff1000ep50으로 수정하고 결과를 새로운 파일에 저장
sed "s/expname/$expname/" $forc_FILE > $forc1_FILE
sed "s|shiftphase|$shiftphase_safe|" $forc1_FILE > $forcexp_FILE
# 수정된 파일 실행
CUDA_VISIBLE_DEVICES=1  python $forcexp_FILE --mu=1.7894e-05 --rho=1.225 --dt=4 --load_index=$loadindex --load_date_time=2025-02-11\ 10:07:59 --average_sequence_length=30000 --n_forcing=10
#CUDA_VISIBLE_DEVICES=1  python $forcexp_FILE --mu=1.7894e-05 --rho=1.225 --dt=4 --load_index=$loadindex --load_date_time=2024-12-11\ 15:52:03 --average_sequence_length=30000 --n_forcing=10
#CUDA_VISIBLE_DEVICES=1  python $forcexp_FILE --mu=0.1 --rho=4.0 --dt=4 --load_index=$loadindex --load_date_time=2024-12-20\ 15:32:44 --average_sequence_length=30000 --n_forcing=10
#CUDA_VISIBLE_DEVICES=1  python $forcexp_FILE --mu=0.1 --rho=4.0 --dt=4 --load_index=$loadindex --load_date_time=2025-01-03\ 17:29:09 --average_sequence_length=250000 --n_forcing=10
#CUDA_VISIBLE_DEVICES=1  python $forcexp_FILE --mu=0.1 --rho=4.0 --dt=4 --load_index=$loadindex --load_date_time=2025-01-09\ 15:59:21 --average_sequence_length=30000 --n_forcing=10
#CUDA_VISIBLE_DEVICES=1  python $forcexp_FILE --mu=0.1 --rho=4.0 --dt=4 --load_index=$loadindex --load_date_time=2024-12-31\ 09:53:33 --average_sequence_length=30000 --n_forcing=10

# 원본 파일
phase_FILE="plotphase_tosh.py"
# 수정된 파일 저장 경로
phase1_FILE="imsi_plotphase.py"
phase2_FILE="imsi2_plotphase.py"
phase3_FILE="plotphase_now.py"
sed "s/expname/$expname/" $phase_FILE > $phase1_FILE
sed "s/targetfreq/$targetfreq/" $phase1_FILE > $phase2_FILE
sed "s|shiftphase|$shiftphase_safe|" $phase2_FILE > $phase3_FILE

python $phase3_FILE

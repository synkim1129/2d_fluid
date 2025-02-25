
논문작업 실행을 하려면 bash do.sh

do.sh
같은 모델내 학습 epoch, shift phase 수정시: loadindex, shiftphase 수정
다른 모델에서 수행할시 : python justexp_FILE 라인과 forcexp_FILE 라인 수정 및 expname 수정

 forcing_5p.py
observation point 수정시: x_obs, y_obs 수정

findtargetfreq_tosh.py, plotphase_tosh.py, shift_tosh.py
시공간 analysis area 수정시: boxstart, spongebuf, spinup등 조정
*filterfreq: findtargetfreq_tosh.py에서 target frequency를 찾는 과정 중 앞에 튀는값을 탐색하는 문제를 해결하기 위한 변수. 수정할 필요 없을 것으로 추정. karmen vortex의 main fequency를 잘못 찾고 있는 것 같은 경우 수정 혹은 삭제


vardo.sh
논문 이후 작업(observation의 shift phase를 변화하며 한 실험)을 수행하는 코드. do.sh와 같은 구조로 진행
shift하는 phase를 수정하고 싶을 때: shiftphase1에서 shiftphase2로 바뀌는 구조. 이를 수정
shift phase가 바뀌는 방법을 수정하고 싶을 때: varshift_tosh.py를 쓰면 중간에 한 번 바뀌는 구조. varshift2_tosh.py를 쓰면 shiftphase1부터 shiftphase2까지 바뀌는 구조


mask 관련: obs_mask 업데이트와 관련되어 변경된 do.sh 관련 코드들은 imsi폴더 안에 있으며, mask 업데이트로 모델을 수행하기위해서는 pde_cnn등을 다시 맞추어서 수행하여야함





#CUDA_VISIBLE_DEVICES=1  python demo_interactive_netcdf.py --mu=0.1 --rho=4.0 --dt=4.0 --load_index=48 --average_sequence_length=2000
#CUDA_VISIBLE_DEVICES=1  python demo_interactive_forcing.py --mu=0.1 --rho=4.0 --dt=4.0 --load_index=246 --average_sequence_length=999
#CUDA_VISIBLE_DEVICES=1  python demo_interactive_netcdf.py --mu=1.0 --rho=1.225 --dt=0.1 --load_index=15 --average_sequence_length=50000
#CUDA_VISIBLE_DEVICES=1  python demo_interactive_forcing.py --mu=1.0 --rho=1.225 --dt=0.1 --load_index=51 --average_sequence_length=24999
#CUDA_VISIBLE_DEVICES=3   python train_sponge.py --regularize_grad_p=0.00001 --mu=1.7894e-05 --rho=1.225 --dt=0.1 --average_sequence_length=50000 --n_batches_per_epoch=50000
#CUDA_VISIBLE_DEVICES=1   python train_sponge.py --height=50 --width=150 --regularize_grad_p=0.001 --mu=1.7894e-05 --rho=1.225 --dt=0.1 --average_sequence_length=50000 --n_batches_per_epoch=50000
#CUDA_VISIBLE_DEVICES=1   python train.py --mu=1e-05 --rho=4 --dt=4 --average_sequence_length=5000 --n_batches_per_epoch=5000 --integrator='imex'
#CUDA_VISIBLE_DEVICES=1   python train.py --mu=1.7894e-05 --rho=1.225 --dt=4 --average_sequence_length=5000 --n_batches_per_epoch=5000 --integrator='implicit'
CUDA_VISIBLE_DEVICES=1   python train_forcing.py --mu=1.7894e-05 --rho=1.225 --dt=4 --load_pretrained_index=90 --load_pretrained_date_time=2024-10-14\ 16:52:30 --average_sequence_length=30000 --n_batches_per_epoch=30000 --integrator='implicit' --loss_diff=5 --n_forcing=10  
#CUDA_VISIBLE_DEVICES=1   python train_forcing2.py --load_pretrained_index=8 --load_pretrained_date_time=2024-07-18\ 12:24:20 --regularize_grad_p=0.00001 --mu=1.7894e-05 --rho=1.225 --dt=0.1 --average_sequence_length=50000 --n_batches_per_epoch=50000 --loss_diff=2000 
#CUDA_VISIBLE_DEVICES=1   python train_forcing2.py --load_pretrained_index=4 --load_pretrained_date_time=2024-07-22\ 09:55:29 --regularize_grad_p=0.001 --mu=1.7894e-05 --rho=1.225 --dt=0.1 --average_sequence_length=50000 --n_batches_per_epoch=50000 --loss_diff=2500 
#CUDA_VISIBLE_DEVICES=3   python train_forcing.py --load_pretrained_index=4 --load_pretrained_date_time=2024-07-22\ 09:55:29 --regularize_grad_p=0.001 --mu=1.7894e-05 --rho=1.225 --dt=0.1 --average_sequence_length=50000 --n_batches_per_epoch=50000 --loss_diff=1000
#CUDA_VISIBLE_DEVICES=1   python train_forcing.py --load_pretrained_index=8 --load_pretrained_date_time=2024-07-18\ 12:24:20 --regularize_grad_p=0.00001 --mu=1.7894e-05 --rho=1.225 --dt=0.1 --average_sequence_length=50000 --n_batches_per_epoch=50000 --loss_diff=10000 --load_index=16 --load_date_time=2024-07-24\ 11:20:47
#CUDA_VISIBLE_DEVICES=1   python train_open3.py --regularize_grad_p=0.001 --mu=1.7894e-05 --rho=1.225 --dt=0.1 --average_sequence_length=50000 --n_batches_per_epoch=50000 --loss_rad=0.05 --loss_bound=200 --loss_pbound=0
#CUDA_VISIBLE_DEVICES=1  python demo_interactive_forcing.py --mu=1.7894e-05 --rho=1.225 --dt=0.1 --load_index=13 --average_sequence_length=50000

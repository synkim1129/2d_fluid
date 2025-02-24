#CUDA_VISIBLE_DEVICES=1  python demo_interactive_netcdf.py --mu=0.1 --rho=4.0 --dt=4.0 --load_index=48 --average_sequence_length=2000
#CUDA_VISIBLE_DEVICES=1  python demo_interactive_forcing.py --mu=0.1 --rho=4.0 --dt=4.0 --load_index=246 --average_sequence_length=999
#CUDA_VISIBLE_DEVICES=1  python demo_interactive_netcdf.py --mu=1.0 --rho=1.225 --dt=0.1 --load_index=15 --average_sequence_length=50000
#CUDA_VISIBLE_DEVICES=1  python demo_interactive_forcing.py --mu=1.0 --rho=1.225 --dt=0.1 --load_index=51 --average_sequence_length=24999

#CUDA_VISIBLE_DEVICES=1  python demo_interactive_netcdf.py --mu=1.7894e-05 --rho=1.225 --dt=4 --load_index=90 --load_date_time=2024-10-14\ 16:52:30 --average_sequence_length=30000  
#CUDA_VISIBLE_DEVICES=1  python demo_interactive_netcdf.py --mu=0.1 --rho=4.0 --dt=4 --load_index=48 --load_date_time=2020-05-28\ 16:13:12 --average_sequence_length=30000  
#CUDA_VISIBLE_DEVICES=1  python demo_interactive_forcing.py --mu=1.7894e-05 --rho=1.225 --dt=4 --load_index=500 --load_date_time=2024-10-16\ 10:56:54 --average_sequence_length=4100
#CUDA_VISIBLE_DEVICES=0  python demo_interactive_just_5p.py --mu=1.7894e-05 --rho=1.225 --dt=4 --load_index=500 --load_date_time=2024-10-24\ 17:10:03 --average_sequence_length=30000 --n_forcing=10 
#CUDA_VISIBLE_DEVICES=0  python demo_interactive_just_5p.py --mu=1.7894e-05 --rho=1.225 --dt=4 --load_index=1000 --load_date_time=2024-10-24\ 17:10:25 --average_sequence_length=30000 --n_forcing=10 
CUDA_VISIBLE_DEVICES=1  python demo_interactive_just_5p.py --mu=1.7894e-05 --rho=1.225 --dt=4 --load_index=240 --load_date_time=2024-11-19\ 18:26:48 --average_sequence_length=30000 --n_forcing=10 
#CUDA_VISIBLE_DEVICES=0  python demo_interactive_just_5p.py --mu=1.7894e-05 --rho=1.225 --dt=4 --load_index=24 --load_date_time=2025-01-22\ 11:44:29 --average_sequence_length=30000 --n_forcing=10 
#CUDA_VISIBLE_DEVICES=0  python demo_interactive_just_5p.py --mu=1.7894e-05 --rho=1.225 --dt=4 --load_index=114 --load_date_time=2025-01-22\ 11:45:20 --average_sequence_length=30000 --n_forcing=10 
#CUDA_VISIBLE_DEVICES=0  python demo_mask.py --mu=1.7894e-05 --rho=1.225 --dt=4 --load_index=82 --load_date_time=2025-02-03\ 16:57:15 --average_sequence_length=30000 --n_forcing=10 
#CUDA_VISIBLE_DEVICES=0  python demo_mask.py --mu=1.7894e-05 --rho=1.225 --dt=4 --load_index=7 --load_date_time=2025-02-11\ 10:07:59 --average_sequence_length=30000 --n_forcing=10 
#CUDA_VISIBLE_DEVICES=1  python demo_interactive_forcing_5p.py --mu=1.7894e-05 --rho=1.225 --dt=4 --load_index=100 --load_date_time=2024-11-19\ 18:26:48 --average_sequence_length=30000 --n_forcing=10 

#CUDA_VISIBLE_DEVICES=0  python demo_interactive_forcing_5p.py --mu=1.7894e-05 --rho=1.225 --dt=4 --load_index=939 --load_date_time=2024-10-24\ 17:10:03 --average_sequence_length=50000 --n_forcing=10 
#CUDA_VISIBLE_DEVICES=3  python demo_interactive_forcing.py --mu=1.7894e-05 --rho=1.225 --dt=0.1 --load_index=13 --load_date_time=2024-07-24\ 11:20:47 --average_sequence_length=17000
#CUDA_VISIBLE_DEVICES=0  python demo_interactive_forcing2.py --mu=1.7894e-05 --rho=1.225 --dt=0.1 --load_index=10 --load_date_time=2024-08-01\ 11:49:51 --average_sequence_length=17000
#CUDA_VISIBLE_DEVICES=3  python demo_interactive_open.py --mu=1.7894e-05 --rho=1.225 --dt=0.1 --load_index=17 --load_date_time=2024-09-27\ 17:45:35 --average_sequence_length=41000

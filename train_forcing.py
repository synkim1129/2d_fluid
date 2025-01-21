import torch
from torch import nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from derivatives import (
	dx,
	dy,
	dx_left,
	dy_top,
	dx_right,
	dy_bottom,
	laplace,
	map_vx2vy_left,
	map_vy2vx_top,
	map_vx2vy_right,
	map_vy2vx_bottom,
	normal2staggered,
	toCuda,
	toCpu,
	params,
)
from derivatives import vector2HSV, rot_mac
from setups import Dataset
from Logger import Logger, t_step
from pde_cnn import get_Net
from get_param import get_hyperparam
import pdb

torch.manual_seed(0)
torch.set_num_threads(4)
np.random.seed(0)

print(f"Parameters: {vars(params)}")

mu = params.mu
rho = params.rho
dt = params.dt

# initialize fluid model
fluid_model = toCuda(get_Net(params, forcing=True))
fluid_model.train()

# load pretrained model
logger_pretrained = Logger(get_hyperparam(params), use_csv=False, use_tensorboard=False)
pretrained_model = toCuda(get_Net(params, forcing=False))

date_time, index = logger_pretrained.load_state(
	pretrained_model, None, datetime=params.load_pretrained_date_time, index=params.load_pretrained_index
)
print(f"loaded pretrained model {params.net}: {date_time}, index: {index}")
pretrained_model.eval()

# initialize Optimizer
optimizer = Adam(fluid_model.parameters(), lr=params.lr)

# initialize Logger and load model / optimizer if according parameters were given
logger = Logger(get_hyperparam(params) + " with_forcing", datetime=params.load_date_time, use_csv=False, use_tensorboard=params.log)
logger.save_params_to_file(params)

if params.load_latest or params.load_date_time is not None or params.load_index is not None:
	load_logger = Logger(get_hyperparam(params) + " with_forcing", use_csv=False, use_tensorboard=False)
	if params.load_optimizer:
		params.load_date_time, params.load_index = logger.load_state(fluid_model, optimizer, params.load_date_time, params.load_index)
	else:
		params.load_date_time, params.load_index = logger.load_state(fluid_model, None, params.load_date_time, params.load_index)
	params.load_index = int(params.load_index)
	print(f"loaded: {params.load_date_time}, {params.load_index}")
params.load_index = 0 if params.load_index is None else params.load_index

# initialize Dataset
dataset = Dataset(params.width,params.height,params.batch_size,params.dataset_size,params.average_sequence_length,max_speed=params.max_speed,dt=params.dt,forcing=True,n_forcing=params.n_forcing,forcing_type=params.forcing_type)

def bilinear_interpolation_mid(v1, v2, v3, v4):
    """
    a = 0.5, b = 0.5로 고정된 양선형 보간 함수.
    
    v1, v2, v3, v4: (N, D) 모양의 벡터 집합 텐서
                    N은 점의 개수, D는 벡터의 차원
    """
    # 각 벡터에 동일한 가중치(1/4)를 부여하여 보간
    v = 0.25 * (v1 + v2 + v3 + v4)
    return v


def bilinear_interpolation_rand(a, b, v1, v2, v3, v4):
    """
    여러 점에 대해 양선형 보간을 수행하는 함수.
    
    a, b: (N,) 모양의 좌표 텐서 (0 <= a, b < 1)
    v1, v2, v3, v4: (N, D) 모양의 벡터 집합 텐서
                    N은 점의 개수, D는 벡터의 차원
    """
    # a, b의 모양을 (N, 1)로 변환하여 브로드캐스팅 가능하도록 만듭니다.
    a = a.unsqueeze(1)  # (N, 1)
    b = b.unsqueeze(1)  # (N, 1)

    # x축 보간
    vA = (1 - a) * v1 + a * v2  # (N, D)
    vB = (1 - a) * v3 + a * v4  # (N, D)

    # y축 보간
    v = (1 - b) * vA + b * vB  # (N, D)

    return v


def loss_function(x):
	return torch.pow(x, 2)

# training loop
for epoch in range(params.load_index, params.n_epochs):

	for i in range(params.n_batches_per_epoch):
		if params.forcing_type in ["lattice", "mid_lattice"]:
			v_cond, cond_mask, flow_mask, a_old, p_old, X_obs, Y_obs, v_obs, obs_mask = toCuda(dataset.ask())
		elif params.forcing_type == "random":
			v_cond, cond_mask, flow_mask, a_old, p_old, X_obs, Y_obs, v_obs, obs_positions, obs_mask = toCuda(dataset.ask())

		# convert v_cond,cond_mask,flow_mask to MAC grid
		v_cond = normal2staggered(v_cond)
		cond_mask_mac = (normal2staggered(cond_mask.repeat(1, 2, 1, 1)) == 1).float()
		flow_mask_mac = (normal2staggered(flow_mask.repeat(1, 2, 1, 1)) >= 0.5).float()

		v_old = rot_mac(a_old)

		# predict new fluid state from old fluid state and boundary conditions using the neural fluid model
		a_new, p_new = fluid_model(a_old, p_old, flow_mask, v_cond, cond_mask, v_obs, obs_mask)
		v_new = rot_mac(a_new)

		# predict new fluid state using pretrained model
		with torch.no_grad():
			
			a_pre_new, _ = pretrained_model(a_old, p_old, flow_mask, v_cond, cond_mask)
			batch_indices = torch.arange(params.batch_size, dtype=torch.int64)[:, None]
			if params.forcing_type == 'lattice':
				index_obs = (batch_indices, slice(None), Y_obs, X_obs)
				v_obs[index_obs] = rot_mac(a_pre_new)[index_obs]
			elif params.forcing_type == 'mid_lattice':
				index_obs_1 = (batch_indices, slice(None), Y_obs, X_obs)
				index_obs_2 = (batch_indices, slice(None), Y_obs, X_obs + 1)
				index_obs_3 = (batch_indices, slice(None), Y_obs + 1, X_obs)
				index_obs_4 = (batch_indices, slice(None), Y_obs + 1, X_obs + 1)
				v_obs[index_obs_1] = bilinear_interpolation_mid(rot_mac(a_pre_new)[index_obs_1], rot_mac(a_pre_new)[index_obs_2], rot_mac(a_pre_new)[index_obs_3], rot_mac(a_pre_new)[index_obs_4])
			elif params.forcing_type == 'random':
				index_obs_1 = (batch_indices, slice(None), Y_obs, X_obs)
				index_obs_2 = (batch_indices, slice(None), Y_obs, X_obs + 1)
				index_obs_3 = (batch_indices, slice(None), Y_obs + 1, X_obs)
				index_obs_4 = (batch_indices, slice(None), Y_obs + 1, X_obs + 1)
				X_obs_position = obs_positions[:,:,0]
				Y_obs_position = obs_positions[:,:,1]
				v_obs[index_obs_1] = bilinear_interpolation_rand(X_obs_position, Y_obs_position, rot_mac(a_pre_new)[index_obs_1], rot_mac(a_pre_new)[index_obs_2], rot_mac(a_pre_new)[index_obs_3], rot_mac(a_pre_new)[index_obs_4])
    

		# compute boundary loss
		loss_bound = torch.mean(loss_function(cond_mask_mac * (v_new - v_cond))[:, :, 1:-1, 1:-1],dim=(1, 2, 3))

		# explicit / implicit / IMEX integration schemes
		if params.integrator == "explicit":
			v = v_old
		if params.integrator == "implicit":
			v = v_new
		if params.integrator == "imex":
			v = (v_new + v_old) / 2

		# compute loss for momentum equation
		loss_nav =  torch.mean(loss_function(flow_mask_mac[:,1:2]*(rho*((v_new[:,1:2]-v_old[:,1:2])/dt+v[:,1:2]*dx(v[:,1:2])+0.5*(map_vy2vx_top(v[:,0:1])*dy_top(v[:,1:2])+map_vy2vx_bottom(v[:,0:1])*dy_bottom(v[:,1:2])))+dx_left(p_new)-mu*laplace(v[:,1:2])))[:,:,1:-1,1:-1],dim=(1,2,3))+\
                torch.mean(loss_function(flow_mask_mac[:,0:1]*(rho*((v_new[:,0:1]-v_old[:,0:1])/dt+v[:,0:1]*dy(v[:,0:1])+0.5*(map_vx2vy_left(v[:,1:2])*dx_left(v[:,0:1])+map_vx2vy_right(v[:,1:2])*dx_right(v[:,0:1])))+dy_top(p_new)-mu*laplace(v[:,0:1])))[:,:,1:-1,1:-1],dim=(1,2,3))
		
		regularize_grad_p = torch.mean((dx_right(p_new)**2+dy_bottom(p_new)**2)[:,:,2:-2,2:-2],dim=(1,2,3))
		
		# optional: additional loss to keep mean of a / p close to 0
		loss_mean_a = torch.mean(a_new,dim=(1,2,3))**2
		loss_mean_p = torch.mean(p_new,dim=(1,2,3))**2

		# additional loss for the difference between v_new and v_obs at point (X, Y)
		selected_v_obs = v_obs[torch.arange(params.batch_size).unsqueeze(1).unsqueeze(2), :, Y_obs.unsqueeze(1), X_obs.unsqueeze(1)]
		selected_flow_mask = flow_mask[torch.arange(params.batch_size).unsqueeze(1).unsqueeze(2), :, Y_obs.unsqueeze(1), X_obs.unsqueeze(1)]


		if params.forcing_type == 'lattice':
			selected_v_new = v_new[torch.arange(params.batch_size).unsqueeze(1).unsqueeze(2), :, Y_obs.unsqueeze(1), X_obs.unsqueeze(1)]
			loss_diff = torch.sum(((selected_v_new - selected_v_obs) ** 2) * selected_flow_mask, dim=(1, 2, 3))
		elif params.forcing_type == 'mid_lattice':
			selected_v_new_1 = v_new[torch.arange(params.batch_size).unsqueeze(1).unsqueeze(2), :, Y_obs.unsqueeze(1), X_obs.unsqueeze(1)]
			selected_v_new_2 = v_new[torch.arange(params.batch_size).unsqueeze(1).unsqueeze(2), :, Y_obs.unsqueeze(1) + 1, X_obs.unsqueeze(1)]
			selected_v_new_3 = v_new[torch.arange(params.batch_size).unsqueeze(1).unsqueeze(2), :, Y_obs.unsqueeze(1), X_obs.unsqueeze(1) + 1]
			selected_v_new_4 = v_new[torch.arange(params.batch_size).unsqueeze(1).unsqueeze(2), :, Y_obs.unsqueeze(1) + 1, X_obs.unsqueeze(1) + 1]
			loss_diff = torch.sum((bilinear_interpolation_mid(selected_v_new_1, selected_v_new_2, selected_v_new_3, selected_v_new_4) - selected_v_obs) ** 2 * selected_flow_mask, dim=(1, 2, 3))
		elif params.forcing_type == 'random':
			selected_v_new_1 = v_new[torch.arange(params.batch_size).unsqueeze(1).unsqueeze(2), :, Y_obs.unsqueeze(1), X_obs.unsqueeze(1)]
			selected_v_new_2 = v_new[torch.arange(params.batch_size).unsqueeze(1).unsqueeze(2), :, Y_obs.unsqueeze(1), X_obs.unsqueeze(1) + 1]
			selected_v_new_3 = v_new[torch.arange(params.batch_size).unsqueeze(1).unsqueeze(2), :, Y_obs.unsqueeze(1) + 1, X_obs.unsqueeze(1)]
			selected_v_new_4 = v_new[torch.arange(params.batch_size).unsqueeze(1).unsqueeze(2), :, Y_obs.unsqueeze(1) + 1, X_obs.unsqueeze(1) + 1]
			loss_diff = torch.sum((bilinear_interpolation_rand(obs_positions[:,:, 0], obs_positions[:,:, 1], selected_v_new_1, selected_v_new_2, selected_v_new_3, selected_v_new_4) - selected_v_obs) ** 2 * selected_flow_mask, dim=(1, 2, 3))
		
		loss = (
			params.loss_bound * loss_bound
			+ params.loss_nav * loss_nav
			+ params.loss_mean_a * loss_mean_a
			+ params.loss_mean_p * loss_mean_p
			+ params.regularize_grad_p * regularize_grad_p
			+ params.loss_diff * loss_diff
		)

		loss = torch.mean(torch.log(loss))

		# compute gradients
		optimizer.zero_grad()
		loss = loss * params.loss_multiplier # ignore the loss_multiplier (could be used to scale gradients)
		loss.backward()

		# optional: clip gradients
		if params.clip_grad_value is not None:
			torch.nn.utils.clip_grad_value_(fluid_model.parameters(),params.clip_grad_value)
		if params.clip_grad_norm is not None:
			torch.nn.utils.clip_grad_norm_(fluid_model.parameters(),params.clip_grad_norm)

		# perform optimization step
		optimizer.step()

		p_new.data = (p_new.data-torch.mean(p_new.data,dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))#normalize pressure
		a_new.data = (a_new.data-torch.mean(a_new.data,dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))#normalize a

		# recycle data to improve fluid state statistics in dataset
		dataset.tell(toCpu(a_new), toCpu(p_new), v_obs=toCpu(v_obs))

		# log training metrics
		if i % 10 == 0:
			loss = toCpu(loss).numpy()
			loss_bound = toCpu(torch.mean(loss_bound)).numpy()
			loss_nav = toCpu(torch.mean(loss_nav)).numpy()
			loss_diff = toCpu(torch.mean(loss_diff)).numpy()
			regularize_grad_p = toCpu(torch.mean(regularize_grad_p)).numpy()
			logger.log(f"loss_{params.loss}", loss, epoch * params.n_batches_per_epoch + i)
			logger.log(f"loss_bound_{params.loss}",loss_bound,epoch * params.n_batches_per_epoch + i,)
			logger.log(f"loss_nav_{params.loss}",loss_nav,epoch * params.n_batches_per_epoch + i,)
			logger.log(f"loss_diff_{params.loss}",loss_diff,epoch * params.n_batches_per_epoch + i,)
			logger.log(f"regularize_grad_p",regularize_grad_p,epoch * params.n_batches_per_epoch + i,)

			if i % 100 == 0:
				print(f"{epoch}: i:{i}: loss: {loss}; loss_bound: {loss_bound}; loss_nav: {loss_nav}; loss_diff: {loss_diff}")

	# safe state after every epoch
	if params.log:
		logger.save_state(fluid_model, optimizer, epoch + 1)

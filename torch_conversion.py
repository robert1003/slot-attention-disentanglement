"""
Hacked together script to convert TF slot attention checkpoint to a PyTorch model checkpoint. Some changes were made to the PyTorch
model in order to make it match this TF implementation. Put this file in object_discovery directory of the TF implementation and 
update paths below before running.

A similar TF/torch converter for Slot Attention is implemented here: https://github.com/vadimkantorov/yet_another_pytorch_slot_attention

Official TF implementation: https://github.com/google-research/google-research/tree/master/slot_attention/
CLEVR Weights: https://console.cloud.google.com/storage/browser/gresearch/slot-attention/object-discovery;tab=objects?prefix=&forceOnObjectsSortingFiltering=false 
"""


from absl import logging
import tensorflow as tf



# import sys
# sys.path.append("/Users/andrewstange/Desktop/CMU/Spring_2023/16-824/Project/slot_attention")


import data as data_utils
import model as model_utils

# Hyperparameters.
seed = 0
batch_size = 1
num_slots = 7
num_iterations = 3
resolution = (128, 128)
ckpt_path = "./checkpoint/tf/object_discovery"               # Path to directory containing TF model checkpoint
output_ckpt_path = "./checkpoint/tf/pytorch/object_discovery.pt"         # Path to store torch checkpoint

# Path to torch implementation, rename model.py to torch_model.py to avoid file name conflicts with TF model.py
pytorch_impl_path = "/home/robert1003/Desktop/spring-23/16-824/project" 



def load_model(checkpoint_dir, num_slots=11, num_iters=3, batch_size=16):
	resolution = (128, 128)
	model = model_utils.build_model(
		resolution, batch_size, num_slots, num_iters,
		model_type="object_discovery")

	ckpt = tf.train.Checkpoint(network=model)
	ckpt_manager = tf.train.CheckpointManager(
		ckpt, directory=checkpoint_dir, max_to_keep=5)

	if ckpt_manager.latest_checkpoint:
		ckpt.restore(ckpt_manager.latest_checkpoint)
		logging.info("Restored from %s", ckpt_manager.latest_checkpoint)
	else:
		print("\n\nFAILED TO LOAD CHECKPOINT\n\n")

	return model


tfmodel = load_model(ckpt_path, num_slots=num_slots, num_iters=num_iterations,
				   batch_size=batch_size)



#  ---------------------- (ASTANGE) personal experimentation below this line ----------------------
import torch
import sys
sys.path.insert(1, pytorch_impl_path)
from torch_model import SlotAttentionAutoEncoder as SAAE		# changed name of Pytorch model.py to torch_model.py to avoid file naming conflicts
hid_dim = 64        # hardcoded in this implementation

# Create torch model
torchmodel = SAAE(resolution=resolution, num_slots=num_slots, num_iterations=num_iterations, hid_dim=hid_dim)


def key_translate(torch_key):
	if "conv" in torch_key:
		# Handles encoder/decoder convs
		foo = torch_key.split(".")
		if int(foo[1][-1]) > 1:
			# tf and torch conv layer indices are off by 1
			foo[1] = "_" + str(int(foo[1][-1]) - 1) + "/"		
		else:
			foo[1] = str("/")
		torch_key = "".join(foo)
		out = torch_key.replace("encoder_cnn", "conv2d").replace("decoder_cnn", "conv2d_transpose").replace("weight", "kernel:0").replace("bias", "bias:0")
	elif "_pos." in torch_key:
		# Handles encoder and decoder positional encoding layers
		out = torch_key.replace("encoder_cnn.encoder_pos.embedding.", "slot_attention_auto_encoder/soft_position_embed/dense/")
		out = out.replace("decoder_cnn.decoder_pos.embedding.", "slot_attention_auto_encoder/soft_position_embed_1/dense_1/")
		out = out.replace("weight", "kernel:0").replace("bias", "bias:0")
	elif "encoder_layer_norm" in torch_key:
		# post-encoder layer norm
		out = torch_key.replace("encoder_layer_norm.", "slot_attention_auto_encoder/layer_normalization/").replace("weight", "gamma:0").replace("bias", "beta:0")
	elif "fc" in torch_key[:2]:
		# pre-slot attention linear layers
		out = torch_key.replace("fc1.", "dense_2/").replace("fc2.", "dense_3/").replace("weight", "kernel:0").replace("bias", "bias:0")
	elif "slot_attention.slots" in torch_key:
		# slot sampling parameters
		out = torch_key.replace("slot_attention.slots_mu", "slots_mu:0").replace("slot_attention.slots_log_sigma", "slots_log_sigma:0")
	elif "fc" in torch_key:
		# slot attention linear layers
		out = torch_key.split(".", maxsplit=1)[1].replace("fc1.", "dense_4/").replace("fc2.", "dense_5/")
		out = out.replace("weight", "kernel:0").replace("bias", "bias:0")
	elif "slot_attention" in torch_key:
		# attention mechanism and slot attention layer norms
		out = torch_key.replace("slot_attention.", "slot_attention_auto_encoder/slot_attention/")
		out = out.replace("to_q.", "q/").replace("to_k.", "k/").replace("to_v.", "v/")
		out = out.replace("gru.", "gru_cell/").replace("weight_ih", "kernel:0").replace("weight_hh", "recurrent_kernel:0")
		if "norm" in out:
			out = out.replace("norm_input.", "layer_normalization_1/").replace("norm_slots.", "layer_normalization_2/").replace("norm_pre_ff.", "layer_normalization_3/")
			out = out.replace("weight", "gamma:0").replace("bias", "beta:0")
		else:
			out = out.replace("weight", "kernel:0").replace("bias", "bias:0")
	else:
		out = torch_key
	return out


# Iterate through tensorflow weights and transfer them into torch model
tf_params = {param.name : param.numpy() for param in tfmodel.trainable_variables}
for i in list(tf_params.keys()):
	print(i)

torch_names = set()
with torch.no_grad():
	for name, param in torchmodel.named_parameters():
		# Translate torch name to tf name to get parameter value
		k = key_translate(name)
		if name == "slot_attention.gru.bias_ih":
			# tf stacks GRU bias into single parameter, torch doesn't
			v = tf_params["slot_attention_auto_encoder/slot_attention/gru_cell/bias:0"][0]
			k = "slot_attention_auto_encoder/slot_attention/gru_cell/bias:0"
		elif name == "slot_attention.gru.bias_hh":
			v = tf_params["slot_attention_auto_encoder/slot_attention/gru_cell/bias:0"][1]
			k = "slot_attention_auto_encoder/slot_attention/gru_cell/bias:0"
		else:
			v = tf_params[k]

		# Set torch parameter value to tf parameter value 
		v = torch.tensor(v)
		if "weight" in name:
			if v.dim() == 4:
				# tf parameters stored in a different order than torch
				# Generally follows cell 12 https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb 
				# https://discuss.pytorch.org/t/converting-tensorflow-model-weights-to-pytorch/99761 
				v = v.permute((3, 2, 0, 1))
			elif v.dim() == 2:
				v = v.T
		assert param.shape == v.shape
		param.copy_(v)

		if k in tf_params:
			print(name, " --> ", k)
		else:
			print(name)

		# Ensure every torch parameter is matched to a tf parameter
		assert k in tf_params
		torch_names.add(k)

# Ensure every tf parameter is match to a torch parameter
for i in list(tf_params.keys()):
	assert i in torch_names



# Save torch model to a file
torch.save(torchmodel.state_dict(), output_ckpt_path)
print("\n\nWeight successfully converted and stored.")


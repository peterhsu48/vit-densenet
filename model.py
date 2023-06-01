import random
import time

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from wilds.common.data_loaders import get_eval_loader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision
from torchvision.models import VisionTransformer
from torchvision.models import DenseNet
import numpy as np

# Tracks approximate runtime duration
# MAX_RUNTIME may be useful if using a shared computing resource
runtime_start = time.time()
MAX_RUNTIME = 24 * 60 * 60

# Set restore_path to the path of the checkpoint file if one is used
# Otherwise, set to None
restore_path = None
if restore_path != None:
  checkpoint = torch.load(restore_path)
  print("Checkpoint restored from " + restore_path)
else:
  checkpoint = None

# Downloads dataset to current working directory
dataset = get_dataset(dataset="camelyon17", download=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Train one epoch
# Returns average loss per mini-batch
def train_epoch(train_loader, model, loss_fn, optimizer, device):

  # Store loss from each mini-batch in this total for the current epoch
  loss_total = 0.0

  model.train()

  for batch, (x, y, metadata) in enumerate(train_loader):
    # Mini-batch numbers start at 0

    x = x.to(device)
    y = y.to(device)

    predictions = model(x)
    loss = loss_fn(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_total += loss.item()

    if batch % 1000 == 0:
      print("Batch {0} Loss: {1}".format(batch, loss.item()))

  return loss_total / len(train_loader)

# Calculate number of true positives and true negatives
def calculate_tp_tn(predictions, y):

  # Determine the index with the greater probability value for each example
  predictions_binary = torch.argmax(predictions, dim=1)

  # Determine which examples have a match in prediction and label
  matches = torch.eq(predictions_binary, y)

  # Return the number of true positives and true negatives
  return matches.to(torch.int).sum().item()

# Run validation data
# Returns average loss per mini-batch and validation accuracy for entire val split
def validation(val_loader, model, loss_fn, len_val, device):
  # len_val: number of images in the validation split

  model.eval()

  with torch.no_grad():

    # Store loss from each batch in this total
    loss_total = 0.0

    # Store number of true positives and true negatives for entire dataset
    tp_tn = 0

    for batch, (x, y, metadata) in enumerate(val_loader):

      x = x.to(device)
      y = y.to(device)

      predictions = model(x)
      loss = loss_fn(predictions, y)

      loss_total += loss.item()

      tp_tn += calculate_tp_tn(predictions, y)

  return loss_total / len(val_loader), tp_tn / len_val

class VisionTransformerNoHead(VisionTransformer):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    del self.heads

  def forward(self, x):
    # forward function from PyTorch with model head operation removed
    # Reshape and permute the input tensor
    x = self._process_input(x)
    n = x.shape[0]

    # Expand the class token to the full batch
    batch_class_token = self.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)

    x = self.encoder(x)

    return x
  
class DenseNetNoHead(DenseNet):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    del self.features.norm5
    del self.classifier
    
  def forward(self, x: Tensor) -> Tensor:
    features = self.features(x)
    return features
  
class Transition_2(nn.Sequential):
  def __init__(self, num_input_features):
    super().__init__()
    self.norm = nn.BatchNorm2d(num_input_features)
    self.relu = nn.ReLU(inplace=True)
    self.conv = nn.Conv2d(num_input_features, num_input_features//2, kernel_size=1, stride=1, bias=False)
    self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

class DenseNetNoTail(DenseNet):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    del self.features.conv0
    del self.features.norm0
    del self.features.relu0
    del self.features.pool0

class Av1(nn.Module):
  
  def __init__(self,
               image_size,
               patch_size,
               num_layers,
               num_heads,
               hidden_dim,
               mlp_dim,
               transpose_out,
               growth_rate,
               block_config_branch_1,
               block_config_branch_2,
               block_config_shared,
               num_init_features,
               num_classes,
               cls_out
               ):
    super().__init__()
    self.image_size = image_size
    self.patch_size = patch_size
    # Vision Transformer
    self.vit = VisionTransformerNoHead(image_size=image_size,
                      patch_size=patch_size,
                      num_layers=num_layers,
                      num_heads=num_heads,
                      hidden_dim=hidden_dim,
                      mlp_dim=mlp_dim)
    
    # Transition 1
    self.convt = nn.ConvTranspose2d(in_channels=hidden_dim,
                                    out_channels=transpose_out,
                                    kernel_size=patch_size,
                                    stride=patch_size)
    
    # Branch 1
    self.branch_1 = DenseNetNoHead(growth_rate=growth_rate,
                            block_config=block_config_branch_1,
                            num_init_features=num_init_features,
                            num_classes=num_classes) # num_classes means nothing here
    
    # Branch 2
    self.branch_2 = DenseNetNoHead(growth_rate=growth_rate,
                            block_config=block_config_branch_2,
                            num_init_features=num_init_features,
                            num_classes=num_classes) # num_classes means nothing here

    # Calculate number of features going into Transition 2
    total_features = 0
    for block_config in [block_config_branch_1, block_config_branch_2]:
      num_features = num_init_features
      for i, layers in enumerate(block_config):
        num_features = num_features + layers * growth_rate
        if i != len(block_config) - 1:
          num_features = num_features // 2
      total_features = total_features + num_features

    # Transition 2
    self.transition_2 = Transition_2(num_input_features=total_features)

    # Shared
    self.shared = DenseNetNoTail(growth_rate=growth_rate,
                            block_config=block_config_shared,
                            num_init_features=total_features//2,
                            num_classes=num_classes)
    
    # Classifier
    self.classifier = nn.Sequential(
      nn.Linear(in_features=num_classes+cls_out, out_features=2),
      nn.Softmax(dim=1)
    )

  def forward(self, x):
    # Vision Transformer
    x = self.vit(x)
    
    # Transition 1
    # Remove classification token
    x = x[:, 1:]
    # Reverse permute
    x = x.permute((0, 2, 1))
    # Reverse reshape
    x = x.reshape((x.shape[0],
                   x.shape[1],
                   self.image_size // self.patch_size,
                   self.image_size // self.patch_size))
    # Transposed convolution
    x = self.convt(x)

    # Branch 1 (Complex)
    x1 = self.branch_1(x.clone())

    # Branch 2 (Simple)
    x2 = self.branch_2(x)

    # Concatenation
    x = torch.cat((x1,x2), dim=1)
    del x1
    del x2

    # Transition 2
    x = self.transition_2(x)

    # Shared
    x = self.shared(x)
    
    # Classifier
    x = self.classifier(x)

    return x

# Architecture Specifications
# ViT
image_size = 224 # H (or W) of ViT input
patch_size = 16 # ViT patch size
num_layers = 12 # Encoder layers
num_heads = 16 # Heads for multi-headed self-attention
hidden_dim = 1024 # Depth of each token
mlp_dim = 4096 # Dim for MLP
# Transition
transpose_out = 3 # Output dim for transposed convolution
# DenseNet
growth_rate = 48 # Growth rate for DenseNet
block_config_branch_1 = (10, 12) # Dense layers for each block of branch 1
block_config_branch_2 = (6, 8) # Dense layers for each block of branch 2
block_config_shared = (16,) # Dense layers for each block of shared
num_init_features = 96 # Filters for first convolutional layer
num_classes = 1000 # Number of output values from DenseNet
# Classification
cls_out = 0

# Model instantiation function
def av1(image_size,
        patch_size,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        transpose_out,
        growth_rate,
        block_config_branch_1,
        block_config_branch_2,
        block_config_shared,
        num_init_features,
        num_classes,
        cls_out,
        checkpoint
        ):
  model = Av1(image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            transpose_out=transpose_out,
            growth_rate=growth_rate,
            block_config_branch_1=block_config_branch_1,
            block_config_branch_2=block_config_branch_2,
            block_config_shared=block_config_shared,
            num_init_features=num_init_features,
            num_classes=num_classes,
            cls_out=cls_out)
  
  if checkpoint == None:
    av1_state_dict = model.state_dict()

    vit_enum = torchvision.models.ViT_L_16_Weights.IMAGENET1K_V1
    vit_state_dict = vit_enum.get_state_dict(progress=False)

    # Av1 parameter names have a "vit" prefix for ViT layers, whereas
    # torchvision ImageNet weights do not.
    # Append "vit" to ImageNet parameter names.

    # ImageNet weights from torchvision name the mlp linear layers as "linear_1"
    # and "linear_2". Av1 uses "0" and "3", respectively, instead. Change ImageNet weight
    # names to match Av1.

    vit_keys = list(vit_state_dict.keys())
    av1_keys = list(av1_state_dict.keys())
    for key in vit_keys:
      new_key_name = key
      if "mlp" in key:
        linear_num = -1
        linear_name = ""
        if "linear_1" in key:
          linear_name = "linear_1"
          linear_num = 0
        elif "linear_2" in key:
          linear_name = "linear_2"
          linear_num = 3
        else:
          assert False
        prefix = key[:key.find(linear_name)]
        postfix = key[key.find(linear_name)+len(linear_name):]
        new_key_name = prefix + str(linear_num) + postfix
      new_key_name = "vit." + new_key_name
      vit_state_dict[new_key_name] = vit_state_dict[key]
      del vit_state_dict[key]
      if new_key_name not in av1_keys:
        del vit_state_dict[new_key_name]

    av1_state_dict.update(vit_state_dict)
    model.load_state_dict(av1_state_dict)
  else:
    model.load_state_dict(checkpoint["model_state_dict"])

  return model

def get_subsets_loaders(dataset, batch_size):

  vit_transforms = torchvision.models.ViT_L_16_Weights.IMAGENET1K_V1.transforms()

  # Train subset
  train_data = dataset.get_subset(
      "train",
      transform=vit_transforms
  )

  # Train loader
  train_loader = get_train_loader("standard", train_data, batch_size=batch_size)

  # Validation subset
  val_data = dataset.get_subset(
      "val",
      transform=vit_transforms
  )

  # Validation loader
  val_loader = get_eval_loader("standard", val_data, batch_size=batch_size)

  return train_data, train_loader, val_data, val_loader

# Hyperparameters
learning_rate = 0.01
l2_regularization = 0.1
momentum = 0.9
epochs = 5
batch_size = 18

seed = 44

if checkpoint == None:
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
  print("Manually seeded torch, random, and np")
else:
  torch.random.set_rng_state(checkpoint["torch_state"])
  random.setstate(checkpoint["random_state"])
  np.random.set_state(checkpoint["np_state"])
  print("Restored random state for torch, random, and np")
  
train_data, train_loader, val_data, val_loader = get_subsets_loaders(dataset, batch_size)

model = av1(image_size, patch_size, num_layers, num_heads, hidden_dim,
          mlp_dim, transpose_out, growth_rate, block_config_branch_1, block_config_branch_2,
          block_config_shared, num_init_features, num_classes, cls_out, checkpoint)

# Freeze ViT layers
for param in model.vit.parameters():
  param.requires_grad = False

branch_1_params = []
for param in model.branch_1.parameters():
  branch_1_params.append(param)

other_params_requires_grad = []
for name, param in model.named_parameters():
  if param.requires_grad and name.find("branch_1") == -1:
    other_params_requires_grad.append(param)

model = model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()
  
optimizer = torch.optim.SGD([
    {"params": branch_1_params, "weight_decay": 0.01},
    {"params": other_params_requires_grad}
], lr=learning_rate, momentum=momentum, weight_decay=l2_regularization)

starting_epoch = 1
if checkpoint != None:
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  starting_epoch = checkpoint['epoch'] + 1

# Store training loss values from each epoch in this list
train_loss_record = []

# Store validation loss values from each epoch in this list
val_loss_record = []

# Store validation accuracy values from each epoch in this list
val_acc_record = []

max_epoch_time = 0

# Epoch numbers start at 1
for epoch in range(starting_epoch, epochs + 1):

  epoch_start = time.time()
    
  print("------- Epoch {0} -------".format(epoch))

  if epoch == 3:
    for group in optimizer.param_groups:
      group['lr'] = 0.003
    print("Changed learning rate to 0.003")
  elif epoch > 3:
    for group in optimizer.param_groups:
      group['lr'] = 0.001
    print("Changed learning rate to 0.001")

  # Training
  start = int(time.time())

  train_loss = train_epoch(train_loader, model, loss_fn, optimizer, device)
  train_loss_record.append(train_loss)

  end = int(time.time())

  print("Training loss for this epoch: {0}".format(train_loss_record[-1]))
  print("Approximate training time: {0} minutes and {1} seconds".format((end - start) // 60, (end - start) % 60))

  start = int(time.time())

  # Validation
  val_loss, val_acc = validation(val_loader, model, loss_fn, len(val_data), device)
  val_loss_record.append(val_loss)
  val_acc_record.append(val_acc)

  end = int(time.time())

  print("Validation loss for this epoch: {0}".format(val_loss_record[-1]))
  print("Validation accuracy for this epoch: {0}".format(val_acc_record[-1]))
  print("Approximate validation time: {0} minutes and {1} seconds".format((end - start) // 60, (end - start) % 60))

  # Save checkpoint
  checkpoint_path = "./checkpoint" + "_seed_" + str(seed) + "_epoch_" + str(epoch) + ".pt"
  torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "torch_state": torch.random.get_rng_state(),
    "random_state": random.getstate(),
    "np_state": np.random.get_state(),
  }, checkpoint_path)
  print("Saved checkpoint to", checkpoint_path)

  epoch_end = time.time()

  # Calculate the epoch duration and record if it is the longest epoch
  epoch_time = epoch_end - epoch_start
  if epoch_time > max_epoch_time:
    max_epoch_time = epoch_time
  
  # Calculate the runtime duration currently
  runtime_time = time.time() - runtime_start

  # Calculate the projected amount of time for one more epoch
  # If projected time exceeds MAX_RUNTIME, exit
  projected_runtime = runtime_time + max_epoch_time
  if projected_runtime >= MAX_RUNTIME:
    print("Projected runtime exceeds max time")
    break
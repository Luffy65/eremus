from eremus.utils.saver import Saver

import eremus.models as models
from eremus.utils.models import add_net_to_params
from eremus.models.eeg_transformer import generate_square_subsequent_mask

from tqdm import tqdm
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

# Histogram logging utilities
hooks = None # initialized each time
module_name_map = None # initialized each time
module_calling_order = None # initialized each time
name_position_map = None # initialized each time
log_histograms_errors = None # initialized each time
                                
# Setup forward hooks
def setup_forward_hooks(net):
    # Globals
    global hooks, module_name_map, module_calling_order, name_position_map, log_histograms_errors
    # Reset variables
    module_name_map = {}
    module_calling_order = []
    name_position_map = {}
    hooks = []
    log_histograms_errors = []
    # Add hooks
    for name,module in net.named_modules():
        if name != '':
            # Add to module-name map
            module_name_map[module] = name
            # Add hook
            h = module.register_forward_hook(forward_hook)
            hooks.append(h)

# Forward hook to save module output and update module order
def forward_hook(module, input, output):
    #*********************************
    #Hook to store last output of a module into a module's variable.
    #Used to log output histograms.
    #This function handles tensor outputs and tuple of tensor outputs
    #*****************************
    # Globals
    global hooks, module_name_map, module_calling_order, name_position_map, log_histograms_errors
    # Update calling order
    name = module_name_map[module]
    module_calling_order.append(name)
    name_position_map[name] = module_calling_order.index(name)
    # Check type
    if isinstance(output, torch.Tensor):
        # Save output in module
        module.last_output = output.detach().clone()
    elif isinstance(output, tuple):
        # Concatenate into single tensor
        tensors = []
        for t in output:
            tensors.append(t.detach().clone().view(-1))
        tensors = torch.cat(tensors, 0)
        # Save output in module
        module.last_output = tensors

# Log histograms
def log_histograms(net, step, saver, histograms=False):
    # Globals
    global hooks, module_name_map, module_calling_order, name_position_map, log_histograms_errors
    # Locals
    grad_norm_by_depth = {name: [] for name in module_calling_order}
    param_norm_by_depth = {name: [] for name in module_calling_order}
    output_norm_by_depth = {name: [] for name in module_calling_order}
    # Log output histograms
    for name,module in net.named_modules():
        if name != '' and hasattr(module, 'last_output'):
            # Get module position
            module_pos = module_calling_order.index(name)
            # Log norm
            output_norm = module.last_output.norm(p=2).item()
            saver.dump_metric(output_norm, step, f'output_norm/{module_pos:04}_{name}')
            # Add norm to depth map
            output_norm_by_depth[name].append(output_norm)
            # Log histogram
            if histograms:
                saver.dump_histogram(module.last_output, step, f'output_hist/{module_pos:04}_{name}')
    # Remove hooks
    for h in hooks:
        h.remove()
    # Log parameters and gradients 
    for name,param in net.named_parameters():
        try:
            # Get module position
            module_name = None
            module_pos = None
            parts = name.split('.')
            while len(parts) > 0:
                # Check name
                module_name = '.'.join(parts)
                if module_name in name_position_map:
                    module_pos = name_position_map[module_name]
                    break
                else:
                    parts = parts[:-1]
            # Check module position
            if module_pos is None:
                print(f'Could not find module position for param {name}')
            else:
                # Log norm
                grad_norm = param.grad.norm(p=2).item()
                param_norm = param.data.norm(p=2).item()
                saver.dump_metric(grad_norm, step, f'grad_norm/{module_pos:04}_{name}')
                saver.dump_metric(param_norm, step, f'param_norm/{module_pos:04}_{name}')
                # Add norm to depth map
                grad_norm_by_depth[module_name].append((grad_norm,name))
                param_norm_by_depth[module_name].append((param_norm,name))
                # Log histogram
                if histograms:
                    saver.dump_histogram(param.grad, step, f'grad/{module_pos:04}_{name}')
                    saver.dump_histogram(param.data, step, f'param/{module_pos:04}_{name}')
        except:
            if name not in log_histograms_errors:
                print(f'Param {name} has no gradient')
                log_histograms_errors.append(name)
    # Log by depth
    if histograms:
        # Initialize lines
        output_norm_by_depth_line = []
        grad_norm_by_depth_line = []
        param_norm_by_depth_line = []
        output_labels = []
        param_labels = []
        # Process sequence of modules
        for module_name in module_calling_order:
            # Add points for outputs
            output_norm_by_depth_line += output_norm_by_depth[module_name]
            # Add output labels
            output_labels.append(module_name)
            # Add points 
            grad_norm_by_depth_line += [x[0] for x in grad_norm_by_depth[module_name]]
            param_norm_by_depth_line += [x[0] for x in param_norm_by_depth[module_name]]
            # Add param labels
            param_labels += [x[1] for x in grad_norm_by_depth[module_name]]
        # Save lines
        saver.dump_line(torch.tensor(output_norm_by_depth_line), step, 'train', f'norms_by_depth/output', labels=output_labels)
        saver.dump_line(torch.tensor(grad_norm_by_depth_line), step, 'train', f'norms_by_depth/grad', labels=param_labels)
        saver.dump_line(torch.tensor(param_norm_by_depth_line), step, 'train', f'norms_by_depth/param', labels=param_labels)
    # Print errors
    if len(log_histograms_errors) > 0:
        print("Errors in logging histograms for params:")
        for name in log_histograms_errors:
            print(f"- {name}")

def norm_01(x):
    x = x.clone()
    for b in range(x.shape[0]):
        x[b] = (x[b] - x[b].min())/(x[b].max() - x[b].min())
    return x

class Trainer:
    """
    Use this class to train EREMUS Dataset.
    
    Arguments
    -----------
    args : dict
        A dictionary containing the following keys:
        
        | **Experiment Options**
        | <*saver*> eremus.utils.saver.Saver : the saver to use in order to keep metrics.
        | <*log_every*> int : set log frequency in ephocs.
        | <*plot_every*> int : set plot frequency in ephocs.
        | <*save_every*> int : set saving frequency in ephocs.
        | <*log_histograms*> bool : If True, plot histograms every *plot_every* epochs. You shouoldn't use it.
        | <*tag*> str : give a tag to the experiment.
        
        | **Model Options**
        | <*model*> str : the neural network module (e.g. ra_cnn). Module arguments should be also added to this dictionary. 
        | <*num_classes*> int : set the number of classes for classification tasks.
        
        | **Training Options**
        | <*resume*> str : the path to a checkpoint directory or *None*.
        | <*multi_gpu*> bool : if available and True, use multiple GPUs.
        | <*device*> str : set device to *cpu* or *cuda*.
        | <*train_labels*> list : pre-load labels and pass them.
        | <*batch_size*> int : the batch size.
        | <*workers*> int : set the number of workers for multi-threading trainings.
        | <*overfit_batch*> bool : if True, set the dataset to a small sample number (single batch per split). 
        | <*lr*> float : the learning rate.
        | <*weight_decay*> double : the weight decay.
        | <*optim*> str : set Ooptimizer to *SGD* or *Adam*.
        | <*reduce_lr_every*> int : If not None, reduce learning rate every *reduce_lr_every* epochs of *reduce_lr_factor*.
        | <*reduce_lr_factor*> float : reduce learning rate every *reduce_lr_every* epochs of *reduce_lr_factor*. 
        | <*epochs*> int : set the number of training ephocs.
        | <*eval_after*> int : do not use it (set to 0).
    """

    def __init__(self, args):
        # Store args
        self.args = args

    def train(self, datasets):
        """
        Train the given dataset!
        
        Parameters
        -------------
        datasets : dict
            A dictionary containing the following keys:
            
            | <*train*> torch.utils.data.Dataset : the train split of Dataset.
            | <*test*> torch.utils.data.Dataset : the test split of Dataset.
            | <*val*> torch.utils.data.Dataset : the validation split of Dataset.
        
        Returns
        -------------
        nn.Module
            The neural network used in training.
            
        dict
            A dictionary containing the following keys:
            
            | <*train*> dict : train loss and accuracy for each epoch.
            | <*test*> dict: test loss and accuracy for each epoch.
            | <*val*> dict: train validation and accuracy for each epoch.
            
        """
        # Get args
        args = self.args
        saver = args.saver
        log_every = args.log_every
        plot_every = args.plot_every
        save_every = args.save_every
        # Compute splits names
        splits = list(datasets.keys())

        # Setup model
        module = getattr(models, args.model)
        net = getattr(module, "Model")(vars(args))
        # Check resume
        if args.resume is not None:
            net.load_state_dict(Saver.load_state_dict(args.resume))
        # Check for multiple GPUs
        if torch.cuda.device_count() > 1 and args.multi_gpu:
            print(f"Training on {torch.cuda.device_count()} GPUs...")
            net = nn.DataParallel(net)
        # Move to device
        net.to(args.device)
        # Add network to params
        add_net_to_params(net, args, 'net')

        # Define weights for WeightedRandomSampler
        if args.train_labels is not None:
            train_labels = args.train_labels
        else:
            train_labels = [datasets['train'][index][1] for index in range(len(datasets['train']))]
        weights = 1/torch.FloatTensor([(torch.LongTensor(train_labels) == label).sum().item() for label in range(args.num_classes)])
        item_weights = weights[torch.LongTensor(train_labels)].to(args.device)
        #item_weights = weights[datasets["train"].y]

        # Setup data loader
        loaders = {
            s: DataLoader(datasets[s], batch_size=args.batch_size,
                shuffle=False,
                sampler=(WeightedRandomSampler(item_weights, len(datasets[s])) if s == 'train' else None),
                num_workers=(args.workers if not args.overfit_batch else 0),
                drop_last=(s=='train'))
            for s in splits
        }

        # Optimizer params
        optim_params = {'lr': args.lr, 'weight_decay': args.weight_decay}
        if args.optim == 'Adam':
            optim_params = {**optim_params, 'betas': (0.9, 0.999)}
        elif args.optim == 'SGD':
            optim_params = {**optim_params, 'momentum': 0.9}
        # Create optimizer
        optim_class = getattr(torch.optim, args.optim)
        optim = optim_class(params=[param for param in net.parameters() if param.requires_grad], **optim_params)
        # Configure LR scheduler
        scheduler = None
        if args.reduce_lr_every is not None:
            print("Setting up LR scheduler")
            scheduler = torch.optim.lr_scheduler.StepLR(
                optim, args.reduce_lr_every, args.reduce_lr_factor
            )

        # Initialize output metrics
        result_metrics = {s: {} for s in splits}
        # Initialize final performance measures
        max_test_accuracy = -1
        max_val_accuracy = -1
        lowest_train_loss = 3
        test_accuracy_at_lowest_train = 0
        test_acc_at_max_val_acc = -1
        # Process each epoch
        try:
            # Initialize epoch confusion matrix
            conf_matrix = torch.LongTensor(args.num_classes, args.num_classes).to(args.device)
            for epoch in range(args.epochs):
                # Process each split
                for split in splits:
                    # Epoch metrics
                    epoch_metrics = {}
                    # Set network mode
                    if split == 'train':
                        net.train()
                        torch.set_grad_enabled(True)
                    elif epoch >= args.eval_after:
                        net.eval()
                        torch.set_grad_enabled(False)
                    else:
                        break
                    # Initialize confusion matrix
                    conf_matrix.fill_(0)
                    # Process each batch
                    dl = loaders[split]
                    pbar = tqdm(dl, leave=False)
                    for batch_idx, (inputs,labels) in enumerate(pbar):
                        # Compute step
                        step = (epoch * len(dl)) + batch_idx
                        # Set progress bar description
                        pbar_desc = f'{split}, epoch {epoch+1}'
                        if split == 'train':
                            pbar_desc += f', step {step}'
                        pbar.set_description(pbar_desc)
                        # Move to device
                        inputs = inputs.to(args.device)
                        labels = labels.to(args.device)
                        # Training monitoring: setup histogram logging
                        if args.log_histograms and split == 'train' and step % plot_every == 0:
                            setup_forward_hooks(net)
                        # Model-specific forward
                        if args.model == 'eeg_transformer':
                            # move batch size
                            inputs = inputs.permute(2, 0, 1)
                            seq_len = inputs.size(0)
                            outputs = net(inputs, generate_square_subsequent_mask(seq_len).to(args.device))
                        elif args.model == 'graph_attention':
                            # Forward step with graph attention model pre-training (without updating GP Adapter parameters)
                            outputs = net(inputs, pre_train=(args.pre_train and epoch < args.pre_train_epochs))
                        else:
                            # Forward step
                            outputs = net(inputs)
                        # Check NaN
                        if torch.isnan(outputs).any():
                            raise FloatingPointError('Found NaN values')
                        # Compute loss
                        #print(outputs.size(), labels.size())
                        loss = F.cross_entropy(outputs, labels)
                        # Optimize
                        if split == 'train':
                            optim.zero_grad()
                            loss.backward()
                            optim.step()
                        # Compute accuracy
                        preds = torch.argmax(outputs, dim=1)
                        accuracy = (preds == labels).sum().item()/inputs.shape[0]
                        # Update confusion matrix
                        for t, p in zip(labels.view(-1), preds.view(-1)):
                            conf_matrix[t, p] += 1
                        # Initialize metrics
                        metrics = {'loss': loss.item(),
                                   'accuracy': accuracy,
                        }
                        # Add metrics to epoch results
                        for k, v in metrics.items():
                            v *= inputs.shape[0]
                            epoch_metrics[k] = epoch_metrics[k] + [v] if k in epoch_metrics else [v]
                        # Log metrics
                        if step % log_every == 0:
                            for k, v in metrics.items():
                                saver.dump_metric(v, step, split, k, 'batch')
                        # Plot stuff
                        if step % plot_every == 0:
                            # Log inputs
                            #inputs_view = inputs.detach().cpu()
                            #inputs_view = (inputs_view - inputs_view.min())/(inputs_view.max() - inputs_view.min())
                            #saver.dump_batch_image(norm_01(inputs), step, split, 'inputs')
                            # Check log histograms
                            if args.log_histograms and split == 'train':
                                # Log output histograms
                                log_histograms(net, step, saver, True)
                    # Epoch end: compute epoch metrics
                    num_samples = len(dl.dataset) if not dl.drop_last else len(dl)*dl.batch_size
                    epoch_loss = sum(epoch_metrics['loss'])/num_samples
                    epoch_accuracy = sum(epoch_metrics['accuracy'])/num_samples
                    # Check strange loss values
                    if epoch_loss > 1000000000:
                        raise SystemExit(1)
                    # Print to screen
                    pbar.close()
                    print(f'{split}, {epoch+1}: loss={epoch_loss:.4f}, accuracy={epoch_accuracy:.4f}')
                    # Dump to saver
                    saver.dump_metric(epoch_loss, epoch, split, 'loss', 'epoch')
                    saver.dump_metric(epoch_accuracy, epoch, split, 'accuracy', 'epoch')
                    # Log confusion matrix
                    saver.log_confusion_matrix(conf_matrix, epoch, split)
                    # Add to output results
                    result_metrics[split]['loss'] = result_metrics[split]['loss'] + [epoch_loss] if 'loss' in result_metrics[split] else [epoch_loss]
                    result_metrics[split]['accuracy'] = result_metrics[split]['accuracy'] + [epoch_accuracy] if 'accuracy' in result_metrics[split] else [epoch_accuracy]
                # Checks for measuring final performance
                if result_metrics['test']['accuracy'][-1] > max_test_accuracy:
                    max_test_accuracy = result_metrics['test']['accuracy'][-1]
                    # Dump to saver
                    saver.dump_metric(max_test_accuracy, epoch, 'test', 'max accuracy', 'epoch')
                if result_metrics['train']['loss'][-1] < lowest_train_loss:
                    lowest_train_loss = result_metrics['train']['loss'][-1]
                    test_accuracy_at_lowest_train = result_metrics['test']['accuracy'][-1]
                    # Dump to saver
                    saver.dump_metric(lowest_train_loss, epoch, 'train', 'lowest loss', 'epoch')
                    saver.dump_metric(test_accuracy_at_lowest_train, epoch, 'test', 'accuracy at lowest train', 'epoch')
                if 'val' in result_metrics and result_metrics['val']['accuracy'][-1] > max_val_accuracy:
                    max_val_accuracy = result_metrics['val']['accuracy'][-1]
                    test_acc_at_max_val_acc = result_metrics['test']['accuracy'][-1]
                    # Dump to saver
                    saver.dump_metric(test_acc_at_max_val_acc, epoch, 'test', 'acc_at_max_val_acc', 'epoch')
                # Check LR scheduler
                if scheduler is not None:
                    scheduler.step()
                # Save checkpoint
                if epoch % save_every == 0:
                    saver.save_model(net, args.model, epoch)
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            pass
        except FloatingPointError as err:
            print(f'Error: {err}')
        # Print main metrics
        print(f'Max test accuracy:      {max_test_accuracy:.4f}')
        print(f'Max val. accuracy:      {max_val_accuracy:.4f}')
        print(f'Max val. test accuracy: {test_acc_at_max_val_acc:.4f}')
        # Save to file
        with open(f'{args.tag}.txt', 'a') as fp:
            fp.write(f'Max test accuracy:      {max_test_accuracy:.4f}\n')
            fp.write(f'Max val. accuracy:      {max_val_accuracy:.4f}\n')
            fp.write(f'Max val. test accuracy: {test_acc_at_max_val_acc:.4f}\n')
        # Return
        return net, result_metrics

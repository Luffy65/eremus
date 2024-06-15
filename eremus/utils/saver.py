import subprocess
from datetime import datetime
import os
from os.path import getmtime
from pathlib import Path
from time import time
from typing import Union
try:
    import comet_ml
    has_cml = True
except:
    has_cml = False
from misc import image_tensor_to_grid
from matplotlib import pyplot as plt
import sys
import ast
import cv2
import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
    has_tb = True
except:
    has_tb = False
import torchvision.utils
import torchvision.transforms.functional as TF


class Saver(object):
    """
    Saver allows for saving and restore networks.
    
    Arguments
    ------------
    base_output_dir : pathlib.Path
        Path to output directory, in which checkpoints and other data will be saved.
    args : dict
        A dictionary containing the following keys:
        
        | **Experiment Options**
        | <*cometml_api_key_path*> str : path to the *cometml_api_key.txt* file, with comet APIs key.
        | <*cometml_workspace*> str : the comet workspace.
        | <*cometml_project*> str : the comet project (inside the specified workspace).
        
        | **Desired Hyperparameters**
        | <*batch_size*> int : the batch size.
        | <*trainer*> str : the trainer module to use(e.g. ..//trainers//trainer_eremus).
        | <*lr*> float : the learning rate.
        | <*weight_decay*> double : the weight decay.
        | <*optim*> str : set Ooptimizer to *SGD* or *Adam*.
        | <*reduce_lr_every*> int : If not None, reduce learning rate every *reduce_lr_every* epochs of *reduce_lr_factor*.
        | <*reduce_lr_factor*> float : reduce learning rate every *reduce_lr_every* epochs of *reduce_lr_factor*. 
        | <*momentum*> float : the momentum to use in optimization algorithm
        | <*epochs*> int : set the number of training ephocs.
        | <*seed*> int : a integer seed used in random number generation.
        | <*device*> str : set device to *cpu* or *cuda*.
        | <*multi_gpu*> bool : if available and True, use multiple GPUs.
        | <*overfit_batch*> bool : if True, set the dataset to a small sample number (single batch per split). 
        
        | **Other Useful Information**
        | <*dataset*> str : the used dataset. Specify it as *some_random_name/real_dataset_name*.
        | <*fold*> str : while in k-fold cross validation, specify the fold.
        | <*model*> str : the neural network module (e.g. ra_cnn). 
        | <*subject*> int : while in subject training, specify the subject ID.
    sub_dirs : Union[list, tuple]
        List of sub-directories to create. Use one sub-dir for split. (e.g. ['train', 'test', 'val'])
    tag : str
        A tag associated with the experiment. **Tips**: use a meaningful tag.
    """
    def __init__(
        self, base_output_dir: Path, args: dict,
        sub_dirs=('train', 'test'), tag=''):

        # Store args
        self.args = args
        # Create experiment directory
        timestamp_str = datetime.fromtimestamp(time()).strftime('%Y_%m_%d_%H_%M_%S')
        if isinstance(tag, str) and len(tag) > 0:
            # Append tag
            timestamp_str += f"_{tag}"
        self.path = base_output_dir / f'{timestamp_str}'
        self.path.mkdir(parents=True, exist_ok=True)

        # Setup loggers
        self.tb = None
        self.cml = None
        if has_tb:
            self.tb = SummaryWriter(str(self.path))
        if has_cml and args.cometml_api_key_path is not None and \
                       args.cometml_workspace is not None and \
                       args.cometml_project is not None:
            # Read API key
            with open(args.cometml_api_key_path, 'r') as file:
                api_key = file.read().strip()
            # Read project and workspace
            cometml_project = args.cometml_project.strip()
            cometml_workspace = args.cometml_workspace.strip()
            # Create experiment
            self.cml = comet_ml.Experiment(
                api_key=api_key, project_name=cometml_project,
                workspace=cometml_workspace, parse_args=False
            )
            self.cml.set_name(tag)
        # Warnings
        if self.tb is None and self.cml is None:
            print('Saver: warning: no logger')
        else:
            if self.tb is not None:
                print('Saver: using TensorBoard')
            if self.cml is not None:
                print('Saver: using CometML')
        # Create checkpoint sub-directory
        self.ckpt_path = self.path / 'ckpt'
        self.ckpt_path.mkdir(parents=True, exist_ok=True)
        # Create output sub-directories
        self.sub_dirs = sub_dirs
        self.output_path = {}
        for s in self.sub_dirs:
            self.output_path[s] = self.path / 'output' / s
        for d in self.output_path.values():
            d.mkdir(parents=True, exist_ok=False)
        # Dump experiment hyper-params
        with open(self.path / 'hyperparams.txt', mode='wt') as f:
            args_str = [f'{a}: {v}\n' for a, v in self.args.__dict__.items()]
            args_str.append(f'exp_name: {timestamp_str}\n')
            f.writelines(sorted(args_str))
        # Dump some hyperparams on comet
        desired_hyperparams=["batch_size",
                "trainer",
                "optim",
                "lr",
                "reduce_lr_every",
                "reduce_lr_factor",
                "weight_decay",
                "momentum",
                "epochs",
                "seed",
                "device",
                "multi_gpu",
                "overfit_batch"]
        hyperparams={hp: self.args.__dict__[hp] for hp in desired_hyperparams}
        if self.cml is not None:
            self.cml.log_parameters(hyperparams)
        # Dumb other additional informations on comet
        other_useful_infos=["dataset",
            "fold",
            "model",
            "subject"
            ]
        others={ot: self.args.__dict__[ot] for ot in other_useful_infos}
        others["dataset_name"]=os.path.split(others["dataset"])[1]
        if self.cml is not None:
            self.cml.log_others(others)
        
        # Dump command
        with open(self.path / 'command.txt', mode='wt') as f:
            cmd_args = ' '.join(sys.argv)
            f.write(cmd_args)
            f.write('\n')
            if self.cml is not None:
                self.cml.log_other("command", cmd_args)
        # Dump the `git log` and `git diff`. In this way one can checkout
        #  the last commit, add the diff and should be in the same state.
        for cmd in ['log', 'diff']:
            with open(self.path / f'git_{cmd}.txt', mode='wt') as f:
                subprocess.run(['git', cmd], stdout=f)
        # Dump model's code on cometml
        if self.cml is not None:
            with open(os.path.dirname(__file__) + '\\..\\models\\' + self.args.__dict__["model"] + ".py") as file:
                code_str = file.read()
                self.cml.set_code(code_str,overwrite=True, filename=self.args.__dict__["model"]+".py")

    def save_data(self, data, name: str):
        """
        Save generic data.
        """
        torch.save(data, self.path / f'{name}.pth')

    def save_model(self, net: torch.nn.Module, name: str, epoch: int):
        """
        Save model parameters in the checkpoint directory.
        """
        # Get state dict
        state_dict = net.state_dict()
        # Copy to CPU
        for k,v in state_dict.items():
            state_dict[k] = v.cpu()
        # Save
        torch.save(state_dict, self.ckpt_path / f'{name}_{epoch:05d}.pth')
        
    def add_graph(self,model,images):
        if self.tb is not None:
            self.tb.add_graph(model,images)
            
    def dump_batch_image(self, image: torch.FloatTensor, epoch: int, split: str, name: str):
        """
        Dump image batch into folder (as grid) and tb
        TODO: something's wrong with the whole BGR2RGB stuff, we shouldn't need it
        """
        assert split in self.sub_dirs
        assert len(image.shape) == 4, f'shape {image.shape} differs from BxCxHxW format'
        assert image.min() >= 0 and image.max() <= 1, 'image must be between 0 and 1!'

        out_image_path = self.output_path[split] / f'{epoch:05d}_{name}.jpg'
        image_rolled = torchvision.utils.make_grid(image.cpu(), nrow=8, pad_value=1) #, normalize=True, scale_each=True)
        # Save image file
        TF.to_pil_image(image_rolled).save(out_image_path)
        # TensorBoard
        if self.tb is not None:
            self.tb.add_image(f'{split}/{name}', image_rolled, epoch)
        # CometML
        if self.cml is not None:
            self.cml.log_image(image_rolled, name=f'{split}/{name}', step=epoch,
                image_channels='first')

    def dump_batch_video(self, video: torch.FloatTensor, epoch: int, split: str, name: str):
        """
        Dump video batch into folder (as grid) and tb
        FIXME: not sure this works
        """
        assert split in self.sub_dirs
        assert len(video.shape) == 5, f'shape {video.shape} differs from BxTxCxHxW format'
        assert video.min() >= 0 and video.max() <= 1, 'video must be between 0 and 1!'
        out_image_path = self.output_path[split] / f'{epoch:05d}_{name}.jpg'
        video_rolled = video_tensor_to_grid(video, return_image=True)
        cv2.imwrite(str(out_image_path), np.transpose(video_rolled, (1, 2, 0)))
        if self.tb is not None:
            self.tb.add_video(name, video, epoch, fps=5)
    
    def dump_line(self, line, step, split, name=None, fmt=None, labels=None):
        """
        Dump line as matplotlib figure into folder and tb
        TODO: test CometML
        """
        # Line data
        fig = plt.figure()
        if isinstance(line, tuple):
            line_x, line_y = line
            line_x = line_x.cpu().detach()
            line_y = line_y.cpy().detach()
        else:
            line_x = torch.arange(line.numel())
            line_y = line.cpu().detach()
        # kwargs
        kwargs = {}
        if fmt is not None: kwargs['fmt'] = fmt
        # Plot
        plt.plot(line_x, line_y, **kwargs)
        # Ticks
        if labels is not None:
            pass
            #plt.xticks(line_x, labels, rotation='vertical', fontsize=4)
            #plt.margins(0.9)
            #plt.subplots_adjust(bottom=0.8)
        # Save
        if name is not None:
            assert split in self.sub_dirs
            out_path = self.output_path[split] / f'line_{step:08d}_{name.replace("/", "_")}.jpg'
            plt.savefig(out_path)
        if self.tb is not None:
            self.tb.add_figure(f'{split}/{name}' if name is not None else split, fig, step)
        if self.cml is not None:
            self.cml.log_figure(figure_name=f'{split}/{name}' if name is not None else split, figure=fig, step=step)

    def dump_histogram(self, tensor: torch.Tensor, epoch: int, desc: str):
        """
        TODO: disabled for CometML, too slow
        """
        values = tensor.contiguous().view(-1)
        if self.tb is not None:
            #try:
            self.tb.add_histogram(desc, values, epoch)
            #except:
            #print('Error writing histogram')
        #if self.cml is not None:
        #    self.cml.log_histogram_3d(values, desc, epoch)
    
    def dump_metric(self, value: float, epoch: int, *tags):
        if self.tb is not None:
            self.tb.add_scalar('/'.join(tags), value, epoch)
        if self.cml is not None:
            self.cml.log_metric('/'.join(tags), value, step=epoch)
    
    def log_confusion_matrix(self, conf_matrix, epoch, split):
        if self.cml is not None:
            self.cml.log_confusion_matrix(matrix=conf_matrix.tolist(), title=f"Confusion Matrix, {split}, Epoch {epoch}", file_name=f"{split}-confusion-matrix-{epoch:03}.json")
        else:
            raise NotImplementedError('Only CometML is supported for logging confusion matrices')
    
    @staticmethod
    def load_hyperparams(hyperparams_path):
        """
        Load hyperparams from file. Tries to convert them to best type.
        """
        # Process input
        hyperparams_path = Path(hyperparams_path)
        if not hyperparams_path.exists():
            raise OSError('Please provide a valid path')
        if hyperparams_path.is_dir():
            hyperparams_path = os.path.join(hyperparams_path, 'hyperparams.txt')
        # Prepare output
        output = {}
        # Read file
        with open(hyperparams_path) as file:
            # Read lines
            for l in file:
                # Remove new line
                l = l.strip()
                # Separate name from value
                toks = l.split(':')
                name = toks[0]
                value = ':'.join(toks[1:]).strip()
                # Parse value
                try:
                    value = ast.literal_eval(value)
                except:
                    pass
                # Add to output
                output[name] = value
        # Return
        return output

    @staticmethod
    def load_state_dict(model_path: Union[str, Path], verbose: bool = True):
        """
        Load state dict from pre-trained checkpoint. In case a directory is given as `model_path`, the last modified checkpoint is loaded.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise OSError('Please provide a valid path for restoring weights.')

        if model_path.is_dir():
            # Check there are files in that directory
            file_list = sorted(model_path.glob('*.pth'), key=getmtime)
            if len(file_list) == 0:
                # Check there are files in the 'ckpt' subdirectory
                model_path = model_path / 'ckpt'
                file_list = sorted(model_path.glob('*.pth'), key=getmtime)
                if len(file_list) == 0:
                    raise OSError("Couldn't find pth file.")
            checkpoint = file_list[-1]
        elif model_path.is_file():
            checkpoint = model_path

        if verbose:
            print(f'Loading pre-trained weight from {checkpoint}...')

        return torch.load(checkpoint)

    def close(self):
        if self.tb is not None:
            self.tb.close()

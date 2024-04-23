# crop-health-model

This file goes over a few useful things to know about how to configure the entire training process including the data and the model, as well as how to run the training and evaluation processes. Finally, instructions are given on how to generate a model archive (MAR) file that can be provided to TorchServe.

## Main model configurations

To train a model, start by configuring the data, model and training through the `config.yaml` file inside the `configs` folder. Currently, three main model configurations are defined. 

### Binary classifier

The first model is a binary classifier, predicting whether a crop is healthy or sick. To use this configuration, the following fields need to be defined in the following way:
````
fit.model.model.init_args.num_classes: 2
fit.data.task: binary
````
Additionally, all loggers in `fit.trainer.logger` need to have `name` set to `binary`.

### Multiclass classifier with a single healthy class

The second model is a multiclass classifier with a single healthy class, predicting whether a crop is healthy or afflicted by one of several diseases. To use this configuration, the following fields need to be defined in the following way:
````
fit.model.model.init_args.num_classes: 13
fit.data.task: single-HLT
````
Additionally, all loggers in `fit.trainer.logger` need to have `name` set to `single_HLT`.

### Multiclass classifier with multiple healthy classes

The third model is a multiclass classifier with several healthy classes (one for each crop type), predicting whether a crop of a specific type is healthy or afflicted by one of several diseases. To use this configuration, the following fields need to be defined in the following way:
````
fit.model.model.init_args.num_classes: 17
fit.data.task: multi-HLT
````
Additionally, all loggers in `fit.trainer.logger` need to have `name` set to `multi_HLT`.

## Additional configuration

A lot of additional elements within the data, the model and training process can be configured.

### Data configuration

Several aspects of the data can be configured. For example:
- The data can be configured to use a specific data split through `fit.data.data_split`. A value of `[0.8, 0.2]` will first split the entire dataset into a train and test set of proportions 80% and 20%, respectively. The train data will then be split a second time with the same 80/20 split, to become the final train and validation sets.
- The number of workers to use in the `DataLoader` through `fit.data.num_workers`.
- The transforms to apply to the training and test/validation data through `fit.data.train_transforms` and `fit.data.test_transforms`, respectively.
- The normalization to apply to apply to the data (which will be the same for both the training and test/validation data) through `fit.data.normalization`

During a development stage, the `fit.data.limit` can be set to a positive integer, such as `1000`, to limit the size of the entire dataset to 1000 images,

### Model configuration

The PyTorch model itself can be configured by specifying the class path and any potential initialization arguments. For example, to use the custom `ResNet` model, we set `fit.model.mode.class_path` to `crop_health_model.models.model.ResNet`. For this model, we can also specify the use of pre-trained weights by setting `weights` to `DEFAULT`, the number of layers by setting `num_layers` to (for example) `18`, and the number of output features by setting `num_classes` to (for example) `2`.

Currently, only a ResNet model has been defined for use in this project, but other models can be easily added. Note that if more models are added, they should be split into their respective files, such as `resnet.py`, `densenet.py`, etc.. inside the `models` folder. This is because torch-model-archiver needs to receive a script containing a single `nn.Module` corresponding to the appropriate model defintion.

### Training configuration

The training itself is also highly configurable. In our case, we set `fit.trainer.deterministic` to `true` in order to have reproducible experiments. We also set the maximum number of epochs to train for through `fit.train.max_epochs`. Which device(s) to use can also be configured through `fit.trainer.devices`. Its default value is `auto`, but it can for example be set to `[0,1]` if the training should be done on two GPUs.

Existing PyTorch Lightning callbacks as well as custom callbacks can be added through `fit.training.callbacks`. Loggers can also be added through `fit.train.logger`.

The optimizer and learning rate scheduler can be set through the `fit.optimizer` and `fit.lr_scheduler` fields.

## Launch training and evaluation

To run a training process, run the following command from the root of this project:
```
python3 crop_health_model/scripts/train.py --config crop_health_model/configs/config.yaml fit
```

To evaluate a model on the validation set, identify the model version and the filename of its checkpoint, then run the following from the root of this project:
```
python3 crop_health_model/scripts/train.py validate --config tb_logs/multi_HLT/version_0/config.yaml --ckpt_path tb_logs/multi_HLT/version_0/checkpoints/crop_health_model-epoch=5-step=7260-val_loss=0.181.ckpt --trainer.devices=1
```
Here we set the number of devices to be one to avoid getting a warning about using distributed dataloaders in a validation/test phase.

Similarly, to evaluate the model above on the test set, run the following:
```
python3 crop_health_model/scripts/train.py test --config tb_logs/multi_HLT/version_0/config.yaml --ckpt_path tb_logs/multi_HLT/version_0/checkpoints/crop_health_model-epoch=5-step=7260-val_loss=0.181.ckpt --trainer.devices=1
```

## Create a model archive (MAR) file

To create a model archive file which can be used by TorchServe, simple navigate to the folder of the specific model and version (in this case single-HLT version 2) to archive and run:
```
torch-model-archiver --model-name single_HLT --version 2.0 --model-file model_script.py --serialized-file checkpoints/best_model.pt --handler model_handler.py --extra-files index_to_name.json
```

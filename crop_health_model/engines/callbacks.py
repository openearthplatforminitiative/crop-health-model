import json
import math
import os

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import Callback


class ImagePredictionLogger(Callback):
    """Callback to log image predictions during validation to TensorBoard."""

    def __init__(self, num_samples: int = 32) -> None:
        super().__init__()
        self.num_samples = num_samples

    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called when the validation loop begins."""
        # Get the first batch of validation data
        val_samples = next(iter(trainer.datamodule.val_dataloader()))
        val_imgs = val_samples[0].to(device=pl_module.device)
        val_labels = val_samples[1].to(device=pl_module.device)

        # Get model prediction
        log_probs = pl_module(val_imgs)
        preds = torch.argmax(log_probs, dim=-1)

        # Determine grid size
        num_images = min(self.num_samples, len(val_imgs))
        num_cols = int(math.sqrt(num_images))
        num_rows = math.ceil(num_images / num_cols)

        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows)
        )
        fig.subplots_adjust(hspace=0.3, wspace=0.1)

        for i, ax in enumerate(axes.flatten()):
            if i < num_images:
                img = val_imgs[i].cpu().numpy()
                img = np.transpose(img, (1, 2, 0))
                img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
                ax.imshow(img)
                ax.set_title(
                    f"Pred: {preds[i].item()}, Label: {val_labels[i].item()}",
                    fontsize=10,
                )
                ax.axis("off")
            else:
                ax.axis("off")  # Hide unused subplots

        # Use the logger tied to the Trainer
        if trainer.logger:
            # This assumes you have a logger that can log matplotlib figures directly.
            trainer.logger.experiment.add_figure(
                "Validation Predictions", fig, global_step=trainer.current_epoch
            )

        plt.close(fig)


class ClassWeightsCallback(Callback):
    """Callback to set class weights in the model."""
    
    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called when the training starts."""
        weights = trainer.datamodule.compute_class_weights().to(pl_module.device)
        pl_module.set_class_weights(weights)
        print("Class weights set in the model.")


class SaveDataDictionaryCallback(Callback):
    """Callback to save a dictionary from the LightningDataModule as a JSON file."""

    def __init__(self, dict_name: str, filename: str) -> None:
        """
        Args:
            dict_name (str): Attribute name of the dictionary in the LightningDataModule.
            filename (str): Filename to save the dictionary as.
        """
        self.dict_name = dict_name
        self.filename = filename

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called when the training starts."""

        # Access the dictionary from the LightningDataModule
        dataset = trainer.datamodule.data
        dict_to_save = getattr(dataset, self.dict_name, None)

        if dict_to_save is not None:
            # Invert the dictionary
            dict_to_save = {value: key for key, value in dict_to_save.items()}
            # Build the path using trainer's log_dir
            filepath = os.path.join(trainer.log_dir, self.filename)
            # save as JSON file
            with open(filepath + ".json", "w") as f:
                json.dump(dict_to_save, f)


class SaveModelScriptCallback(Callback):
    """Callback to save the model script with updated hyperparameters.

    This callback is useful when you want to save the model script with the hyperparameters used for training.
    Then, the script can be used with torch-model-archiver to easily create a TorchServe model archive.
    """

    def __init__(self, filename: str, template_path: str) -> None:
        """
        Args:
            filepath (str): Path to save the updated model script.
            template_path (str): Path to the model template script.
        """
        self.filename = filename
        self.template_path = template_path

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called when the train starts."""
        # Get the hyperparameters from the model
        hyperparameters = pl_module.model.get_hyperparameters()
        filepath = os.path.join(trainer.log_dir, self.filename)

        # Save the model script based on hyperparameters
        self.save_model_script(hyperparameters, filepath)

    def save_model_script(self, hyperparameters: dict, filepath: str) -> None:
        """Saves the model script based on the provided hyperparameters."""
        with open(self.template_path, "r", encoding="utf-8") as file:
            content = file.read()

        for key, value in hyperparameters.items():
            # This assumes `key` directly matches the string in the template.
            content = content.replace(
                f"{key}: {type(value).__name__}",
                f"{key}: {type(value).__name__} = {value}",
                1,  # Replace only the first occurrence
            )

        # Write the updated content back to the specified filepath
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(content)

        print(f"Updated model script saved to {filepath}")


class SaveModelHandlerCallback(Callback):
    """Callback to generate a handler script when training starts.

    This callback is useful when you want to generate a handler script for TorchServe when training starts.
    The handler only defines the image transform pipeline based on the configuration file.
    """

    def __init__(self, filename: str, config_path: str) -> None:
        """
        Args:
            filename (str): The filename to save the generated handler script.
            config_path (str): The path to the YAML configuration file.
        """
        self.filename = filename
        self.config_path = config_path

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called when the train starts to generate the handler script."""

        # Generate the handler script
        filepath = os.path.join(trainer.log_dir, self.filename)
        self.generate_handler_script(filepath)

    def generate_handler_script(self, filepath: str) -> None:
        """Reads the YAML config and generates a handler script."""
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)

        test_transforms = config["fit"]["data"].get("test_transforms", [])
        normalization = config["fit"]["data"].get("normalization", None)

        transform_lines = [
            f"        {self.get_transform_code(transform)},"
            for transform in test_transforms
        ]
        transform_lines.append("        transforms.ToTensor(),")
        if normalization is not None:
            transform_lines.append(f"        {self.get_transform_code(normalization)},")

        transform_pipeline_code = "\n".join(transform_lines)

        handler_script = f"""
from torchvision import transforms
from ts.torch_handler.image_classifier import ImageClassifier

class CustomHandler(ImageClassifier):
    image_processing = transforms.Compose([
{transform_pipeline_code}
    ])
"""
        with open(filepath, "w") as f:
            f.write(handler_script)
        print(f"Handler script saved to {filepath}")

    def get_transform_code(self, transform: dict) -> str:
        """Generates the Python code for a transform."""
        class_path = transform["class_path"].split(".")[-1]
        init_args = ", ".join(f"{k}={v}" for k, v in transform["init_args"].items())
        return f"transforms.{class_path}({init_args})"


class SaveSimplifiedCheckpoint(Callback):
    """Callback to save a simplified checkpoint with only the model's state_dict."""

    def __init__(self, filename) -> None:
        """
        Args:
            save_path (str): Directory where the simplified checkpoints will be saved.
        """
        self.filename = filename

    def on_save_checkpoint(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict
    ) -> None:
        """Called when saving a checkpoint."""
        # Extract the state_dict from the checkpoint
        state_dict = checkpoint["state_dict"]

        # remove "model." prefix from the keys because the underlying PyTorch model
        # is wrapped by the LightningModule
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        # define the directory and create it if it doesn't exist
        dir_path = f"{trainer.logger.log_dir}/checkpoints"
        os.makedirs(dir_path, exist_ok=True)

        # Define the path for the simplified checkpoint
        file_path = f"{dir_path}/{self.filename}.pt"

        # Save only the state_dict to a new checkpoint file
        torch.save(state_dict, file_path)

        print(f"Simplified checkpoint saved to: {file_path}")

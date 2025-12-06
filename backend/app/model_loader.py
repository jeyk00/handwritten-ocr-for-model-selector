import torch
from huggingface_hub import hf_hub_download
from .model_arch import MyModel

def load_model():
    """
    Downloads the model weights from Hugging Face and initializes the model.
    Returns:
        model (MyModel): The loaded PyTorch model in eval mode.
    """
    repo_id = "Artificial-Intelligence-in-Medicine-AGH/handwritten-digits-recognizer"
    filename = "model.pth"

    # Download the model file
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)

    # Initialize the model architecture
    model = MyModel()

    # Load the state dict
    # map_location='cpu' is a safe default for inference, can be changed if GPU is needed
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    # Set to eval mode
    model.eval()

    return model

import esm
import torch
import nsp3.models as module_arch

from nsp3.main import get_instance

def export(experiment: dict, weights: str):
    model = get_instance(module_arch, 'arch', experiment)

    weights = torch.load(weights)
    model.load_state_dict(weights['state_dict'])

    model.eval()

    example_input = [("protein", "GLLQVATERYVGDEIERQLDDYGLGDVVNPTTPGALHINFSILCTYSMHEHRMPVEPPDV")]

    traced_model = torch.jit.trace(model, batch_tokens)
    traced_model.save("deployment/export.pt")

    print("Model exported succesfully. Saved as deployment/export.pt")
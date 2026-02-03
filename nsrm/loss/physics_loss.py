import torch

def eikonal_loss(model_output, coordinates):
    """
    Computes the Eikonal gradient penalty.
    Ideally, the gradient of an SDF w.r.t coordinates should have a norm of 1.
    """
    # Calculate gradients of the SDF output w.r.t. the input coordinates
    gradients = torch.autograd.grad(
        outputs=model_output,
        inputs=coordinates,
        grad_outputs=torch.ones_like(model_output),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Calculate the L2 norm of the gradients
    grad_norm = gradients.norm(2, dim=-1)
    
    # Penalty: How far is the gradient norm from 1?
    loss = ((grad_norm - 1.0) ** 2).mean()
    
    return loss

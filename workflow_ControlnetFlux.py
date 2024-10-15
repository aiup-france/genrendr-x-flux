from .nodes import ApplyFluxControlNet, XlabsSampler,LoadFluxControlNet
from core.config import parse_config 


def controlnetFlux(image, model, pos_conditioning, neg_conditioning,seed,latent_image,steps ):
    
    args = parse_config()
    #load control flux

    controlnet = LoadFluxControlNet().loadmodel(model_name="flux-dev", controlnet_path=args.controlnet_flux)
    


    #apply 
    controlnet_condition= ApplyFluxControlNet().prepare(controlnet, image, strength=0.70, controlnet_condition = None)


    #xsampler

    latent=XlabsSampler().sampling(model, pos_conditioning, neg_conditioning,
                     seed, steps=steps, timestep_to_start_cfg=1, true_gs=3.5,
                     image_to_image_strength=0.00, denoise_strength=1.00,
                     latent_image=latent_image, controlnet_condition=controlnet_condition)
    
    
    return latent 
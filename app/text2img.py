import re
import time
import inspect
import json
import sys
import base64
from PIL import Image
from io import BytesIO
from fastapi.encoders import jsonable_encoder
import torch
import diffusers


##################################################
# Utils

def pil_to_b64(input):
    buffer = BytesIO()
    input.save( buffer, 'PNG' )
    output = base64.b64encode( buffer.getvalue() ).decode( 'utf-8' ).replace( '\n', '' )
    buffer.close()
    return output

def b64_to_pil(input):
    output = Image.open( BytesIO( base64.b64decode( input ) ) )
    return output

def get_compute_platform(context):
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available() and context == 'engine':
            return 'mps'
        else:
            return 'cpu'
    except ImportError:
        return 'cpu'

##################################################
# Engines

class Engine(object):
    def __init__(self):
        pass

    def process(self, kwargs):
        return []

class EngineStableDiffusion(Engine):
    def __init__(self, pipe, sibling=None, custom_model_path=None, requires_safety_checker=True):
        super().__init__()
        if sibling == None:
            self.engine = pipe.from_pretrained( 'runwayml/stable-diffusion-v1-5', use_auth_token=hf_token.strip() )
        elif custom_model_path:
            if requires_safety_checker:
                self.engine = diffusers.StableDiffusionPipeline.from_pretrained(custom_model_path,
                                                                                safety_checker=sibling.engine.safety_checker,
                                                                                feature_extractor=sibling.engine.feature_extractor)
            else:
                self.engine = diffusers.StableDiffusionPipeline.from_pretrained(custom_model_path,
                                                                                feature_extractor=sibling.engine.feature_extractor)
        else:
            self.engine = pipe(
                vae=sibling.engine.vae,
                text_encoder=sibling.engine.text_encoder,
                tokenizer=sibling.engine.tokenizer,
                unet=sibling.engine.unet,
                scheduler=sibling.engine.scheduler,
                safety_checker=sibling.engine.safety_checker,
                feature_extractor=sibling.engine.feature_extractor
            )
        self.engine.to( get_compute_platform('engine') )

    def process(self, kwargs):
        output = self.engine( **kwargs )
        return {'image': output.images[0], 'nsfw':output.nsfw_content_detected[0]}

class EngineManager(object):
    def __init__(self):
        self.engines = {}

    def has_engine(self, name):
        return ( name in self.engines )

    def add_engine(self, name, engine):
        if self.has_engine( name ):
            return False
        self.engines[ name ] = engine
        return True

    def get_engine(self, name):
        if not self.has_engine( name ):
            return None
        engine = self.engines[ name ]
        return engine

##################################################
# App

# Load and parse the config file:
try:
    config_file = open ('config.json', 'r')
except:
    sys.exit('config.json not found.')

config = json.loads(config_file.read())

hf_token = config['hf_token']

if (hf_token == None):
    sys.exit('No Hugging Face token found in config.json.')

custom_models = config['custom_models'] if 'custom_models' in config else []

# Initialize engine manager:
manager = EngineManager()

# Add supported engines to manager:
manager.add_engine( 'txt2img', EngineStableDiffusion( diffusers.StableDiffusionPipeline,        sibling=None ) )

def _generate(prompt, negative_prompt, seed, count, num_inference_steps, guidance_scale, eta, width, height):
    # Retrieve engine:
    task = 'txt2img'
    engine = 'txt2img'

    engine = manager.get_engine( engine )

    # Prepare output container:
    output_data = {}

    # Handle request:
    try:
        seed = seed
        count = count
        total_results = []
        for i in range( count ):
            if (seed == 0):
                generator = torch.Generator( device=get_compute_platform('generator') )
            else:
                generator = torch.Generator( device=get_compute_platform('generator') ).manual_seed( seed )
            new_seed = generator.seed()
            prompt = prompt
            args_dict = {
                'prompt' : [ prompt ],
                'negative_prompt' : [ negative_prompt ],
                'num_inference_steps' : num_inference_steps,
                'guidance_scale' : guidance_scale,
                'eta' : eta,
                'generator' : generator
            }
            if (task == 'txt2img'):
                args_dict[ 'width' ] = width
                args_dict[ 'height' ] = height
            # Perform inference:
            pipeline_output = engine.process( args_dict )
            pipeline_output[ 'seed' ] = new_seed
            total_results.append( pipeline_output )
        # Prepare response
        output_data[ 'status' ] = 'success'
        images = []
        for result in total_results:
            images.append({
                'base64' : pil_to_b64( result['image'].convert( 'RGB' ) ),
                'seed' : result['seed'],
                'mime_type': 'image/png',
                'nsfw': result['nsfw']
            })
        output_data[ 'images' ] = images
    except RuntimeError as e:
        output_data[ 'status' ] = 'failure'
        output_data[ 'message' ] = 'A RuntimeError occurred. You probably ran out of GPU memory. Check the server logs for more details.'
        print(str(e))
    return jsonable_encoder(output_data)

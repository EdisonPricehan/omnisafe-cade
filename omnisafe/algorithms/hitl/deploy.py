from omnisafe.algorithms.hitl.hitl_cade import HitlCade, LossType
from omnisafe.algorithms.hitl.perception_infer import PerceptionInfer

import os
from loguru import logger


class HitlCadeDeploy:
    """
    HITL Cade deployment class that contains three components:
    1. HITL Cade algorithm for human-in-the-loop reinforcement learning.
    2. Perception inference engine for semantic segmentation.
    3. Communication protocol for interaction with the SplashDrone4 platform.
    """

    def __init__(
        self,
        model_dir: str,
        model_name: str,
        segmentation_engine_path: str,
        **kwargs
    ):
        # Initialize the HITL Cade algorithm
        self.hitl_cade = HitlCade(
            model_dir=model_dir,
            model_name=model_name,
            eval_episodes=3,
            deterministic=True,
            save_path=os.path.join(model_dir, 'hitl_cade_deploy'),
            save_buffer=True,
            buffer_size=1000,
            retrain_epoch=3,
            loss_type='None',  # Need to specify the loss type
            save_ckpts=True,
            render_mode=None,
            enable_hitl=True,
            enable_retrain=True,
            device='cuda:0',
        )
        logger.info(f'HITL Cade initialized with model {model_name} from {model_dir}.')

        # Initialize the semantic segmentation engine
        self.segmentation_engine = PerceptionInfer(engine_path=segmentation_engine_path)
        logger.info(f'Segmentation engine loaded from {segmentation_engine_path}.')

        # Initialize the communication protocol

    def run(self):
        pass


if __name__ == "__main__":
    # Set constants
    model_dir: str = '../../../examples/models'
    model_name: str = 'epoch-350.pt'
    segmentation_engine_path: str = '/home/orin-nano/Aerial-Fluvial-Semantic-Segmentation/src/models/unet-resnet34-128x128-fp16.trt'

    # Initialize the HITL Cade deployment
    hitl_cade_deploy = HitlCadeDeploy(
        model_dir=model_dir,
        model_name=model_name,
        segmentation_engine_path=segmentation_engine_path
    )

    # Run the HITL Cade inference
    hitl_cade_deploy.run()


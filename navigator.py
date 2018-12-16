from base_navigator import BaseNavigator
import random
import numpy as np


class Navigator(BaseNavigator):
    def __init__(self):
        super(Navigator, self).__init__()

    def navigate(self, start_graph_state, show_info):
        self.graph_state = start_graph_state
        
        while True:
            image_feature = self.get_dummy_image_feature(self.graph_state)
            move = self.random_policy(image_feature)
            if move == 'stop':
                print('Action `stop` is chosen.')
                break
            self.step(move)

            if show_info:
                self.show_state_info(self.graph_state)

    def policy(self, state):
        raise NotImplementedError

    def get_image_feature(self, graph_state):
        raise NotImplementedError

    def random_policy(self, state):
        return random.choice(['forward', 'left', 'right', 'stop'])

    def get_dummy_image_feature(self, graph_state):
        panoid, heading = graph_state
       
        # dummy feature
        image_feature = np.random.randn(100, 464, 128) 

        # rotate the pano feature so the middle is the agent's heading direction
        # `shift_angle` is essential for adjusting to the correct heading
        # please include the following in your own `get_image_feature` function
        shift_angle = 157.5 + self.graph.nodes[panoid].pano_yaw_angle - heading

        width = image_feature.shape[1]
        shift = int(width * shift_angle / 360)
        image_feature = np.roll(image_feature, shift, axis=1)

        return image_feature


if __name__ == '__main__':
    navigator = Navigator()
    navigator.navigate(
        start_graph_state=('sbtZW9Akt4izrxdQRDPwMQ', 209), 
        show_info=True
    )




import reinforcement_learning.task1.britneyworld as bw
import numpy as np
env = bw.Environment(10)


env.reset()
env.display()



env.move_agent('down')
env.move_britney()

#britney_cell_type = environment.map[environment.britney_location[0], environment.britney_location[1]]

#britney_cell_type == 0

env.run('right')

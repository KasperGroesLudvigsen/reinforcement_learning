

import britneyworld

env = britneyworld.Environment(10)

env.reset()
env.display()

env.britney_location[0]

env.move_agent('right')
env.move_britney()

#britney_cell_type = environment.map[environment.britney_location[0], environment.britney_location[1]]

#britney_cell_type == 0

env.run('right')

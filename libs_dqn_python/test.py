import dqn


state_geometry          = dqn.sGeometry()
state_geometry.w        = 8
state_geometry.h        = 8
state_geometry.d        = 3

actions_count           = 5
experience_buffer_size  = 1024

nn = dqn.DQNInterface(state_geometry, actions_count, experience_buffer_size)

nn._print()

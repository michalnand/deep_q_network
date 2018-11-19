import dqn


state_geometry          = dqn.sGeometry()
state_geometry.w        = 8
state_geometry.h        = 8
state_geometry.d        = 3

actions_count           = 5

nn = dqn.DQN("dqn_config.json", state_geometry, actions_count)

nn._print()

print("program done")

#include <iostream>
#include <dqn.h>


std::vector<float> random_state(sGeometry state_geometry)
{
    std::vector<float> state(state_geometry.w*state_geometry.h*state_geometry.d);
    for (unsigned int i = 0; i < state.size(); i++)
        state[i] = (rand()%1000)/1000.0;

    return state;
}

unsigned int random_action(unsigned int actions_count)
{
    return rand()%actions_count;
}

float random_reward()
{
    if ((rand()%8) == 0)
    {
        if ((rand()%2) == 0)
            return 1.0;
        else
            return -1.0;
    }
    else
        return 0;
}

int main()
{
    sGeometry state_geometry;
    state_geometry.w = 1;
    state_geometry.h = 1;
    state_geometry.d = 8;

    unsigned int actions_count = 4;

    std::vector<float> state(state_geometry.w*state_geometry.h*state_geometry.d);

    DQN dqn("network/parameters.json",
            state_geometry,
            actions_count);

    for (unsigned int i = 0; i < 256; i++)
    {
        auto state = random_state(state_geometry);
        dqn.compute_q_values(state);

        auto q_values = dqn.get_q_values();
        auto action = random_action(actions_count);
        auto reward = random_reward();


        if ((i+1)%10 == 0)
        {
            dqn.add_final(state, q_values, action, reward);
        }
        else
        {
            dqn.add(state, q_values, action, reward);
        }

        if (dqn.is_full())
        {
            dqn.learn();
        }

    }


    std::cout << "program dome\n";

    return 0;
}

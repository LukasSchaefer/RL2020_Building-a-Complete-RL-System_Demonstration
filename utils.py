from gym.envs.registration import register
import numpy as np


# register non-slippery/ deterministic FrozenLake environment
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
)

def act_to_str(act: int) -> str:
    """
    Map actions (of FrozenLake environment) to interpretable symbols corresponding to directions

    :param act (int): action to map to string
    :return (str): interpretable action name
    """
    if act == 0:
        return "L"
    elif act == 1:
        return "D"
    elif act == 2:
        return "R"
    elif act == 3:
        return "U"
    else:
        raise ValueError("Invalid action value")

def visualise_q_table(q_table):
    """
    Print q_table in human-readable format

    :param q_table (Dict): q_table in form of a dict mapping (observation, action) pairs to
        q-values
    """
    for key in sorted(q_table.keys()):
        obs, act = key
        act_name = act_to_str(act)
        q_value = q_table[key]
        print(f"Pos={obs}\tAct={act_name}\t->\t{q_value}")

def visualise_policy(q_table):
    """
    Given q_table print greedy policy for each FrozenLake position

    :param q_table (Dict): q_table in form of a dict mapping (observation, action) pairs to
        q-values
    """
    # extract best acts
    act_table = np.zeros((4,4))
    str_table = []
    for row in range(4):
        str_table.append("")
        for col in range(4):
            pos = row * 4 + col
            max_q = None
            max_a = None
            for a in range(4):
                q = q_table[(pos, a)]
                if max_q is None or q > max_q:
                    max_q = q
                    max_a = a
            act_table[row, col] = max_a
            str_table[row] += act_to_str(max_a)
    
    # print best actions in human_readable format
    print("\nAction selection table:")
    for row_str in str_table:
        print(row_str)
    print()

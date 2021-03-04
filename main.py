import numpy as np
import sys


#######################################################################################################
# WEEK 1
#######################################################################################################

def convert_strlist_to_intlist(strlist, delimiter):
    """
    Description: Helper function that converts strings of numbers to python Lists of integers
    Inputs: strlist (String), delimiter (String)
    Outputs: intlist (List of int's)
    """

    l = strlist.strip().split(delimiter)
    intlist = [int(n) for n in l]

    return intlist


def convert_strmatrix_to_numpy_matrix(mat):
    """
    Description: Helper function that converts string matrix into a numpy matrix of int's
    Inputs: mat (String)
    Outputs: np_mat (numpy ndarray)
    """

    mat = mat.strip().split('\n')
    num_list = []
    for s in mat:
        n = [int(x) for x in s.split(' ')]
        num_list.append(n)

    np_mat = np.array(num_list)

    return np_mat


def dpchange(money, coins):
    """
    Description: Computes the smallest amount of coins needed to provide the correct amount of change, 'money'. The
                denominations of coins are given by 'coins'
    Inputs: money (int), coins (List of int's)
    Outputs: min_num_coins (int)
    """

    min_num_coins = {0: 0}
    for m in range(1, money + 1):
        min_num_coins[m] = 1000000
        for i in range(len(coins)):
            if m >= coins[i]:
                if min_num_coins[m - coins[i]] + 1 < min_num_coins[m]:
                    min_num_coins[m] = min_num_coins[m - coins[i]] + 1

    return min_num_coins[money]


def manhattan_tourist(n, m, down, right):
    """
    Description: Dynamic Programming algorithm for finding the longest path in the Manhattan Tourist Problem
    Inputs: n (int), m (int), down (ndarray), right (ndarray)
    Outputs: longest_path_len (int)
    """

    s = np.zeros((n + 1, m + 1))

    for i in range(1, n + 1):
        s[i, 0] = s[i - 1, 0] + down[i - 1, 0]

    for j in range(1, m + 1):
        s[0, j] = s[0, j - 1] + right[0, j - 1]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            downward = s[i - 1, j] + down[i - 1, j]
            rightward = s[i, j - 1] + right[i, j - 1]
            s[i, j] = max(downward, rightward)

    return s[n, m]


def lcs_backtrack(v, w):
    """
    Description: Generates the matrix of backtrack pointers to be used in the longest common subsequence problem
    Inputs: v (String), w (String)
    Outputs: b (matrix of backtrack pointers (len(v) + 1 by len(w) + 1 ndarray of strings))
    """

    v_len = len(v)
    w_len = len(w)

    s = np.zeros((v_len + 1, w_len + 1))
    b = np.zeros((v_len + 1, w_len + 1), dtype=str)
    b[1:, 0] = 'w'
    b[0, 1:] = 'r'

    for i in range(1, v_len + 1):
        for j in range(1, w_len + 1):
            match = 0
            if v[i - 1] == w[j - 1]:
                match = 1
            downward = s[i - 1, j]
            rightward = s[i, j - 1]
            diagonal = s[i - 1, j - 1] + match
            s[i, j] = max(downward, rightward, diagonal)
            if s[i, j] == s[i - 1, j]:
                b[i, j] = 'w'  # w := down-arrow
            elif s[i, j] == s[i, j - 1]:
                b[i, j] = 'r'  # r := right-arrow
            elif s[i, j] == s[i - 1, j - 1] + match:
                b[i, j] = 'g'  # g := diagonal

    return b


def output_lcs(backtrack, v, i, j):
    """
    Description: Recursive function that generates the longest common subsequence (lcs) of the given two strings
    Inputs: backtrack (len(str1) + 1 by len(str2) ndarray of strings, v (String), i (int), j (int)
    Outputs: lcs (String)
    """

    if i == 0 or j == 0:
        return ""
    if backtrack[i, j] == 'w':
        return output_lcs(backtrack, v, i - 1, j)
    elif backtrack[i, j] == 'r':
        return output_lcs(backtrack, v, i, j - 1)
    else:
        return output_lcs(backtrack, v, i - 1, j - 1) + v[i - 1]


def process_edges(edges):
    """
    Description: Helper function for generate_topological_ordering() function
    Inputs: edges (String)
    Outputs: processed_edges (List of tuples of int's)
    """

    edges_cleaned = edges.strip().split('\n')

    processed_edges = []
    for edge in edges_cleaned:
        s = int(edge.split('->')[0])
        e = int(edge.split('->')[1].split(':')[0])
        w = int(edge.split('->')[1].split(':')[1])
        edge_tuple = (s, e, w)
        processed_edges.append(edge_tuple)

    return processed_edges


def generate_topological_ordering(processed_edges):
    """
    Description: Helper function for longest_path_in_a_dag() function
    Inputs: processed_edges (List of tuples of int's)
    Outputs: sorted List of int's
    """

    nodes = []
    for edge in processed_edges:
        for i in range(0, len(edge) - 1):
            if edge[i] not in nodes:
                nodes.append(edge[i])

    return sorted(nodes)


def compute_score(topo_nodes, i, processed_edges, backtrack):
    """
    Description: Helper function for longest_path_in_a_dag() function
    Inputs: topo_nodes (List of int's), i (int), processed_edges (List of tuples of int's),
            backtrack (List of tuples of int's)
    Outputs: tuple (containing an edge and its score)
    """

    score = -1000000
    for edge in processed_edges:
        if edge[1] == topo_nodes[i]:
            predecessor = edge[0]
            predecessor_score = 0
            for edge_backtrack in backtrack:
                if edge_backtrack[1] == predecessor:
                    predecessor_score = edge_backtrack[2]
                    break
            if predecessor_score + edge[2] > score:
                score = predecessor_score + edge[2]
                highlight_edge = edge

    return (highlight_edge[0], highlight_edge[1], score)


def longest_path_backtrack(processed_edges):
    """
    Description: Finds the longest path in a directed acyclic graph (DAG) given the starting and ending nodes as well
                 as a collection of edges with their corresponding weights
    Inputs: processed_edges (List of tuples of int's)
    Outputs: backtrack (List of tuples of int's)
    """

    topo_nodes = generate_topological_ordering(processed_edges)
    backtrack = []
    for i in range(1, len(topo_nodes)):
        f = compute_score(topo_nodes, i, processed_edges, backtrack)
        backtrack.append(f)

    return backtrack


def output_longest_path(backtrack, current_node, seq, start, end):
    """
    Description: Output the sequence of nodes that comprise the longest path in an arbitrary graph
    Inputs: backtrack (List of tuples of int's), current_node (int), seq (List of int's), start (int), end (int)
    Outputs: seq (List of int's)
    """

    if current_node == start:
        seq.append(current_node)
        seq = [n for n in reversed(seq)]
        return seq
    else:
        for edge_backtrack in backtrack:
            if current_node == edge_backtrack[1]:
                seq.append(current_node)
                return output_longest_path(backtrack, edge_backtrack[0], seq, start, end)
    return seq


def format_path_output(path):
    """
    Description: Generates the appropriately formatted string that represents a path in a graph
    Inputs: path (List of int's)
    Outputs: formatted_path (String)
    """

    formatted_path = ''
    for i in range(len(path)):
        formatted_path += str(path[i])
        if i < len(path) - 1:
            formatted_path += '->'

    return formatted_path


if __name__ == '__main__':
    #######################################################################################################
    # WEEK 1
    #######################################################################################################

    #######################################################################################################
    # Dynamic Programming Change Challenge - Given some amount of money and a set of coin denominations
    # generate the minimum amount of coins needs to satisfy the given amount of money
    #######################################################################################################

    # m_arg = 16538
    # c = "5,3,1"
    # c_arg = convert_strlist_to_intlist(c, ',')

    # c_out = dpchange(m_arg, c_arg)
    # print(c_out)

    #######################################################################################################
    # Manhattan Tourist Problem - Given a graph and endpoint, compute the longest path to reach the endpoint
    #######################################################################################################

    #     nm_arg = "15 5"
    #     d_arg = """
    # 1 3 0 3 0 4
    # 2 2 3 0 4 1
    # 4 3 2 4 4 4
    # 1 4 1 4 1 3
    # 4 2 3 2 0 2
    # 1 3 4 0 4 4
    # 0 2 0 0 2 1
    # 2 0 2 4 2 4
    # 0 0 0 0 3 4
    # 3 3 0 0 1 1
    # 4 4 2 0 2 2
    # 2 2 3 2 1 0
    # 4 0 4 0 4 1
    # 3 2 3 3 4 2
    # 3 4 4 3 1 0
    # """
    #
    #     r_arg = """
    # 0 0 0 1 3
    # 3 3 4 3 0
    # 3 4 4 4 3
    # 4 0 1 4 1
    # 2 3 1 3 4
    # 4 3 4 3 3
    # 3 2 3 2 3
    # 2 2 1 3 0
    # 2 1 1 1 3
    # 0 1 3 2 4
    # 0 2 1 2 1
    # 0 1 2 2 4
    # 1 4 3 3 2
    # 0 2 4 4 4
    # 3 3 2 0 3
    # 0 1 4 1 3
    # """
    #
    #     nm_arg = convert_strlist_to_intlist(nm_arg, ' ')
    #     d_arg = convert_strmatrix_to_numpy_matrix(d_arg)
    #     r_arg = convert_strmatrix_to_numpy_matrix(r_arg)
    #
    #     l_out = manhattan_tourist(nm_arg[0], nm_arg[1], d_arg, r_arg)
    #     print(l_out)

    #######################################################################################################
    # Testing out helper function lcs_backtrack() needed for lcs challenge
    #######################################################################################################

    # v_arg = "AACC"
    # w_arg = "ACAC"
    # b_out = lcs_backtrack(v_arg, w_arg)

    #######################################################################################################
    # Longest Common Subsequence (LCS) Challenge - given two strings, generate the longest common subsequence
    # shared by the two strings
    #######################################################################################################

    # s_arg = "AGTAATTTCCCTTTGCTCAACTCGCGCATGACAGTTTACGCTTCTGGTATGCATTTAGCACATCCACTTAGGCGAAGCTATAGCCGAGACGCACTTCCTCGACCGTTTCATTCGGGGCGACAAGTCACAGGCCAGATCGTCTGAGCGGTATGAACAGCACTTGGAGGAATGCAAGGGGGCGAGACATATTCAAGTCGGCATATTAGAACGCAGCAGAACGGCGCACGCTCACAGCGCCTAGAGCCTAGAGGAGTCCTTACATCTTCAACTGTCAACCCCCTGACTTATCATGGTATAACCCAGAACATCCCTGGCTTGTTTGCGCTACTATTAGCCGTGACTGTCTCTAGTTATACTCGTATGATATAAATATGTAGCAAACCTTTACTGTTTAGGCTTTAGTCCATGCCCGGATATAACACGGGACACGTGTAGTGCTAGCGAAAGATGCGCGGATCAGCCAGCCGCCTTCTATCTTCTAACTAAGAACCTCAATCCCTACAATCTAACGAGTATGATCCCTACTGTATCATGCCCAACATAACGCTCAAGGCCCAAGAAGTCACCCAGCCCGGGTGTATTACCGTTCCTACGTTCGAACGGTAACTGAGTCAAGGCGACCGCCTGCTGCCTAGACGTCACGGGCGTAACCTGAGTATCTTCATAACAGGGCGCATGGACCTTTTCTTACATTAATGAATAGGCATCCTGCGATTGGCCGGTCAACCCATGTCGTAGCACTGGTCGGCAAGGCTTTCCGAGTTTGCCAATTTAGTTAGTGCCAGCCCGGCCAATCGCCAGTTCGAAGTCTTGGGGCGCAACGAGCTCCAGCTCGATCGAGGTCGAGTGATTGCTGCTTAGACCCCTCGTTAAGCGGTGAGTTGGCGT"
    # t_arg = "AGAAGAACAAAGAGGCTTCGGATTAAATTGCTACTAGTTATGCGGAAACACTGGGCAGCGAACGTTAACCATACGGAGTTCCGCTACCGGCTCGTGATAGTTTGGATAATATAAGGGCCCACGACAATAGTGATCCCAAACTCAGCAGTGTGATCGCGTCCTCTTTATGGTTAGGCGCGATCGTAGCAAAGTTGAGCTCCCGTATATCCAATATGGGGATTCCGCGGTGCGACGGATCGATGGAAGCTCACCGTGGTGCGTCATGTTCGGACCCGTTAAACGTAGAATAAGCGGTAATGTTACACGAACGTTAGCTGGGAATTGTCAAGTGGAATCTGGAAGAATGGAAGTGCATGTTTGGGCATTGCTACCGTTGCTATACGCCACGAAGGAGACTAGTACGGTTCTGTGTCCCCTGGGTAAGCCCCGATATACAAGGGACAACTGTACGGCAGAACACTTTGGAACTTCCCAGGGGTTCGGACAGGGGCCTAGTATCAGAGCACCTGATGCGACCGTAATGCAGCGACAACCCCCTGGACGCCATGGTCTTACGCGGGAGCATTAGTGGATATGCACCTTGCGTGGCGGTTCTTGCTCGTTCTATACCAGTATGATGGGGGCCTTACGAAGACGCGACGTCGACATTTAGGACTCGAGTATGGCACGGTAACTCTAAGCCATGTATTTACATCAGATAACGCCGCACATCAGTCTTCTCGTCCTTGGAGATCCCGCAACATGTGGCATCAAACAGATATCTCCAGTGTCCGCCGACAAGGACTCGACTGCGGGGGGGTGGCTAGCGAGGGACATTATTTAACCGATAGGAGACACGATCCCAAAGGCCTGGCTTAGCTATATATAATGTTTGAGTTGT"

    # b_out = lcs_backtrack(t_arg, s_arg)
    # print(b_out)
    # print('---')
    # b_out = lcs_backtrack(s_arg, t_arg)

    # sys.setrecursionlimit(1500)
    # lcs_out = output_lcs(b_out, s_arg, len(s_arg), len(t_arg))
    # sys.setrecursionlimit(1000)

    # print(lcs_out)

    #######################################################################################################
    # Longest Path in an (arbitrary) DAG - Given start and ending nodes, generate the longest path found
    # in a graph defined by the given weighted edges; for challenge data set, see 'test_datasets/' directory
    #######################################################################################################

    # s_arg = 0
    # e_arg = 49
    # d = """ [challenge dataset removed from here because it is too large] """

    # d_arg = process_edges(d)
    # b_out = longest_path_backtrack(d_arg)

    # len_path = b_out[len(b_out) - 1][len(b_out[len(b_out) - 1]) - 1]
    # print(len_path)
    # path = output_longest_path(b_out, e_arg, [], s_arg, e_arg)
    # p_out = format_path_output(path)
    # print(p_out)


def get_function(function_id, dimension):
    n_var = dimension

    match (function_id):
        case 1:
            upper_bound = 100
            lower_bound = -100
            optimum_value = 0
        case 2:
            upper_bound = 10
            lower_bound = -10
            optimum_value = 0
        case 3:
            upper_bound = 100
            lower_bound = -100
            optimum_value = 0
        case 4:
            upper_bound = 100
            lower_bound = -100
            optimum_value = 0
        case 5:
            upper_bound = 30
            lower_bound = -30
            optimum_value = 0
        case 6:
            upper_bound = 100
            lower_bound = -100
            optimum_value = 0
        case 7:
            upper_bound = 1.28
            lower_bound = -1.28
            optimum_value = 0
        case 8:
            upper_bound = 500
            lower_bound = -500
            optimum_value = -418.9829 * n_var
        case 9:
            upper_bound = 5.12
            lower_bound = -5.12
            optimum_value = 0
        case 10:
            upper_bound = 32
            lower_bound = -32
            optimum_value = 0
        case 11:
            upper_bound = 600
            lower_bound = -600
            optimum_value = 0
        case 12:
            upper_bound = 50
            lower_bound = -50
            optimum_value = 0
        case 13:
            upper_bound = 50
            lower_bound = -50
            optimum_value = 0
        case 14:
            n_var = 2
            upper_bound = 65
            lower_bound = -65
            optimum_value = 0.998
        case 15:
            n_var = 4
            upper_bound = 5
            lower_bound = -5
            optimum_value = 0.0003
        case 16:
            n_var = 2
            upper_bound = 5
            lower_bound = -5
            optimum_value = -1.0316
        case 17:
            n_var = 2
            upper_bound = 5
            lower_bound = -5
            optimum_value = 0.398
        case 18:
            n_var = 2
            upper_bound = 2
            lower_bound = -2
            optimum_value = 3
        case 19:
            n_var = 3
            upper_bound = 1
            lower_bound = 0
            optimum_value = -3.86
        case 20:
            n_var = 6
            upper_bound = 1
            lower_bound = 0
            optimum_value = -3.32
        case 21:
            n_var = 4
            upper_bound = 10
            lower_bound = 0
            optimum_value = -10.1532
        case 22:
            n_var = 4
            upper_bound = 10
            lower_bound = 0
            optimum_value = -10.4029
        case 23:
            n_var = 4
            upper_bound = 10
            lower_bound = 0
            optimum_value = -10.5364
    return upper_bound, lower_bound, optimum_value

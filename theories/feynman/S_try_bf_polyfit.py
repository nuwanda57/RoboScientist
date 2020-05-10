import shutil

from theories.feynman.S_brute_force import brute_force
from theories.feynman.S_polyclean_file import polyfit
from theories.feynman.S_replace_variables import replace_variables


def try_bf_polyfit(pathdir, filename, methods_tried, BF_try_time, BF_ops_file_type, BF_sep_type, use_MDL,
                   check_prefactor, dim_red_file, maxdeg_polyfit, err_threshold_polyfit, first_run, move_dir,
                   original_dir, solved_dir, BF_transf_name, poly_transf_name):
    if first_run:
        # Try brute force                                                                                                                                                    

        BF_formula, methods_tried = brute_force(pathdir, filename, methods_tried, BF_transf_name, BF_try_time,
                                                BF_ops_file_type, 2, use_MDL, check_prefactor)
        if BF_formula != 0:
            not_replaced_BF_formula = BF_formula
            BF_formula = replace_variables(dim_red_file, filename.split('-')[0], BF_formula)
            if move_dir != 0:
                shutil.move(original_dir + filename.split('-')[0], solved_dir)
            return (BF_formula, methods_tried, not_replaced_BF_formula)

        BF_formula, methods_tried = brute_force(pathdir, filename, methods_tried, BF_transf_name, BF_try_time,
                                                BF_ops_file_type, 3, use_MDL, check_prefactor)
        if BF_formula != 0:
            not_replaced_BF_formula = BF_formula
            BF_formula = replace_variables(dim_red_file, filename.split('-')[0], BF_formula)
            if move_dir != 0:
                shutil.move(original_dir + filename.split('-')[0], solved_dir)
            return (BF_formula, methods_tried, not_replaced_BF_formula)

    else:
        BF_formula, methods_tried = brute_force(pathdir, filename, methods_tried, BF_transf_name, BF_try_time,
                                                BF_ops_file_type, BF_sep_type, use_MDL, check_prefactor)
        if BF_formula != 0:
            not_replaced_BF_formula = BF_formula
            BF_formula = replace_variables(dim_red_file, filename.split('-')[0], BF_formula)
            if move_dir != 0:
                shutil.move(original_dir + filename.split('-')[0], solved_dir)
            return (BF_formula, methods_tried, not_replaced_BF_formula)

    # Try polyfit
    methods_tried = methods_tried + [poly_transf_name]
    poly_solved, obtained_eq = polyfit(maxdeg_polyfit, pathdir + filename, err_threshold_polyfit)
    if poly_solved == 1:
        methods_tried = methods_tried + ["solved"]
        obtained_eq = replace_variables(dim_red_file, filename.split('-')[0], obtained_eq)
        if move_dir != 0:
            shutil.move(original_dir + filename.split('-')[0], solved_dir)
        return (obtained_eq, methods_tried, obtained_eq)

    return (0, methods_tried, 0)

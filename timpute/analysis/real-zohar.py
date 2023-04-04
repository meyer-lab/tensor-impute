from timpute.figures import compare_imputation, generateTensor

# # REAL DATA TESTS

# # TEST 1: imputation missingness
# compare_imputation(impute_type='entry', impute_reps=15, impute_perc=0.05,
#                            tensor=generateTensor(type="zohar"), save="zohar/test1/entry-low")
# compare_imputation(impute_type='chord', impute_reps=15, impute_perc=0.05,
#                            tensor=generateTensor(type="zohar"), save="zohar/test1/chord-low")
# compare_imputation(impute_type='entry', impute_reps=15, impute_perc=0.25,
#                            tensor=generateTensor(type="zohar"), save="zohar/test1/entry-med")
# compare_imputation(impute_type='chord', impute_reps=15, impute_perc=0.25,
#                            tensor=generateTensor(type="zohar"), save="zohar/test1/chord-med")
# compare_imputation(impute_type='entry', impute_reps=15, impute_perc=0.4,
#                            tensor=generateTensor(type="zohar"), save="zohar/test1/entry-high")
# compare_imputation(impute_type='chord', impute_reps=15, impute_perc=0.4,
#                            tensor=generateTensor(type="zohar"), save="zohar/test1/chord-high")

# # TEST 2: initialization
compare_imputation(impute_type='entry', impute_reps=15, init="random",
                           tensor=generateTensor(type="zohar"), save="zohar/test2/entry-random")
compare_imputation(impute_type='chord', impute_reps=15, init="random",
                           tensor=generateTensor(type="zohar"), save="zohar/test2/chord-random")
# compare_imputation(impute_type='entry', impute_reps=15, init="svd",
#                             tensor=generateTensor(type="zohar"), save="zohar/test2/entry-svd")
# compare_imputation(impute_type='chord', impute_reps=15, init="svd",
#                             tensor=generateTensor(type="zohar"), save="zohar/test2/chord-svd")
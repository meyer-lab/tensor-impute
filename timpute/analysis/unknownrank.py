from timpute.figures import compare_imputation, generateTensor

# UNKNOWN RANK SIMULATED DATA TESTS

# TEST 1: imputation missingness
compare_imputation(impute_type='entry', impute_reps=15, impute_perc=0.05,
                           tensor=generateTensor(type="unknown"), save="unknownrank/test1/entry-low")
compare_imputation(impute_type='chord', impute_reps=15, impute_perc=0.05,
                           tensor=generateTensor(type="unknown"), save="unknownrank/test1/chord-low")
compare_imputation(impute_type='entry', impute_reps=15, impute_perc=0.25,
                           tensor=generateTensor(type="unknown"), save="unknownrank/test1/entry-high")
compare_imputation(impute_type='chord', impute_reps=15, impute_perc=0.25,
                           tensor=generateTensor(type="unknown"), save="unknownrank/test1/chord-high")

# TEST 2: initialization
compare_imputation(impute_type='entry', impute_reps=15, init="random",
                           tensor=generateTensor(type="unknown"), save="unknownrank/test2/entry-random")
compare_imputation(impute_type='chord', impute_reps=15, init="random",
                           tensor=generateTensor(type="unknown"), save="unknownrank/test2/chord-random")
compare_imputation(impute_type='entry', impute_reps=15, init="svd",
                            tensor=generateTensor(type="unknown"), save="unknownrank/test2/entry-svd")
compare_imputation(impute_type='chord', impute_reps=15, init="svd",
                            tensor=generateTensor(type="unknown"), save="unknownrank/test2/chord-svd")

# TEST 3: original data missingness & data type
compare_imputation(impute_type='entry', impute_reps=15,
                           tensor=generateTensor(type="unknown", shape=(20,25,30), missingness=0), save="unknownrank/test3/entry-full")
compare_imputation(impute_type='chord', impute_reps=15,
                           tensor=generateTensor(type="unknown", shape=(20,25,30), missingness=0), save="unknownrank/test3/chord-full")
compare_imputation(impute_type='entry', impute_reps=15,
                           tensor=generateTensor(type="unknown", shape=(20,25,30), missingness=0.4), save="unknownrank/test3/entry-missing")
compare_imputation(impute_type='chord', impute_reps=15,
                           tensor=generateTensor(type="unknown", shape=(20,25,30), missingness=0.4), save="unknownrank/test3/chord-missing")
# t3_3e = compare_imputation(impute_type='entry', impute_reps=15,
#                            tensor=generateTensor(type="unknown", shape=(200,250,300), missingness=0), save="unknownrank/test3/entry_large")
# t3_3c = compare_imputation(impute_type='chord', impute_reps=15,
#                            tensor=generateTensor(type="unknown", shape=(200,250,300), missingness=0), save="unknownrank/test3/chord_large")

# TEST 4: original data shape
compare_imputation(impute_type='entry', impute_reps=15,
                           tensor=generateTensor(type="unknown", shape=(50,50,50), missingness=0), save="unknownrank/test4/entry-cube")
compare_imputation(impute_type='chord', impute_reps=15,
                           tensor=generateTensor(type="unknown", shape=(50,50,50), missingness=0), save="unknownrank/test4/chord-cube")
compare_imputation(impute_type='entry', impute_reps=15,
                           tensor=generateTensor(type="unknown", shape=(250,50,10), missingness=0), save="unknownrank/test4/entry-sheet")
compare_imputation(impute_type='chord', impute_reps=15,
                           tensor=generateTensor(type="unknown", shape=(250,50,10), missingness=0), save="unknownrank/test4/chord-sheet")
compare_imputation(impute_type='entry', impute_reps=15,
                           tensor=generateTensor(type="unknown", shape=(1250,10,10), missingness=0), save="unknownrank/test4/entry-string")
compare_imputation(impute_type='chord', impute_reps=15,
                           tensor=generateTensor(type="unknown", shape=(1250,10,10), missingness=0), save="unknownrank/test4/chord-string")
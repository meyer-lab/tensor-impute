from timpute.figures import compare_imputation, generateTensor

# KNOWN RANK SIMULATED DATA TESTS


# TEST 1: imputation missingness
compare_imputation(impute_type='entry', impute_reps=15, impute_perc=0.05, save="knownrank/test1/entry-low")
compare_imputation(impute_type='chord', impute_reps=15, impute_perc=0.05, save="knownrank/test1/chord-low")
compare_imputation(impute_type='entry', impute_reps=15, impute_perc=0.25, save="knownrank/test1/entry-high")
compare_imputation(impute_type='chord', impute_reps=15, impute_perc=0.25, save="knownrank/test1/chord-high")

compare_imputation(impute_type='entry', impute_reps=15, impute_perc=0.05, tensor=generateTensor(r=20), save="knownrank/test1/entry-low-r20")
compare_imputation(impute_type='chord', impute_reps=15, impute_perc=0.05, tensor=generateTensor(r=20), save="knownrank/test1/chord-low-r20")
compare_imputation(impute_type='entry', impute_reps=15, impute_perc=0.25, tensor=generateTensor(r=20), save="knownrank/test1/entry-high-r20")
compare_imputation(impute_type='chord', impute_reps=15, impute_perc=0.25, tensor=generateTensor(r=20), save="knownrank/test1/chord-high-r20")


# TEST 2: initialization
compare_imputation(impute_type='entry', impute_reps=15, init="random", save="knownrank/test2/entry-random")
compare_imputation(impute_type='chord', impute_reps=15, init="random", save="knownrank/test2/chord-random")
compare_imputation(impute_type='entry', impute_reps=15, init="svd", save="knownrank/test2/entry-svd")
compare_imputation(impute_type='chord', impute_reps=15, init="svd", save="knownrank/test2/chord-svd")

compare_imputation(impute_type='entry', impute_reps=15, tensor=generateTensor(r=20), init="random", save="knownrank/test2/entry-random-r20")
compare_imputation(impute_type='chord', impute_reps=15, tensor=generateTensor(r=20), init="random", save="knownrank/test2/chord-random-r20")
compare_imputation(impute_type='entry', impute_reps=15, tensor=generateTensor(r=20), init="svd", save="knownrank/test2/entry-svd-r20")
compare_imputation(impute_type='chord', impute_reps=15, tensor=generateTensor(r=20), init="svd", save="knownrank/test2/chord-svd-r20")


# TEST 3: original data missingness & data type
compare_imputation(impute_type='entry', impute_reps=15,
                           tensor=generateTensor(type="known", shape=(20,25,30), missingness=0), save="knownrank/test3/entry-full")
compare_imputation(impute_type='chord', impute_reps=15,
                           tensor=generateTensor(type="known", shape=(20,25,30), missingness=0), save="knownrank/test3/chord-full")
compare_imputation(impute_type='entry', impute_reps=15,
                           tensor=generateTensor(type="known", shape=(20,25,30), missingness=0.4), save="knownrank/test3/entry-missing")
compare_imputation(impute_type='chord', impute_reps=15,
                           tensor=generateTensor(type="known", shape=(20,25,30), missingness=0.4), save="knownrank/test3/chord-missing")
# t3_3e = compare_imputation(impute_type='entry', impute_reps=15,
#                            tensor=generateTensor(type="known", shape=(200,250,300), missingness=0), save="knownrank/test3/entry-large")
# t3_3c = compare_imputation(impute_type='chord', impute_reps=15,
#                            tensor=generateTensor(type="known", shape=(200,250,300), missingness=0), save="knownrank/test3/chord-large")

compare_imputation(impute_type='entry', impute_reps=15,
                           tensor=generateTensor(type="known", r=20, shape=(20,25,30), missingness=0), save="knownrank/test3/entry-full-r20")
compare_imputation(impute_type='chord', impute_reps=15,
                           tensor=generateTensor(type="known", r=20, shape=(20,25,30), missingness=0), save="knownrank/test3/chord-full-r20")
compare_imputation(impute_type='entry', impute_reps=15,
                           tensor=generateTensor(type="known", r=20, shape=(20,25,30), missingness=0.4), save="knownrank/test3/entry-missing-r20")
compare_imputation(impute_type='chord', impute_reps=15,
                           tensor=generateTensor(type="known", r=20, shape=(20,25,30), missingness=0.4), save="knownrank/test3/chord-missing-r20")
# t3_6e = compare_imputation(impute_type='entry', impute_reps=15,
#                            tensor=generateTensor(type="known", r=20, shape=(200,250,300), missingness=0), save="knownrank/test3/entry-large")
# t3_6c = compare_imputation(impute_type='chord', impute_reps=15,
#                            tensor=generateTensor(type="known", r=20, shape=(200,250,300), missingness=0), save="knownrank/test3/chord-large")

# TEST 4: original data shape
compare_imputation(impute_type='entry', impute_reps=15,
                           tensor=generateTensor(type="known", shape=(50,50,50), missingness=0), save="knownrank/test4/entry-cube")
compare_imputation(impute_type='chord', impute_reps=15,
                           tensor=generateTensor(type="known", shape=(50,50,50), missingness=0), save="knownrank/test4/chord-cube")
compare_imputation(impute_type='entry', impute_reps=15,
                           tensor=generateTensor(type="known", shape=(250,50,10), missingness=0), save="knownrank/test4/entry-sheet")
compare_imputation(impute_type='chord', impute_reps=15,
                           tensor=generateTensor(type="known", shape=(250,50,10), missingness=0), save="knownrank/test4/chord-sheet")
compare_imputation(impute_type='entry', impute_reps=15,
                           tensor=generateTensor(type="known", shape=(1250,10,10), missingness=0), save="knownrank/test4/entry-string")
compare_imputation(impute_type='chord', impute_reps=15,
                           tensor=generateTensor(type="known", shape=(1250,10,10), missingness=0), save="knownrank/test4/chord-string")

# TEST 5: dimensionality
compare_imputation(impute_type='entry', impute_reps=15,
                           tensor=generateTensor(type="known", shape=(40,40,40), missingness=0), save="knownrank/test5/entry-3d")
compare_imputation(impute_type='chord', impute_reps=15,
                           tensor=generateTensor(type="known", shape=(40,40,40), missingness=0), save="knownrank/test5/chord-3d")
compare_imputation(impute_type='entry', impute_reps=15,
                           tensor=generateTensor(type="known", shape=(20,20,20,8), missingness=0), save="knownrank/test5/entry-4d")
compare_imputation(impute_type='chord', impute_reps=15,
                           tensor=generateTensor(type="known", shape=(20,20,10,8), missingness=0), save="knownrank/test5/chord-4d")
compare_imputation(impute_type='entry', impute_reps=15,
                           tensor=generateTensor(type="known", shape=(8,10,10,10,8), missingness=0), save="knownrank/test5/entry-5d")
compare_imputation(impute_type='chord', impute_reps=15,
                           tensor=generateTensor(type="known", shape=(8,10,10,10,8), missingness=0), save="knownrank/test5/chord-5d")
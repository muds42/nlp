# Identifying Nominals with no head match

Paper found here: https://arxiv.org/abs/1710.00936

The best entry into the code can be found in m2_dotprod_nn's main function.  Related functions give some indication as to the range and structure we were able to test.  The interaction between DotProdConfig and DotProdNN classes are where we implemented the range of models we wanted to test.

When we were adding our customized features to the code, we used the preprocessed data and ran custom_feature_append.  Regarding getting the data, we were given this generously from the Stanford coreference group which included both the coreference data and the w2v embeddings.  These files are too large to upload to github, but the structure can be inferred from our data processing code.  We were also provided the code for the baseline (logistic_regression_baseline) by the Stanford group, as a benchmark for our paper.

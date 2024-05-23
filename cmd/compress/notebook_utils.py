
import numpy as np
import scipy.stats as stats
from models.adapters.mobilenet_adapter import MobileNetV2Adapter
import scikit_posthocs as sp
import pandas as pd

metrics = {
    "lenet": {
        "mnist": {
            "test": {
            "top-1": 0.991600,
            "top-5": 1,
            "loss": 0.03202170803940765
            }
        },
        "qmnist": {
            "test50k": {
                "top-1": 0.988660,
                "top-5": 0.999840,
                "loss": 0.04984482934950747
            },
            "nist": {
                "top-1": 0.9938106925621598,
                "top-5": 0.999839,
                "loss": 0.030931375168995457
            }
        }
    },
    "mobilenetv2": {
        "default": {
                "validation": {
                "top-1": 0.71596,
                "top-5": 0.90236,
                "loss": 1.162748210296631
            }
        }
    }
}

indices = dict([(k, i) for i, k in enumerate(MobileNetV2Adapter.expected_weight_count.keys())])

def get_model_metrics(model: str, dataset: str, split: str):
    return metrics[model][dataset][split]
        
def get_mobilenet_output_count(layer: str) -> int:
    return MobileNetV2Adapter.expected_weight_count[layer]

def get_mobilenet_layer_number(layer: str) -> int:
    return indices[layer]

# Take from https://stackoverflow.com/a/40239615
def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient

def examine_mean_hypothesis(df, group1, group2, variable, alternative="less", df_column="Grid"):
    sign = "<" if alternative == "less" else "!=" if alternative == "two-sided" else ">"
    group1_df = df.loc[df[df_column] == group1, variable]
    group2_df = df.loc[df[df_column] == group2, variable]
    levene_stat, levene_p = stats.levene(group1_df, group2_df)
    normal_a_stat = stats.normaltest(group1_df)
    normal_b_stat = stats.normaltest(group2_df)
    if normal_a_stat.pvalue >= 0.05 and normal_b_stat.pvalue >= 0.05:
        result = stats.ttest_ind(group1_df, group2_df, equal_var=levene_p>=0.5, alternative=alternative)
        test = "T-Test"
    else:
        result = stats.mannwhitneyu(group1_df, group2_df, alternative=alternative)
        test = "MWU"    
    print(f"Result: {'HA' if result.pvalue < 0.05 else 'H0'} , Hypotheses: H0: {variable}[{group1}] == {variable}[{group2}]; HA: {variable}[{group1}] {sign} {variable}[{group2}], samples = {len(group1_df.index)}: {test}, {result}")
    return result

def examine_anova(*groups):
    is_normal = all([stats.normaltest(g).pvalue >= 0.05 for g in groups])
    is_homoscedasticity = stats.levene(*groups).pvalue >= 0.05
    var_test = stats.f_oneway if is_normal and is_homoscedasticity else stats.kruskal
    post_hoc = stats.tukey_hsd if is_normal and is_homoscedasticity else sp.posthoc_conover
    result = var_test(*groups)
    print(f"Result: {'HA' if result.pvalue < 0.05 else 'H0'} , Hypotheses: H0: equal; HA: not equal, {result}")
    
    result_hoc = None
    if result.pvalue < 0.05:
        result_hoc = post_hoc(list(groups))
        if not isinstance(result_hoc, pd.DataFrame):
            print(f"Result: {'HA' if result_hoc.pvalue < 0.05 else 'H0'} , Hypotheses: H0: equal; HA: not equal, {result_hoc}")
        
    return result, result_hoc

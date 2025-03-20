import json
from scipy import stats

data = {
    "MobileNetV2": [
        0.27554,
        0.2757,
        0.27582,
        0.27507,
        0.27578,
        0.27513,
        0.27357,
        0.27451,
        0.27491,
        0.27603,
        0.27394,
        0.27544,
    ],
    "MobileNetV3 Small": [
        0.0866,
        0.08663,
        0.08581,
        0.08722,
        0.08672,
        0.08432,
        0.08657,
        0.08607,
        0.08294,
        0.08396,
        0.08599,
        0.08464,
    ],
    "MobileNetV3 Large": [
        0.20279,
        0.21029,
        0.20656,
        0.20986,
        0.20825,
        0.21044,
        0.21072,
        0.20907,
        0.21131,
        0.21019,
        0.20926,
        0.2116,
    ],
}


def check_statistical_significance(data):
    """
    Checks for statistical significance within each list of values.

    Args:
        data (dict): A dictionary where keys are model names and values are lists of latency measurements.
    """
    for model, values in data.items():
        # Perform Shapiro-Wilk test for normality
        shapiro_test = stats.shapiro(values)
        print(f"\n{model}:")
        print(
            f"  Shapiro-Wilk test: statistic={shapiro_test.statistic:.3f}, p-value={shapiro_test.pvalue:.3f}"
        )

        # If p-value > 0.05, assume normality and use Levene test; otherwise, use Kruskal-Wallis test
        if shapiro_test.pvalue > 0.05:
            # Perform Levene test for equal variances
            if len(values) >= 2:
                levene_test = stats.levene(
                    values[: len(values) // 2], values[len(values) // 2 :]
                )
                print(
                    f"  Levene test: statistic={levene_test.statistic:.3f}, p-value={levene_test.pvalue:.3f}"
                )

                # If p-value > 0.05, assume equal variances and use ANOVA; otherwise, use Welch's ANOVA
                if levene_test.pvalue > 0.05:
                    # Perform ANOVA test
                    f_statistic, p_value = stats.f_oneway(
                        values[: len(values) // 2], values[len(values) // 2 :]
                    )
                    print(
                        f"  ANOVA test: F-statistic={f_statistic:.3f}, p-value={p_value:.3f}"
                    )
                    if p_value < 0.05:
                        print(
                            "  There is a statistically significant difference within the values."
                        )
                    else:
                        print(
                            "  There is no statistically significant difference within the values."
                        )
                else:
                    # Perform Welch's ANOVA test
                    welch_test = stats.ttest_ind(
                        values[: len(values) // 2],
                        values[len(values) // 2 :],
                        equal_var=False,
                    )
                    print(
                        f"  Welch's t-test: statistic={welch_test.statistic:.3f}, p-value={welch_test.pvalue:.3f}"
                    )
                    if welch_test.pvalue < 0.05:
                        print(
                            "  There is a statistically significant difference within the values."
                        )
                    else:
                        print(
                            "  There is no statistically significant difference within the values."
                        )
            else:
                print("  Not enough data points to perform Levene's test.")
                print(
                    "  There is no statistically significant difference within the values."
                )
        else:
            # Perform Kruskal-Wallis test
            if len(values) >= 2:
                kruskal_test = stats.kruskal(
                    values[: len(values) // 2], values[len(values) // 2 :]
                )
                print(
                    f"  Kruskal-Wallis test: statistic={kruskal_test.statistic:.3f}, p-value={kruskal_test.pvalue:.3f}"
                )
                if kruskal_test.pvalue < 0.05:
                    print(
                        "  There is a statistically significant difference within the values."
                    )
                else:
                    print(
                        "  There is no statistically significant difference within the values."
                    )
            else:
                print("  Not enough data points to perform Kruskal-Wallis test.")
                print(
                    "  There is no statistically significant difference within the values."
                )


# Load data from JSON file
# with open('your_data.json', 'r') as f:
#    data = json.load(f)

check_statistical_significance(data)

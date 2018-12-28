```{.python .input}
import numpy as np
from matplotlib import pyplot as plt
```

```{.python .input}
########################################
# Configs
########################################

rate = .3
num_people = 1000

# Default assumed error rate
error_rate = .2

# Per-group error rates
error_rate_a = .3
error_rate_b = .4

# Default value for num_tests if not specified 
num_tests = 5

# Per-group number of tests
num_tests_a = 5
num_tests_b = 5

threshold = .5


########################################
# Some Utility Functions
########################################
def logodds2prob(logodds):
    odds = np.exp(logodds)
    return (odds / (1+ odds))

def metrics(scores, thresh, talents):
    preds = scores > thresh
    acc = np.mean(preds == talents)
    tp = np.mean((preds == 1)*(talents == 1))
    fp = np.mean((preds == 1)*(talents == 0))
    fn = np.mean((preds == 0)*(talents == 1))
    tn = np.mean((preds == 0)*(talents == 0))
    pr = tp / (tp + fp)
    return (acc, tp, tn, fp, fn, pr)
```

```{.python .input}
def simulate_group(rate=rate, believed_rate=rate, num_people=num_people, 
                   num_tests=num_tests, error_rate=error_rate):
    ################################################
    #  Initializing the talents based on the rate
    ################################################
    talents = np.random.binomial(1, rate, num_people)

    ################################################
    #  Simulating the tests given the error rate
    ################################################
    tests = np.zeros((num_people, num_tests))
    for p in range(num_people):
        for t in range(num_tests):
            if np.random.rand() <= error_rate:
                tests[p,t] = 1 - talents[p]
            else:
                tests[p,t] = talents[p]

    #################################################
    # DEBUGGING CODE  --- test that the disagreement 
    # between tests and talents is roughly equal to 
    # the error rate of the tests
    #################################################
    # np.sum(np.abs(tests.T - talents)) / 10000

    log_odds = np.ones(num_people) * np.log(believed_rate / (1 - believed_rate))
    successes = np.sum(tests, axis=1)
    post_log_odds = log_odds + \
        successes * ( np.log(1-error_rate) - np.log(error_rate)) + \
        (num_tests-successes) * (np.log(error_rate) - np.log(1 - error_rate))

    post_probs = logodds2prob(post_log_odds) 
    return (talents, tests, successes, post_probs)
```

```{.python .input}
talentsA, testsA, succA, postA = simulate_group(error_rate=error_rate_a, num_tests=num_tests_a)
talentsB, testsB, succB, postB = simulate_group(error_rate=error_rate_b, num_tests=num_tests_b)

plt.hist(succA, alpha=.5, bins=50)
plt.hist(succB, alpha=.5, bins=50)
plt.xlabel("# of successes")
plt.ylabel("frequency")
plt.show()

plt.hist(postA, alpha=.5, bins=20)
plt.hist(postB, alpha=.5, bins=20)
plt.xlabel("P(skilled)")
plt.ylabel("frequency")
plt.show()
```

```{.python .input}
################################################################
#   Set threshold based on inferred probability of "skill"
################################################################
# skill_threshold_a = skill_threshold_b = .5
skill_threshold_a = .5
skill_threshold_b = .5

accA, tpA, tnA, fpA, fnA, prA = metrics(postA, skill_threshold_a, talentsA)
accB, tpB, tnB, fpB, fnB, prB = metrics(postB, skill_threshold_b, talentsB)

print("Metrics in group A: Acc:", accA, "Pr:", prA, "TP:", tpA, "TN:", tnA, "FP:", fpA, "FN:", fnA)
print("Metrics in group B: Acc:", accB, "Pr:", prB, "TP:", tpB, "TN:", tnB, "FP:", fpB, "FN:", fnB)
```

```{.python .input}
################################################################
#   Set threshold based on percent successes
################################################################
# success_threshold_a = success_threshold_b = .5
success_threshold_a = .5
success_threshold_b = .5

accA, tpA, tnA, fpA, fnA, prA = metrics(succA / num_tests_a, success_threshold_a, talentsA)
accB, tpB, tnB, fpB, fnB, prB = metrics(succB / num_tests_b, success_threshold_b, talentsB)
print("########################## \n" + \
      "# Setting thresholds based on percent success \n" + \
      "########################## \n")
      
print("Metrics in group A: Acc:", accA, "Pr:", prA, "TP:", tpA, "TN:", tnA, "FP:", fpA, "FN:", fnA)
print("Metrics in group B: Acc:", accB, "Pr:", prB, "TP:", tpB, "TN:", tnB, "FP:", fpB, "FN:", fnB)

```

```{.python .input}
################################################################
#    Sketch of toy idea:
#        What if the employer draws wrong conlusion?
#        Sees different false positive rates among the hired
#           (say, when setting a threshold based 
#           on inferred prob candidate is skilled)
#        Then the employer updates their prior to believe this group
#        ...is worse than the majority group by this amount (updating prior odds)
################################################################

# for i in range(1000):
#     print("*** ITERATION ", i, " ***")
#     believed_rateA = tpA / (tpA + fpA)
#     believed_rateB = tpB / (tpB + fpB)
#     print("believed_rateA, ", believed_rateA)
#     print("believed_rateB, ", believed_rateB)

#     talentsA, testsA, succA, postA = simulate_group(believed_rate=believed_rateA, error_rate=error_rate_a)
#     talentsB, testsB, succB, postB = simulate_group(believed_rate=believed_rateB, error_rate=error_rate_b)

# #     plt.hist(succA, alpha=.5, bins=50)
# #     plt.hist(succB, alpha=.5, bins=50)
# #     plt.xlabel("# of successes")
# #     plt.ylabel("frequency")
# #     plt.show()

# #     plt.hist(postA, alpha=.5, bins=20)
# #     plt.hist(postB, alpha=.5, bins=20)
# #     plt.xlabel("P(skilled)")
# #     plt.ylabel("frequency")
# #     plt.show()

#     accA, tpA, tnA, fpA, fnA = metrics(postA, threshold, talentsA)
#     accB, tpB, tnB, fpB, fnB = metrics(postB, threshold, talentsB)

#     print("Metrics in group A: Acc:", accA, "TP:", tpA, "TN:", tnA, "FP:", fpA, "FN:", fnA)
#     print("Metrics in group B: Acc:", accB, "TP:", tpB, "TN:", tnB, "FP:", fpB, "FN:", fnB)

```

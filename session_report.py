import sys
from glob import glob
import pandas as pd


def subject_correct(df):
    resp = df.responded
    return df.loc[resp, "correct"].mean()


def optimal_correct(df):
    resp = df.responded
    return ((df.trial_llr > 0) == df.target)[resp].mean()


def performance(df):
    return ("{:.1%} correct (optimal: {:.1%})"
            .format(subject_correct(df), optimal_correct(df)))


if __name__ == "__main__":

    _, sess_dir = sys.argv

    parts = sess_dir.split("/")
    subject = parts[-3]
    session = parts[-2]

    print("\nSubject: {} | Session: {}\n".format(subject, session))

    trial_files = glob(sess_dir + "/*_trials.csv")

    dfs = []
    for run, fname in enumerate(sorted(trial_files), 1):

        df = pd.read_csv(fname)
        dfs.append(df)
        print(" Run {}: {}".format(run, performance(df)))

    df = pd.concat(dfs)
    print("-" * 38)
    print(" Total: {}\n".format(performance(df)))

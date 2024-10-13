import os

import matplotlib.pyplot as plt
import pandas as pd


class Scorer:
    class Score:
        def __init__(self, experiment: str, group: str, directory="base", experiments_path=os.path.join(".", "..", "experiments"), metric="trueskill"):
            self.experiment = experiment
            self.experiment_dir = os.path.join(experiments_path, directory, "runs", experiment)
            self.group = group
            self.metric = metric
            self.score_df = self._get_score_df()

        def _get_score_df(self) -> pd.DataFrame:
            result = []
            for file in os.listdir(self.experiment_dir):
                if not file.endswith(".csv"):
                    continue
                df = pd.read_csv(os.path.join(self.experiment_dir, file))
                time = os.path.splitext(file)[0]
                model = "models/" + self.experiment + "/" + time + ".pt"
                result.append({"time": int(time), "score": df[df["name"] == model][self.metric].values[0]})
            result = pd.DataFrame(result)
            result = result.sort_values("time", ignore_index=True)
            return result

    def __init__(self, metric="trueskill"):
        self.metric = metric
        self.scores: list[Scorer.Score] = []
        self.scores_avg: dict[str, pd.DataFrame] = {}

    def add_score(self, experiment: str, group: str, **score_kwargs) -> 'Scorer':
        score = self.Score(experiment, group, **score_kwargs)
        if score.metric != self.metric:
            raise ValueError("Metric mismatch")
        self.scores.append(score)
        return self

    def add_many_scores(self, experiments: list[str], group: str, **score_kwargs) -> 'Scorer':
        for experiment in experiments:
            self.add_score(experiment, group, **score_kwargs)
        return self

    def get_group(self, group: str) -> list['Scorer.Score']:
        scores = []
        for score in self.scores:
            if score.group == group:
                scores.append(score)
        return scores

    def average_scores(self) -> 'Scorer':
        for group in dict.fromkeys([score.group for score in self.scores]).keys():
            scores = self.get_group(group)
            result = pd.concat([score.score_df for score in scores])
            result = result.groupby("time").mean().reset_index()
            self.scores_avg[group] = result
        return self

    def draw_avg(self, title: str, step_limit: float = None, x_line: float = None, file: str = None, file_dir="plots") -> 'Scorer':
        self.average_scores()
        df = None
        for group, score in self.scores_avg.items():
            if step_limit is not None:
                score = score[score["time"] <= step_limit]
            score = score.rename(columns={"score": group})
            if df is None:
                df = score
            else:
                df = pd.merge(df, score, on="time", how="outer", sort=True)
                df = df.fillna((df.ffill() + df.bfill()) / 2)
        df.plot(x="time", y=[group for group in self.scores_avg.keys()], color={k: f"C{i}" for i, k in enumerate(self.scores_avg.keys())}, title=title)
        if len(self.scores_avg) == 1:
            plt.legend([])
        else:
            plt.legend(loc='upper left')
        if x_line is not None:
            plt.axvline(x=x_line, color='black')
            plt.xticks(list(plt.xticks()[0][1:-1]) + [x_line])
        plt.ylabel(self.metric + " metric")
        plt.xlabel("training step")
        plt.grid()
        if file is not None:
            os.makedirs(file_dir, exist_ok=True)
            plt.savefig(os.path.join(file_dir, file + ".png"), bbox_inches='tight')
        plt.show()
        return self
